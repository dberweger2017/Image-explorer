# src/vectorize_images.py
import os
import sqlite3
import hashlib
import time
import numpy as np
import hnswlib
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch

from config import (
    MODEL_ID,
    DEVICE,
    IMAGE_SOURCE_DIR,
    DB_PATH,
    VECTOR_INDEX_PATH,
    BATCH_SIZE,
    IMAGE_EXTENSIONS,
    PROCESSED_DATA_DIR,
)

# --- Database Setup ---
def init_db():
    """Initializes the SQLite database and creates tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Store relative path from IMAGE_SOURCE_DIR
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            relative_path TEXT UNIQUE NOT NULL,
            file_hash TEXT NOT NULL,
            last_modified REAL NOT NULL,
            vectorized_at REAL NOT NULL,
            vector_index_id INTEGER UNIQUE
        )
    """
    )
    conn.commit()
    conn.close()
    print("Database initialized.")

def get_file_hash(filepath):
    """Computes SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_image_info_from_db():
    """Fetches all image info (relative_path, hash, last_modified, vector_index_id) from DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT relative_path, file_hash, last_modified, vector_index_id FROM images")
    # Using a dictionary for faster lookups by relative_path
    db_images = {
        row[0]: {"hash": row[1], "last_modified": row[2], "vector_index_id": row[3]}
        for row in cursor.fetchall()
    }
    conn.close()
    return db_images

def get_images_on_disk():
    """Scans IMAGE_SOURCE_DIR for images and returns their info."""
    disk_images = {}
    for root, _, files in os.walk(IMAGE_SOURCE_DIR):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, IMAGE_SOURCE_DIR)
                try:
                    stat = os.stat(full_path)
                    disk_images[relative_path] = {
                        "full_path": full_path,
                        "last_modified": stat.st_mtime,
                        "hash": None, # To be computed if needed
                    }
                except FileNotFoundError:
                    print(f"Warning: File not found during scan: {full_path}")
                    continue
    return disk_images

# --- Model Loading ---
def load_model_and_processor():
    """Loads the SigLIP model and processor."""
    print(f"Loading model: {MODEL_ID} onto {DEVICE}...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
        model.eval() # Set to evaluation mode
        print("Model loaded successfully.")
        return processor, model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have an internet connection for the first download,")
        print("and that the model ID is correct and you have enough VRAM if using GPU.")
        exit()

# --- Vectorization ---
@torch.no_grad() # Disable gradient calculations for inference
def vectorize_batch(image_paths, processor, model):
    """Vectorizes a batch of images."""
    try:
        images_pil = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = processor(images=images_pil, return_tensors="pt", padding=True).to(DEVICE)
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy()
    except Exception as e:
        print(f"Error vectorizing batch (first image: {image_paths[0] if image_paths else 'N/A'}): {e}")
        return [None] * len(image_paths)


# --- HNSW Index Management ---
def load_hnsw_index(model_dim, index_path, initial_max_elements=1000, ef_construction=200, M=16):
    """Loads HNSWlib index if it exists, otherwise creates a new one."""
    # Using 'ip' for inner product, which is equivalent to cosine similarity for normalized vectors
    index = hnswlib.Index(space='ip', dim=model_dim)

    if os.path.exists(index_path):
        print(f"Loading existing HNSW index from {index_path}...")
        try:
            index.load_index(index_path)
            # load_index in Python bindings infers max_elements from the file
            print(f"HNSW index loaded. Max elements: {index.max_elements_}, "
                  f"Current count: {index.get_current_count()}, Dimension: {index.dim}")
            if index.dim != model_dim:
                 print(f"Warning: HNSW index dimension ({index.dim}) "
                       f"differs from model dimension ({model_dim}). Recreating.")
                 if os.path.exists(index_path): os.remove(index_path)
                 index = hnswlib.Index(space='ip', dim=model_dim) # Re-init
                 index.init_index(max_elements=initial_max_elements, ef_construction=ef_construction, M=M)
                 print("New HNSW index initialized after dimension mismatch.")
        except Exception as e:
            print(f"Error loading HNSW index: {e}. Recreating.")
            if os.path.exists(index_path): os.remove(index_path)
            index.init_index(max_elements=initial_max_elements, ef_construction=ef_construction, M=M)
            print(f"New HNSW index initialized after error. Max elements: {index.max_elements_}")
    else:
        print("Creating new HNSW index...")
        index.init_index(max_elements=initial_max_elements, ef_construction=ef_construction, M=M)
        print(f"New HNSW index initialized. Max elements: {index.max_elements_}")
    return index

def save_hnsw_index(index, index_path):
    """Saves the HNSWlib index to disk."""
    print(f"Saving HNSW index to {index_path} with {index.get_current_count()} active elements "
          f"(max elements: {index.max_elements_})...")
    index.save_index(index_path)
    print("HNSW index saved.")

# --- Main Processing Logic ---
def main():
    init_db()
    processor, model = load_model_and_processor()
    model_dim = model.config.projection_dim

    # Estimate initial capacity for new index based on disk images, or use a default
    # This is only used if the index is created from scratch.
    # HNSWlib can resize, but pre-allocating can be more efficient.
    # For simplicity, we use a fixed default `initial_max_elements` in `load_hnsw_index`.
    # A more dynamic value could be `max(1000, len(get_images_on_disk()))`
    hnsw_index = load_hnsw_index(model_dim, VECTOR_INDEX_PATH, initial_max_elements=10000)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("Scanning images on disk...")
    disk_images_info = get_images_on_disk()
    print(f"Found {len(disk_images_info)} images on disk.")

    print("Fetching processed image info from database...")
    db_images_info = get_image_info_from_db()
    print(f"Found {len(db_images_info)} images in database.")

    disk_paths = set(disk_images_info.keys())
    db_paths = set(db_images_info.keys())

    new_paths = list(disk_paths - db_paths)
    deleted_paths = list(db_paths - disk_paths)
    existing_paths = list(disk_paths.intersection(db_paths))

    images_to_vectorize_paths = []
    images_to_vectorize_rel_paths = []
    images_to_vectorize_metadata = [] # (rel_path, hash, last_modified)
    
    total_vectors_processed_this_run = 0

    print(f"Found {len(new_paths)} new images.")
    for rel_path in new_paths:
        info = disk_images_info[rel_path]
        file_hash = get_file_hash(info["full_path"])
        images_to_vectorize_paths.append(info["full_path"])
        images_to_vectorize_rel_paths.append(rel_path)
        images_to_vectorize_metadata.append((rel_path, file_hash, info["last_modified"]))

    print(f"Checking {len(existing_paths)} existing images for modifications...")
    modified_count = 0
    for rel_path in existing_paths:
        disk_info = disk_images_info[rel_path]
        db_info = db_images_info[rel_path]

        if abs(disk_info["last_modified"] - db_info["last_modified"]) > 1e-6:
            current_hash = get_file_hash(disk_info["full_path"])
            if current_hash != db_info["hash"]:
                print(f"Image modified (hash mismatch): {rel_path}")
                images_to_vectorize_paths.append(disk_info["full_path"])
                images_to_vectorize_rel_paths.append(rel_path)
                images_to_vectorize_metadata.append((rel_path, current_hash, disk_info["last_modified"]))
                modified_count += 1
                
                if db_info["vector_index_id"] is not None:
                    try:
                        # Mark the old vector as deleted in HNSWlib
                        hnsw_index.mark_deleted(db_info["vector_index_id"])
                        print(f"  Marked old HNSW ID {db_info['vector_index_id']} as deleted for modified image.")
                    except Exception as e: # HNSWlib might raise if ID not found
                        print(f"  Warning: Could not mark HNSW ID {db_info['vector_index_id']} as deleted: {e}.")
                # Mark old DB entry for vector_index_id update later (it will get the same ID back)
                cursor.execute("UPDATE images SET vector_index_id = NULL WHERE relative_path = ?", (rel_path,))
    conn.commit() # Commit after potential NULLing of vector_index_id
    print(f"Found {modified_count} modified images to re-vectorize.")

    if deleted_paths:
        print(f"Processing {len(deleted_paths)} deleted images...")
        deleted_ids_marked_count = 0
        for rel_path in deleted_paths:
            image_db_info = db_images_info.get(rel_path) # Get full info
            if image_db_info and image_db_info["vector_index_id"] is not None:
                try:
                    hnsw_index.mark_deleted(image_db_info["vector_index_id"])
                    deleted_ids_marked_count += 1
                    print(f"  Marked HNSW ID {image_db_info['vector_index_id']} for {rel_path} as deleted.")
                except Exception as e:
                    print(f"  Warning: Could not mark HNSW ID {image_db_info['vector_index_id']} for {rel_path} as deleted: {e}.")
            
            cursor.execute("DELETE FROM images WHERE relative_path = ?", (rel_path,))
            print(f"  Removed from DB: {rel_path}")
        conn.commit()
        if deleted_ids_marked_count > 0:
             print(f"  Marked {deleted_ids_marked_count} IDs as deleted in HNSW index.")


    if images_to_vectorize_paths:
        print(f"Vectorizing {len(images_to_vectorize_paths)} new/modified images in batches of {BATCH_SIZE}...")
        
        cursor.execute("SELECT MAX(id) FROM images")
        max_db_id_row = cursor.fetchone()
        current_max_db_id = max_db_id_row[0] if max_db_id_row and max_db_id_row[0] is not None else 0

        # Ensure HNSW index can accommodate new items. add_items handles resizing if needed.
        # We could proactively resize here if adding a very large number of items at once:
        # current_capacity = hnsw_index.max_elements_
        # required_capacity = hnsw_index.get_current_count() + len(images_to_vectorize_paths) # A rough estimate
        # if required_capacity > current_capacity:
        #     print(f"Resizing HNSW index from {current_capacity} to {required_capacity * 1.2}") # Add some buffer
        #     hnsw_index.resize_index(int(required_capacity 

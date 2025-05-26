import os
import sqlite3
import hashlib
import time
from datetime import datetime
import numpy as np
import faiss
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch # PyTorch is usually a dependency for Hugging Face Transformers

from config import (
    MODEL_ID,
    DEVICE,
    IMAGE_SOURCE_DIR,
    DB_PATH,
    FAISS_INDEX_PATH,
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
    # Store faiss_id which is the direct ID used in the FAISS index
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            relative_path TEXT UNIQUE NOT NULL,
            file_hash TEXT NOT NULL,
            last_modified REAL NOT NULL,
            vectorized_at REAL NOT NULL,
            faiss_id INTEGER UNIQUE
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
    """Fetches all image info (relative_path, hash, last_modified, faiss_id) from DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT relative_path, file_hash, last_modified, faiss_id FROM images")
    # Using a dictionary for faster lookups by relative_path
    db_images = {
        row[0]: {"hash": row[1], "last_modified": row[2], "faiss_id": row[3]}
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
        # Normalize embeddings for cosine similarity (common practice)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy()
    except Exception as e:
        print(f"Error vectorizing batch (first image: {image_paths[0] if image_paths else 'N/A'}): {e}")
        # Return None or empty list to indicate failure for this batch
        return [None] * len(image_paths)


# --- FAISS Index Management ---
def load_faiss_index(model_dim):
    """Loads FAISS index if it exists, otherwise creates a new one."""
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index...")
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
            # Sanity check for dimension, though IndexIO doesn't store it directly
            if index.d != model_dim:
                print(f"Warning: FAISS index dimension ({index.d}) "
                      f"differs from model dimension ({model_dim}). Recreating.")
                index = faiss.IndexIDMap(faiss.IndexFlatL2(model_dim))
            else:
                 print(f"FAISS index loaded. Dimension: {index.d}, Total vectors: {index.ntotal}")

        except Exception as e:
            print(f"Error loading FAISS index: {e}. Recreating.")
            # Use IndexIDMap to allow adding vectors with specific IDs and removing them
            index = faiss.IndexIDMap(faiss.IndexFlatL2(model_dim))
    else:
        print("Creating new FAISS index...")
        index = faiss.IndexIDMap(faiss.IndexFlatL2(model_dim)) # L2 distance
        # For cosine similarity with normalized vectors, IndexFlatIP (Inner Product) is equivalent to L2
        # index = faiss.IndexIDMap(faiss.IndexFlatIP(model_dim))
    return index

def save_faiss_index(index):
    """Saves the FAISS index to disk."""
    print(f"Saving FAISS index to {FAISS_INDEX_PATH} with {index.ntotal} vectors...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("FAISS index saved.")

# --- Main Processing Logic ---
def main():
    init_db()
    processor, model = load_model_and_processor()
    model_dim = model.config.projection_dim # Get embedding dimension from model config

    faiss_index = load_faiss_index(model_dim)
    next_faiss_id = faiss_index.ntotal # Start assigning new FAISS IDs from here if not using DB IDs

    # If using DB primary keys as FAISS IDs, you'd need a different strategy for next_faiss_id
    # For simplicity here, we'll use sequential FAISS IDs and store them in DB.
    # A more robust way if using IndexIDMap is to use the DB's primary key as the FAISS ID.
    # Let's adjust to use DB's primary key for FAISS IDs for better removal.
    # We'll need to fetch max(id) from DB to know where to start if index is empty.

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("Scanning images on disk...")
    disk_images_info = get_images_on_disk()
    print(f"Found {len(disk_images_info)} images on disk.")

    print("Fetching processed image info from database...")
    db_images_info = get_image_info_from_db()
    print(f"Found {len(db_images_info)} images in database.")

    # --- Identify changes ---
    disk_paths = set(disk_images_info.keys())
    db_paths = set(db_images_info.keys())

    new_paths = list(disk_paths - db_paths)
    deleted_paths = list(db_paths - disk_paths)
    existing_paths = list(disk_paths.intersection(db_paths))

    images_to_vectorize_paths = [] # Full paths for vectorization
    images_to_vectorize_rel_paths = [] # Relative paths for DB
    images_to_vectorize_metadata = [] # (rel_path, hash, last_modified)

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

        # Check if last_modified time changed
        if abs(disk_info["last_modified"] - db_info["last_modified"]) > 1e-6 : # Compare floats carefully
            # If modified time changed, verify with hash (more robust)
            current_hash = get_file_hash(disk_info["full_path"])
            if current_hash != db_info["hash"]:
                print(f"Image modified (hash mismatch): {rel_path}")
                images_to_vectorize_paths.append(disk_info["full_path"])
                images_to_vectorize_rel_paths.append(rel_path)
                images_to_vectorize_metadata.append((rel_path, current_hash, disk_info["last_modified"]))
                modified_count += 1
                # We'll need to remove its old vector from FAISS
                if db_info["faiss_id"] is not None:
                    try:
                        faiss_index.remove_ids(np.array([db_info["faiss_id"]], dtype=np.int64))
                        print(f"  Removed old FAISS ID {db_info['faiss_id']} for modified image.")
                    except RuntimeError as e:
                        print(f"  Warning: Could not remove FAISS ID {db_info['faiss_id']}: {e}. "
                              "Consider rebuilding index if this persists.")
                # Mark old DB entry for faiss_id update later
                cursor.execute("UPDATE images SET faiss_id = NULL WHERE relative_path = ?", (rel_path,))

    print(f"Found {modified_count} modified images to re-vectorize.")

    # --- Process Deletions ---
    if deleted_paths:
        print(f"Processing {len(deleted_paths)} deleted images...")
        faiss_ids_to_remove = []
        for rel_path in deleted_paths:
            if db_images_info[rel_path]["faiss_id"] is not None:
                faiss_ids_to_remove.append(db_images_info[rel_path]["faiss_id"])
            cursor.execute("DELETE FROM images WHERE relative_path = ?", (rel_path,))
            print(f"  Removed from DB: {rel_path}")

        if faiss_ids_to_remove:
            print(f"  Removing {len(faiss_ids_to_remove)} IDs from FAISS index...")
            try:
                faiss_index.remove_ids(np.array(faiss_ids_to_remove, dtype=np.int64))
                print(f"  Successfully removed {len(faiss_ids_to_remove)} IDs from FAISS.")
            except RuntimeError as e: # faiss.FaissException might also be relevant
                print(f"  Error removing IDs from FAISS: {e}. "
                      "The index might need rebuilding if IDs were not found. "
                      "This can happen if FAISS IDs were not consistently managed.")
        conn.commit()

    # --- Vectorize New and Modified Images ---
    if images_to_vectorize_paths:
        print(f"Vectorizing {len(images_to_vectorize_paths)} new/modified images in batches of {BATCH_SIZE}...")
        new_vectors_list = []
        new_faiss_ids_for_db = [] # Store (db_id, faiss_id_to_assign)

        # Get the next available primary key ID from the DB for new entries
        # This is a bit simplified; for concurrent access, this needs care.
        # For modified images, we'll update existing rows.
        cursor.execute("SELECT MAX(id) FROM images")
        max_db_id_row = cursor.fetchone()
        current_max_db_id = max_db_id_row[0] if max_db_id_row and max_db_id_row[0] is not None else 0

        temp_faiss_id_counter = faiss_index.ntotal # A simple way to get unique IDs if not using DB IDs directly
        # If we want to use DB's primary key as FAISS ID, we need to ensure they are unique
        # and handle cases where FAISS index might have been built differently.
        # For IndexIDMap, the IDs can be arbitrary 64-bit integers.

        for i in range(0, len(images_to_vectorize_paths), BATCH_SIZE):
            batch_paths = images_to_vectorize_paths[i : i + BATCH_SIZE]
            batch_rel_paths = images_to_vectorize_rel_paths[i : i + BATCH_SIZE]
            batch_metadata = images_to_vectorize_metadata[i : i + BATCH_SIZE]

            print(f"  Processing batch {i // BATCH_SIZE + 1}...")
            batch_vectors = vectorize_batch(batch_paths, processor, model)

            valid_vectors_in_batch = []
            valid_faiss_ids_for_batch = []
            db_updates_for_batch = [] # (hash, last_mod, vectorized_at, faiss_id, rel_path) for UPDATE
            db_inserts_for_batch = [] # (rel_path, hash, last_mod, vectorized_at, faiss_id) for INSERT

            for j, vec in enumerate(batch_vectors):
                if vec is not None:
                    rel_path = batch_rel_paths[j]
                    meta = batch_metadata[j] # (rel_path, hash, last_modified)

                    # Determine if it's an update or insert based on whether it was 'modified' or 'new'
                    is_update = rel_path in existing_paths # More accurately, if it was identified as modified

                    # Assign a FAISS ID. Using DB primary key is robust.
                    # For new images:
                    if not is_update: # It's a new image
                        current_max_db_id += 1
                        assigned_faiss_id = current_max_db_id # Use DB ID as FAISS ID
                        db_inserts_for_batch.append(
                            (rel_path, meta[1], meta[2], time.time(), assigned_faiss_id)
                        )
                    else: # It's a modified image, find its DB ID
                        cursor.execute("SELECT id FROM images WHERE relative_path = ?", (rel_path,))
                        db_id_row = cursor.fetchone()
                        if db_id_row:
                            assigned_faiss_id = db_id_row[0]
                            db_updates_for_batch.append(
                                (meta[1], meta[2], time.time(), assigned_faiss_id, rel_path)
                            )
                        else: # Should not happen if logic is correct
                            print(f"Error: Modified image {rel_path} not found in DB for ID retrieval.")
                            continue

                    valid_vectors_in_batch.append(vec)
                    valid_faiss_ids_for_batch.append(assigned_faiss_id)

            if valid_vectors_in_batch:
                vectors_np = np.array(valid_vectors_in_batch).astype("float32")
                faiss_ids_np = np.array(valid_faiss_ids_for_batch, dtype=np.int64)
                faiss_index.add_with_ids(vectors_np, faiss_ids_np)
                new_vectors_list.extend(valid_vectors_in_batch) # Not strictly needed if directly adding to FAISS

                # Update/Insert into DB
                if db_inserts_for_batch:
                    cursor.executemany(
                        """INSERT INTO images (relative_path, file_hash, last_modified, vectorized_at, faiss_id)
                           VALUES (?, ?, ?, ?, ?)""",
                        db_inserts_for_batch,
                    )
                if db_updates_for_batch:
                    cursor.executemany(
                        """UPDATE images SET file_hash = ?, last_modified = ?, vectorized_at = ?, faiss_id = ?
                           WHERE relative_path = ?""",
                        db_updates_for_batch,
                    )
                conn.commit()
                print(f"    Added {len(valid_vectors_in_batch)} vectors to FAISS and DB for this batch.")

        print(f"Finished vectorizing. Total new/modified vectors added: {faiss_index.ntotal - (next_faiss_id - len(deleted_paths))}") # Rough count

    else:
        print("No new or modified images to vectorize.")

    # --- Save final FAISS index ---
    save_faiss_index(faiss_index)
    conn.close()
    print("Processing complete.")

if __name__ == "__main__":
    main()
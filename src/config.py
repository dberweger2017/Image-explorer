# src/config.py
import os
import torch

MODEL_ID = "google/siglip-vit-so400m-patch14-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

IMAGE_SOURCE_DIR = os.path.join(PROJECT_ROOT, "image_source")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(PROCESSED_DATA_DIR, "image_metadata.sqlite3")
FAISS_INDEX_PATH = os.path.join(PROCESSED_DATA_DIR, "image_vectors.index")

BATCH_SIZE = 16
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
)
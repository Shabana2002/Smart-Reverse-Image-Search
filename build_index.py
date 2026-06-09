import os
import pickle
import faiss
import numpy as np
from tqdm import tqdm
from utils.feature_extractor import DinoFeatureExtractor

DATASET_DIR = "dataset"
EMBEDDING_DIR = "embeddings"

os.makedirs(EMBEDDING_DIR, exist_ok=True)

extractor = DinoFeatureExtractor()

all_features = []
all_paths = []

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

image_paths = []

for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith(image_extensions):
            image_paths.append(os.path.join(root, file))

for image_path in tqdm(image_paths):
    try:
        feature = extractor.extract(image_path)
        all_features.append(feature)
        all_paths.append(image_path)

    except Exception as e:
        print(f"Skipped {image_path}: {e}")

features = np.array(all_features).astype("float32")

np.save("embeddings/features.npy", features)

with open("embeddings/image_paths.pkl", "wb") as f:
    pickle.dump(all_paths, f)

dimension = features.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(features)

faiss.write_index(index, "embeddings/faiss_index.bin")

print("Index Created Successfully")

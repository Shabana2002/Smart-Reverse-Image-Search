import os
import pickle
import faiss
import numpy as np
import streamlit as st
from PIL import Image
from utils.feature_extractor import DinoFeatureExtractor

st.title("Smart Reverse Image Search")

# Load DINOv2 feature extractor
extractor = DinoFeatureExtractor()

# Load FAISS index
index = faiss.read_index("embeddings/faiss_index.bin")

# Load image paths
with open("embeddings/image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    # Open uploaded image
    uploaded_image = Image.open(uploaded_file)

    # Display query image
    st.image(uploaded_image, caption="Query Image", width=300)

    # Convert RGBA → RGB
    if uploaded_image.mode != "RGB":
        uploaded_image = uploaded_image.convert("RGB")

    # Save temporary image
    temp_path = "temp_query.jpg"
    uploaded_image.save(temp_path, "JPEG")

    # Extract DINOv2 features
    query_feature = extractor.extract(temp_path)

    # Search FAISS
    scores, indices = index.search(
        np.array([query_feature]).astype("float32"),
        5
    )

    st.subheader("Top Similar Images")

    threshold = 0.50
    valid_results = False

    for score, idx in zip(scores[0], indices[0]):

        if score >= threshold:

            image_path = image_paths[idx]

            # Fix Windows path for Linux deployment
            image_path = image_path.replace("\\", "/")

            if os.path.exists(image_path):

                valid_results = True

                st.image(image_path, width=250)
                st.write(f"Similarity Score: {score:.3f}")

    if not valid_results:
        st.warning("No similar images found in the database.")

    # Optional debugging
    st.write("Raw Similarity Scores:", scores[0])
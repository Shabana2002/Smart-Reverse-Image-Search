import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DinoFeatureExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-small"
        )

        self.model = AutoModel.from_pretrained(
            "facebook/dinov2-small"
        )

        self.model.to(self.device)
        self.model.eval()

    def extract(self, image_path):
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

            embedding = outputs.last_hidden_state[:, 0]
            embedding = embedding.cpu().numpy().flatten()

        embedding = embedding / np.linalg.norm(embedding)

        return embedding
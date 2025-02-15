import streamlit as st
import torch
import os
import wget
from PIL import Image
import numpy as np
from torchvision import models, transforms
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from datetime import datetime
from transformers import BertTokenizer, BertModel
torch.classes.__path__ = []
# --- Configuration ---
CFG_FRCNN_MODEL = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
CFG_SAM_MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
CFG_MODEL_DIR = "models/"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure model directory exists
os.makedirs(CFG_MODEL_DIR, exist_ok=True)


@st.cache_resource
def download_model(url, filename):
    """Download a model if it does not exist."""
    model_path = os.path.join(CFG_MODEL_DIR, filename)
    if not os.path.exists(model_path):
        st.write(f"Downloading {filename}...")
        wget.download(url, out=model_path)
    return model_path


# --- Download SAM Model ---
sam_model_path = download_model(CFG_SAM_MODEL_URL, "sam_vit_h_4b8939.pth")

# --- Define NoviceNutriVision ---
class NoviceNutriVision(torch.nn.Module):
    def __init__(self, food_nutrition_dim, fv_dim, fastfood_dim, device="cuda"):
        super(NoviceNutriVision, self).__init__()
        self.device = device

        # Faster R-CNN for object detection
        self.detector = CFG_FRCNN_MODEL.to(self.device).eval()

        # SAM for segmentation
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_model_path)
        self.sam.to(self.device).eval()
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

        # CNN for visual feature extraction (ResNet18)
        resnet = models.resnet18(pretrained=True)
        self.cnn = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer

        # BERT for text processing
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

        # Feature fusion
        self.fusion_fc = torch.nn.Sequential(
            torch.nn.Linear(512 + 768, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )

        # Regression heads
        self.food_nutrition_head = torch.nn.Linear(512, food_nutrition_dim)
        self.fv_head = torch.nn.Linear(512, fv_dim)
        self.fastfood_head = torch.nn.Linear(512, fastfood_dim)

    def forward(self, image, source):
        # Step 1: Object Detection (Faster R-CNN)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        results = self.detector(image_tensor)[0]  # Get detection results

        if len(results["boxes"]) > 0:
            best_idx = torch.argmax(results["scores"]).item()
            best_box = results["boxes"][best_idx].cpu().numpy().astype(int)
            x1, y1, x2, y2 = best_box
            crop = np.array(image)[y1:y2, x1:x2, :]
            pred_class_label = str(results["labels"][best_idx].item())  # Class ID
        else:
            crop = np.array(image)
            pred_class_label = "food"

        # Step 2: Segmentation (SAM)
        masks = self.mask_generator.generate(crop)
        if masks and len(masks) > 0:
            best_mask = max(masks, key=lambda m: m['area'])
            seg_mask = best_mask['segmentation']
            segmented = np.where(np.stack([seg_mask] * 3, axis=-1), crop, 0)
        else:
            segmented = crop

        # Step 3: Extract Features
        img_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(Image.fromarray(segmented.astype(np.uint8))).unsqueeze(0).to(self.device)

        visual_features = self.cnn(img_tensor).view(1, -1)

        text_input = f"This is a {pred_class_label} food item."
        encoded_input = self.bert_tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        text_features = self.bert_model(**{k: v.to(self.device) for k, v in encoded_input.items()}).last_hidden_state[:, 0, :]

        # Step 4: Fusion and Prediction
        fused = self.fusion_fc(torch.cat([visual_features, text_features], dim=-1))

        if source == "food_nutrition":
            return self.food_nutrition_head(fused), pred_class_label
        elif source == "fv":
            return self.fv_head(fused), pred_class_label
        elif source == "fastfood":
            return self.fastfood_head(fused), pred_class_label
        else:
            raise ValueError("Invalid source")

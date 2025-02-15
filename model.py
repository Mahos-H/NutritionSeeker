import streamlit as st
import torch
import os
import wget
from PIL import Image
import numpy as np
from torchvision import models, transforms
from datetime import datetime
from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel
torch.classes.__path__ = []
# --- Configuration ---
CFG_FRCNN_MODEL = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
CFG_DEEPLAB_MODEL = models.segmentation.deeplabv3_resnet50(pretrained=True)
CFG_MODEL_DIR = "models/"
device = "cuda" if torch.cuda.is_available() else "cpu"
class NoviceNutriVision(torch.nn.Module):
    def __init__(self, food_nutrition_dim, fv_dim, fastfood_dim, device="cuda"):
        super(NoviceNutriVision, self).__init__()
        self.device = device

        # Lightweight CNN for feature extraction
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.cnn = torch.nn.Sequential(*list(mobilenet.features.children())).to(self.device).eval()

        # DistilBERT for text processing
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(self.device).eval()

        # Feature fusion
        self.fusion_fc = torch.nn.Sequential(
            torch.nn.Linear(1280 + 768, 512),  # 1280 (MobileNetV2) + 768 (DistilBERT)
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )

        # Regression heads
        self.food_nutrition_head = torch.nn.Linear(512, food_nutrition_dim)
        self.fv_head = torch.nn.Linear(512, fv_dim)
        self.fastfood_head = torch.nn.Linear(512, fastfood_dim)

    def forward(self, image, source):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # CNN Feature Extraction
        visual_features = self.cnn(image_tensor).view(1, -1)

        # DistilBERT Processing
        text_input = "This is a food item."
        encoded_input = self.bert_tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        text_features = self.bert_model(**{k: v.to(self.device) for k, v in encoded_input.items()}).last_hidden_state[:, 0, :]

        # Fusion and Prediction
        fused = self.fusion_fc(torch.cat([visual_features, text_features], dim=-1))

        if source == "food_nutrition":
            return self.food_nutrition_head(fused)
        elif source == "fv":
            return self.fv_head(fused)
        elif source == "fastfood":
            return self.fastfood_head(fused)
        else:
            raise ValueError("Invalid source")


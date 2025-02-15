import streamlit as st
import torch
import os
import wget
from PIL import Image
import numpy as np
from ultralytics import YOLO,settings
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from datetime import datetime
from torchvision import transforms
from transformers import BertTokenizer, BertModel
import torchvision.models as models
settings_path = "/home/appuser/.config/Ultralytics/settings.json"

# Load current settings
with open(settings_path, "r") as file:
    settings = json.load(file)

# Update the "runs_dir" path
settings["runs_dir"] = "/mount/src/nutritionseeker/test"

# Write the updated settings back to the file
with open(settings_path, "w") as file:
    json.dump(settings, file, indent=4)

st.write("Settings updated successfully!")
st.write(settings)
torch.classes.__path__ = []
CFG_YOLO_MODEL_URL = "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt"
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


# --- Download Models ---
yolo_model_path = download_model(CFG_YOLO_MODEL_URL, "yolov8n.pt")
sam_model_path = download_model(CFG_SAM_MODEL_URL, "sam_vit_h_4b8939.pth")


# --- Define NoviceNutriVision ---
class NoviceNutriVision(torch.nn.Module):
    def __init__(self, food_nutrition_dim, fv_dim, fastfood_dim, device="cuda"):
        super(NoviceNutriVision, self).__init__()
        self.device = device

        # YOLOv8 for object detection
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.model.eval()

        # SAM2 for segmentation
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_model_path)
        self.sam.to(self.device)
        self.sam.eval()
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
        # Step 1: Object Detection (YOLOv8)
        image_np = np.array(image)
        results = self.yolo_model(image_np, verbose=False)
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confs)
            best_box = boxes.xyxy[best_idx].cpu().numpy()
            x1, y1, x2, y2 = best_box.astype(int)
            crop = image_np[y1:y2, x1:x2, :]
            pred_class_label = self.yolo_model.model.names[int(boxes.cls[best_idx].cpu().numpy())]
        else:
            crop = image_np
            pred_class_label = "food"

        # Step 2: Segmentation (SAM2)
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

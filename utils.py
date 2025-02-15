import os
import torch
import torch.nn as nn
from torchvision import models, transforms
#from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel
import streamlit as st
# --- Configuration ---
MODEL_DIR = "models/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Change to "cpu" explicitly if you know CUDA isn't available
#device = "cpu"
# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


def save_model(model, filepath):
    """Save model state dictionary."""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved as {filepath}")

from model import NoviceNutriVision  # Ensure the correct model class is imported

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "models/"
MODEL_PATH = os.path.join(MODEL_DIR, "novice_nutrivision.pth")
@st.cache_resource
def load_model():
    """Load the NoviceNutriVision model and return both the model and device."""
    model = NoviceNutriVision(food_nutrition_dim=10, fv_dim=5, fastfood_dim=3).to(device)  
    #model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
    return model, DEVICE  # Returning both the model and device


from PIL import Image
def predict(model, image_tensor, source):
    """
    Predict the class for a single image using the trained model.
    
    Parameters:
    - model: Trained NoviceNutriVision model.
    - image: Image file path (str) or a PIL.Image object.
    - source: Classification type ('food_nutrition', 'fv', or 'fastfood').
    
    Returns:
    - Prediction scores (tensor) and predicted class label.
    """
    model.to(DEVICE)
    model.eval()

    # Ensure the input is a PIL Image object
    if isinstance(image_tensor, str):  
        image = Image.open(image).convert("RGB")  # Load from path if it's a string

    # Define the same transformations as used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    #image_tensor = transform(image).unsqueeze(0).to(DEVICE)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor, source)  # Pass tensor and classification type
        predicted_class = torch.argmax(output, dim=1).cpu().item()  # Get predicted label
    
    return output, predicted_class
class NoviceNutriVision(torch.nn.Module):
    def __init__(self, food_nutrition_dim, fv_dim, fastfood_dim, device="cpu",visual_fc= nn.Linear(62720, 1280)):
        super(NoviceNutriVision, self).__init__()
        self.device = device
        self.visual_fc = nn.Linear(62720, 1280)
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

    def forward(self, image_tensor, source):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        #image_tensor = transform(image).unsqueeze(0).to(self.device)

        # CNN Feature Extraction
        visual_features = self.cnn(image_tensor).view(1, -1)

        # DistilBERT Processing
        text_input = "This is a food item."
        encoded_input = self.bert_tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        text_features = self.bert_model(**{k: v.to(self.device) for k, v in encoded_input.items()}).last_hidden_state[:, 0, :]

        # Fusion and Prediction
        #st.write(visual_features.shape,text_features.shape)
        visual_features = torch.nn.functional.adaptive_avg_pool1d(visual_features.unsqueeze(0), 768).squeeze(0)
        st.write("abc\n",visual_features.shape,text_features.shape)
        xx=torch.cat([visual_features, text_features], dim=1)
        st.write(xx.shape)
        fused = self.fusion_fc(xx)

        if source == "food_nutrition":
            return self.food_nutrition_head(fused)
        elif source == "fv":
            return self.fv_head(fused)
        elif source == "fastfood":
            return self.fastfood_head(fused)
        else:
            raise ValueError("Invalid source")



# --- Save and Load Example ---
novice_model = NoviceNutriVision(food_nutrition_dim=10, fv_dim=5, fastfood_dim=3)

model_path = os.path.join(MODEL_DIR, "novice_nutrivision.pth")
#save_model(novice_model, model_path)
loaded_model = load_model()

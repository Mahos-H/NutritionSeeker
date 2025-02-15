import os
import torch
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel

# --- Configuration ---
MODEL_DIR = "models/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def load_model():
    """Load the NoviceNutriVision model and return both the model and device."""
    model = NoviceNutriVision(food_nutrition_dim=10, fv_dim=5, fastfood_dim=3).to(DEVICE)  
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
    return model, DEVICE  # Returning both the model and device


def predict(model, dataloader):
    """Run inference on a dataset using a trained model."""
    model.to(DEVICE)
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)

            # If output is logits, apply softmax for classification
            if isinstance(outputs, torch.Tensor):
                outputs = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    return predictions


# --- Define Models ---
class NoviceNutriVision(torch.nn.Module):
    def __init__(self, food_nutrition_dim, fv_dim, fastfood_dim):
        super(NoviceNutriVision, self).__init__()

        # Faster R-CNN for object detection
        self.detector = models.detection.fasterrcnn_resnet50_fpn()
        
        # DeepLabV3 for segmentation
        self.segmenter = models.segmentation.deeplabv3_resnet50()

        # ResNet18 for feature extraction
        resnet = models.resnet18()
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
        # Object Detection
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        results = self.detector(image_tensor)[0]

        if len(results["boxes"]) > 0:
            best_idx = torch.argmax(results["scores"]).item()
            best_box = results["boxes"][best_idx].cpu().numpy().astype(int)
            x1, y1, x2, y2 = best_box
            crop = image.crop((x1, y1, x2, y2))
            pred_class_label = str(results["labels"][best_idx].item())
        else:
            crop = image
            pred_class_label = "food"

        # Segmentation
        crop_tensor = transform(crop).unsqueeze(0).to(DEVICE)
        seg_output = self.segmenter(crop_tensor)["out"]
        seg_mask = torch.argmax(seg_output.squeeze(), dim=0).cpu().numpy()

        segmented = torch.where(torch.tensor(seg_mask, dtype=torch.bool), crop_tensor, torch.zeros_like(crop_tensor))

        # Feature Extraction
        img_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(segmented.squeeze()).unsqueeze(0).to(DEVICE)

        visual_features = self.cnn(img_tensor).view(1, -1)

        # Text Processing
        text_input = f"This is a {pred_class_label} food item."
        encoded_input = self.bert_tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        text_features = self.bert_model(**{k: v.to(DEVICE) for k, v in encoded_input.items()}).last_hidden_state[:, 0, :]

        # Fusion and Prediction
        fused = self.fusion_fc(torch.cat([visual_features, text_features], dim=-1))

        if source == "food_nutrition":
            return self.food_nutrition_head(fused), pred_class_label
        elif source == "fv":
            return self.fv_head(fused), pred_class_label
        elif source == "fastfood":
            return self.fastfood_head(fused), pred_class_label
        else:
            raise ValueError("Invalid source")


# --- Save and Load Example ---
novice_model = NoviceNutriVision(food_nutrition_dim=10, fv_dim=5, fastfood_dim=3)

model_path = os.path.join(MODEL_DIR, "novice_nutrivision.pth")
save_model(novice_model, model_path)
loaded_model = load_model()

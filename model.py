import torch
import torch.nn as nn
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import BertTokenizer, BertModel
import torchvision.models as models
torch.classes.__path__ = []
class NoviceNutriVision(nn.Module):
  def __init__(self, food_nutrition_dim, fv_dim, fastfood_dim, device="cuda"):
        """
        food_nutrition_dim, fv_dim, fastfood_dim: output dimensions for the three nutrition targets.
        device: "cuda" or "cpu".
        """
        super(NoviceNutriVision, self).__init__()
        self.device = device
        
        # ----- YOLOv8 for object detection -----
        # (Assumes you have a YOLOv8 model fine-tuned on food or use a general model)
        self.yolo_model = YOLO("yolov8n.pt")  # adjust path as needed
        self.yolo_model.model.eval()
        
        # ----- SAM2 for segmentation -----
        sam_checkpoint = "sam_vit_h_4b8939.pth"  # adjust to your checkpoint
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        
        # ----- CNN for visual feature extraction (ResNet18) -----
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # remove final fully connected layer
        self.cnn = nn.Sequential(*modules)  # outputs (batch, 512, 1, 1)
        
        # ----- BERT for text feature extraction -----
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        # (Optionally, you can freeze BERT parameters if desired)
        
        # ----- Fusion of visual and text features -----
        # Visual: 512-d, Text: 768-d => Total: 1280-d. Reduce to a hidden dimension (e.g., 512)
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # ----- Regression heads for each nutrition source -----
        self.food_nutrition_head = nn.Sequential(
            nn.Linear(512, food_nutrition_dim)
        )
        self.fv_head = nn.Sequential(
            nn.Linear(512, fv_dim)
        )
        self.fastfood_head = nn.Sequential(
            nn.Linear(512, fastfood_dim)
        )
    
  def forward(self, image, source):
        """
        image: a PIL image.
        source: string indicating which nutrition head to use.
        Returns: (prediction, detected_class_label)
        """
        # ----- Step 1: Object Detection with YOLOv8 -----
        image_np = np.array(image)
        try:
            results = self.yolo_model(image_np, verbose=False)
            if (results and len(results) > 0 and 
                results[0].boxes is not None and len(results[0].boxes) > 0):
                boxes = results[0].boxes
                confs = boxes.conf.cpu().numpy()
                best_idx = np.argmax(confs)
                best_box = boxes.xyxy[best_idx].cpu().numpy()[0]  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = best_box.astype(int)
                h, w, _ = image_np.shape
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, w)
                y2 = min(y2, h)
                crop = image_np[y1:y2, x1:x2, :]
                # Get the predicted class label from YOLO if available
                if hasattr(boxes, 'cls'):
                    pred_class_idx = int(boxes.cls[best_idx].cpu().numpy())
                    if hasattr(self.yolo_model.model, 'names'):
                        pred_class_label = self.yolo_model.model.names[pred_class_idx]
                    else:
                        pred_class_label = str(pred_class_idx)
                else:
                    pred_class_label = "food"
            else:
                crop = image_np
                pred_class_label = "food"
        except Exception as e:
            print("YOLO detection failed:", e)
            crop = image_np
            pred_class_label = "food"
        
        # ----- Step 2: Segmentation with SAM2 -----
        try:
            masks = self.mask_generator.generate(crop)
            if masks and len(masks) > 0:
                best_mask = max(masks, key=lambda m: m['area'])
                seg_mask = best_mask['segmentation']
                seg_mask_3ch = np.stack([seg_mask] * 3, axis=-1)
                segmented = np.where(seg_mask_3ch, crop, 0)
            else:
                segmented = crop
        except Exception as e:
            print("SAM segmentation failed:", e)
            segmented = crop
        
        # ----- Step 3: Visual Feature Extraction via CNN -----
        segmented_pil = Image.fromarray(segmented.astype(np.uint8))
        transform_cnn = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform_cnn(segmented_pil).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        visual_features = self.cnn(img_tensor)  # (1, 512, 1, 1)
        visual_features = visual_features.view(1, -1)  # (1, 512)
        
        # ----- Step 4: Text Feature Extraction via BERT -----
        # Create a text prompt from the detected class label.
        text_input = f"This is a {pred_class_label} food item."
        encoded_input = self.bert_tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        bert_output = self.bert_model(**encoded_input)
        text_features = bert_output.last_hidden_state[:, 0, :]  # (1, 768) from the [CLS] token
        
        # ----- Step 5: Fusion and Regression -----
        fused = torch.cat([visual_features, text_features], dim=-1)  # (1, 1280)
        fused = self.fusion_fc(fused)  # (1, 512)
        
        if source == "food_nutrition":
            pred = self.food_nutrition_head(fused)
        elif source == "fv":
            pred = self.fv_head(fused)
        elif source == "fastfood":
            pred = self.fastfood_head(fused)
        else:
            raise ValueError(f"Unknown source: {source}")
        
        return pred, pred_class_label

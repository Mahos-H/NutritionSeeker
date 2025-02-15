import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class UnifiedMultiSourceDataset(Dataset):
    def __init__(self, roots, mapping_fastfood, mapping_fv, mapping_food, transform=None):
        self.samples = []
        self.transform = transform
        
        for root in roots:
            for dirpath, _, filenames in os.walk(root):
                for file in filenames:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        category = os.path.basename(dirpath).lower().strip()
                        if category in mapping_fastfood:
                            source = "fastfood"
                            target = mapping_fastfood[category]
                        elif category in mapping_fv:
                            source = "fv"
                            target = mapping_fv[category]
                        elif category in mapping_food:
                            source = "food_nutrition"
                            target = mapping_food[category]
                        else:
                            continue
                        img_path = os.path.join(dirpath, file)
                        self.samples.append((img_path, target, source, category))
        if len(self.samples) == 0:
            raise ValueError("No images found matching any nutrition mapping.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target, source, category = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return image, target_tensor, source, category

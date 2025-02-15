import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

def train_model(model, dataloader, num_epochs, optimizer, device):
    model.train()
    mse_loss = nn.MSELoss()
    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            images, targets, sources, class_names = batch
            batch_loss = 0.0
            optimizer.zero_grad()
            for i in range(len(images)):
                image = images[i]
                target = targets[i].to(device).unsqueeze(0)
                source = sources[i]
                pred, detected_label = model(image, source)
                loss = mse_loss(pred, target)
                loss.backward()
                batch_loss += loss.item()
            optimizer.step()
            pbar.set_postfix({"loss": batch_loss/len(images)})
            running_loss += batch_loss
        avg_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
    print("Training complete.")

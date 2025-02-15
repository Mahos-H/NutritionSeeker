import os

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved as {filepath}")

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")

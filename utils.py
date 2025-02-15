import os

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved as {filepath}")

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")
def predict(model, dataloader):
    # Check if GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to the correct device
    
    model.eval()  # Set the model to evaluation mode
    predictions = []
    
    with torch.no_grad():  # No need to track gradients during prediction
        for inputs, _ in dataloader:
            inputs = inputs.to(device)  # Move inputs to the appropriate device
            outputs = model(inputs)  # Get the model's output
            
            # If the model's output is logits, apply softmax for probabilities
            if isinstance(outputs, torch.Tensor):
                outputs = torch.softmax(outputs, dim=1)  # Assuming a classification task

            # Convert the output to the predicted class
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())  # Convert tensor to numpy for easier use

    return predictions

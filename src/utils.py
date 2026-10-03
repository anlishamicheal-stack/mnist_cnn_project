import torch
import matplotlib.pyplot as plt

# Save trained model checkpoint
def save_checkpoint(model, path="outputs/checkpoints/mnist_cnn.pt"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Load model checkpoint
def load_checkpoint(model, path="outputs/checkpoints/mnist_cnn.pt"):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

# Plot training loss curve
def plot_losses(train_losses, save_path="outputs/logs/train_loss.png"):
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Training loss plot saved to {save_path}")

# Visualize predictions
def visualize_predictions(model, device, test_loader, num_images=6):
    import numpy as np
    import matplotlib.pyplot as plt

    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[i].imshow(images[i].cpu().squeeze(), cmap="gray")
        axes[i].set_title(f"Pred: {preds[i].item()}\nTrue: {labels[i].item()}")
        axes[i].axis("off")
    plt.show()

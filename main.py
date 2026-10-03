import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.model import CNN
from src.train import train
from src.test import test

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Datasets
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model + Optimizer
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1, 6):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

from src.utils import plot_losses

train_losses = []
for epoch in range(1, 6):
    train_losses = train(model, device, train_loader, optimizer, epoch, train_losses)
    test(model, device, test_loader)

plot_losses(train_losses, save_path="outputs/logs/train_loss.png")

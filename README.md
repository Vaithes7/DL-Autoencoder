# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
To build and train a Denoising Autoencoder using PyTorch to remove noise from images. The goal is to reconstruct a clean image from a corrupted (noisy) input image. The MNIST dataset is used as a benchmark to demonstrate this image denoising capability. The dataset used in this code is the MNIST dataset. It's a large database of handwritten digits commonly used for training various image processing systems. Each image in the MNIST dataset is a grayscale image of a handwritten digit (0-9) with a size of 28x28 pixels.


## DESIGN STEPS
### STEP 1: 

Problem Understanding and Dataset Selection

### STEP 2: 

Preprocessing the Dataset


### STEP 3: 

Design the Convolutional Autoencoder Architecture

### STEP 4: 

Compile and Train the Model

### STEP 5: 

Evaluate the Model

### STEP 6: 

Visualization and Analysis


## PROGRAM

### Name:Vaitheswaran N

### Register Number: 212224110059

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Name: Vaitheswaran N")
print("Register No: 212224110059")
summary(model, input_size=(1, 28, 28))

def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print("Name: Vaitheswaran N")
    print("Register No: 212224110059")
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")

def visualize_denoising(model, loader, num_images=10):
    model.eval()

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: Vaitheswaran N")
    print("Register No: 212224110059")
    plt.figure(figsize=(18, 6))

    for i in range(num_images):
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")


        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")


        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)

```

### OUTPUT

### Model Summary

<img width="752" height="444" alt="image" src="https://github.com/user-attachments/assets/e448729e-abc7-440e-90d5-5579efdc9e86" />


### Training loss

<img width="359" height="195" alt="image" src="https://github.com/user-attachments/assets/8cd6e35b-c6dc-4ea2-bbce-4aaadd4f24c7" />


## Original vs Noisy Vs Reconstructed Image
<img width="1273" height="463" alt="image" src="https://github.com/user-attachments/assets/287d3416-ff3b-4496-9c85-300940460dbd" />


## RESULT

Thus, develop a convolutional autoencoder for image denoising application excuted succesfully


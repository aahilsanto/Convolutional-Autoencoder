# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

Design and implement a Denoising Autoencoder using PyTorch to remove noise from handwritten digit images in the MNIST dataset. The model should take noisy images as input and learn to reconstruct the original clean images. Train the autoencoder using convolutional layers and evaluate its performance by visualizing the original, noisy, and denoised images after training.

<img width="1221" height="234" alt="image" src="https://github.com/user-attachments/assets/e8ba9deb-04f7-4393-a188-5de93f90bdd6" />


## DESIGN STEPS

### Step 1: 
Import the required PyTorch, Torchvision, NumPy, and Matplotlib libraries. Configure the device to use GPU (CUDA) if available, otherwise use CPU.

### Step 2: 
Load the MNIST dataset and apply transformations to convert images into tensors. Create DataLoader objects for training and testing with a batch size of 128.

### Step 3: 
Define a function to add random noise to the input images. This simulates corrupted images which the model will learn to clean.

### Step 4: 
Build the Denoising Autoencoder model using convolutional layers for the encoder and transposed convolutional layers for the decoder to reconstruct the original image.

### Step 5: 
Initialize the model, loss function (MSELoss), and optimizer (Adam). Train the model for multiple epochs where noisy images are given as input and the original images are used as targets.

### Step 6: 
After training, visualize the results by displaying original images, noisy images, and denoised outputs to evaluate how well the model removes noise.


## PROGRAM
### Name: Ahil Santo A
### Register Number: 212224040018

```python
#model function
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             
            nn.Conv2d(16, 8, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2)              
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 2, stride=2),    
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),    
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),            
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
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
```

## OUTPUT

### Model Summary

<img width="757" height="527" alt="image" src="https://github.com/user-attachments/assets/1f494b6c-7661-4808-a95c-0542d4c17a2b" />

### Original vs Noisy Vs Reconstructed Image

<img width="1601" height="600" alt="image" src="https://github.com/user-attachments/assets/2cb6a348-c739-433d-a675-a859565652b1" />

## RESULT

Thus to develop a convolutional autoencoder for image denoising application is done successfully.

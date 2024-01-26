import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import DatasetFromFolder
from model import SRCNN

parser = argparse.ArgumentParser(description='SRCNN training parameters')
parser.add_argument('--zoom_factor', type=int, required=True)  # Zoom factor for image super-resolution
parser.add_argument('--nb_epochs', type=int, default=200)  # Number of training epochs
parser.add_argument('--cuda', action='store_true')  # Use CUDA for training if available
args = parser.parse_args()

device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")  # Set device for training
torch.manual_seed(0)  # Set random seed for reproducibility
torch.cuda.manual_seed(0)  # Set random seed for reproducibility on CUDA

# Parameters
BATCH_SIZE = 4  # Batch size for training
NUM_WORKERS = 0  # Number of worker threads for data loading (set to 0 on Windows)

trainset = DatasetFromFolder("data/train", zoom_factor=args.zoom_factor)  # Training dataset
testset = DatasetFromFolder("data/test", zoom_factor=args.zoom_factor)  # Test dataset

trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)  # Training data loader
testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)  # Test data loader

model = SRCNN().to(device)  # Create an instance of the SRCNN model and move it to the device
criterion = nn.MSELoss()  # Mean Squared Error loss function
optimizer = optim.Adam(  # Adam optimizer with different learning rates for different layers
    [
        {"params": model.conv1.parameters(), "lr": 0.0001},  
        {"params": model.conv2.parameters(), "lr": 0.0001},
        {"params": model.conv3.parameters(), "lr": 0.00001},
    ], lr=0.00001,
)

for epoch in range(args.nb_epochs):

    # Train
    epoch_loss = 0
    for iteration, batch in enumerate(trainloader):
        input, target = batch[0].to(device), batch[1].to(device)  # Move input and target tensors to the device
        optimizer.zero_grad()  # Clear the gradients

        out = model(input)  # Forward pass
        loss = criterion(out, target)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights
        epoch_loss += loss.item()

    print(f"Epoch {epoch}. Training loss: {epoch_loss / len(trainloader)}")

    # Test
    avg_psnr = 0
    with torch.no_grad():
        for batch in testloader:
            input, target = batch[0].to(device), batch[1].to(device)  # Move input and target tensors to the device

            out = model(input)  # Forward pass
            loss = criterion(out, target)  # Compute the loss
            psnr = 10 * log10(1 / loss.item())  # Compute the Peak Signal-to-Noise Ratio (PSNR)
            avg_psnr += psnr
    print(f"Average PSNR: {avg_psnr / len(testloader)} dB.")

    # Save model
    torch.save(model, f"model_{epoch}.pth")

import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)  # Corrected to have 32 output channels
        self.conv4 = nn.Conv2d(32, 1, kernel_size=5, padding=2)  # Ensure the final layer outputs a single channel

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)  # Ensure the channels match up here
        return x

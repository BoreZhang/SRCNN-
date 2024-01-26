import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)  # First convolutional layer
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)  # Second convolutional layer
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # Third convolutional layer
        self.conv4 = nn.Conv2d(16, 1, kernel_size=5, padding=2)  # Fourth convolutional layer
        
    def forward(self, x):
        out = F.relu(self.conv1(x))  # Apply ReLU activation to the output of the first convolutional layer
        out = F.relu(self.conv2(out))  # Apply ReLU activation to the output of the second convolutional layer
        out = F.relu(self.conv3(out))  # Apply ReLU activation to the output of the third convolutional layer
        out = self.conv4(out)  # Output of the fourth convolutional layer

        return out

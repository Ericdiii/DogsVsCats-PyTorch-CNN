import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

# Create a new network class and inherit the nn.Module parent class of PyTorch
class Net(nn.Module):                                      
    def __init__(self):                                     # Construct function to set the network layer
        super(Net, self).__init__()                         # Standard statement
        
        # First convolutional layer: 3 input channels, 16 output channels, 3×3 convolution kernel, 1 padding, other parameters are default
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        # Second convolutional layer: 16 input channels, 16 output channels, 3×3 convolution kernel, 1 padding, other parameters are default
        self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)

        # First full connected layer: Linear connection, 50×50×16 input nodes, 128 output nodes
        self.fc1 = nn.Linear(50*50*16, 128) 
        # Second full connected layer: Linear connection, 128 input nodes, 64 output nodes
        self.fc2 = nn.Linear(128, 64)
        # Third full connected layer: Linear connection, 64 input nodes, 2 output nodes
        self.fc3 = nn.Linear(64, 2)
    
    # Rewrite the forward method of the parent class: Forward calculation, obtain the output after the network receive the input data
    def forward(self, x):
        
        x = self.conv1(x)                   # ConV1
        x = F.relu(x)                       # Process the result through the ReLU activation function
        x = F.max_pool2d(x, 2)              # 2×2 Max pooling

        x = self.conv2(x)                   # ConV2
        x = F.relu(x)                       # Process the result through the ReLU activation function
        x = F.max_pool2d(x, 2)              # 2×2 Max pooling

        x = x.view(x.size()[0], -1)         # One-dimensional tensor fully connected layer: arrange the [50×50×16] input data into [40000×1]
        x = F.relu(self.fc1(x))             # FC1 activated by ReLu
        x = F.relu(self.fc2(x))             # FC2 activated by ReLu
        y = self.fc3(x)                     # FC3

        return y


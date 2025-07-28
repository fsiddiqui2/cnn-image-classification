import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, image_dim=224):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2) # N, C, H, W -> N, C, H/2, W/2 (C=32)
        self.batchnorm1 = nn.BatchNorm2d(num_features=32)
        
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size = 2) # N, C, H/2, W/2 -> N, C, H/4, W/4 (C=64)
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(in_channels = 64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size = 2) # N, C, H/2, W/2 -> N, C, H/4, W/4 (C=64)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)

        self.flattened_dim = int((image_dim/8)*(image_dim/8)*128)
        self.linear1 = nn.Linear(self.flattened_dim, 128) # (N, C*H*W) -> (N, 128)
        self.linear2 = nn.Linear(128, 1)

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.batchnorm1(x)
        self.relu(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.batchnorm2(x)
        self.relu(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.batchnorm3(x)
        self.relu(x)

        x = x.view(-1, self.flattened_dim)
        x = self.linear1(x)
        self.relu(x)
        x = self.linear2(x)
        return x
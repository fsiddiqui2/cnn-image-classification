import torch.nn as nn

class Model(nn.Module):
    def __init__(self, image_dim=224):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2) # N, C, H, W -> N, C, H/2, W/2 (C=8)
        self.batchnorm1 = nn.BatchNorm2d(num_features=8)
        self.dropout1 = nn.Dropout(0.5)
        
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size = 2) # N, C, H/2, W/2 -> N, C, H/4, W/4 (C=16)
        self.batchnorm2 = nn.BatchNorm2d(num_features=16)
        self.dropout2 = nn.Dropout(0.5)
        
        # self.conv3 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=3, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size = 2) # N, C, H/4, W/4 -> N, C, H/8, W/8 (C=64)

        self.flattened_dim = int((image_dim/4)*(image_dim/4)*16)
        self.linear1 = nn.Linear(self.flattened_dim, 64) # (N, C*H*W) -> (N, 128)
        self.linear2 = nn.Linear(64, 1)

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.batchnorm1(x)
        self.relu(x)
        # x = self.dropout1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.batchnorm2(x)
        self.relu(x)
        # x = self.dropout2(x)

        # x = self.pool3(self.conv3(x))
        # self.relu(x)

        x = x.view(-1, self.flattened_dim)
        x = self.linear1(x)
        self.relu(x)
        x = self.linear2(x)
        return x
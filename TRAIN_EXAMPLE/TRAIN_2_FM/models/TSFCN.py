import torch
import torch.nn as nn
import torch.nn.functional as F

# modified based on source code from: https://github.com/aybchan/time-series-classification.git

class ConvNet1D(nn.Module):
    def __init__(self, n_in, n_classes, n_channels=3):
        super(ConvNet1D, self).__init__()
        self.n_in = n_in
        self.n_classes = n_classes
        self.n_channels = n_channels  # Number of input channels

        # Adjust the first conv layer to accept n_channels
        self.conv1 = nn.Conv1d(self.n_channels, 128, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, self.n_classes)

    def forward(self, x: torch.Tensor):
        # No need to reshape input, assuming it comes in the correct shape (batch_size, channels, time_steps)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.avg_pool1d(x, 2)
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc4(x)

        return F.log_softmax(x, 1)

# small window size down to 6
class ConvNet1D_s(nn.Module):
    def __init__(self, n_in, n_classes, n_channels=3):
        super(ConvNet1D_s, self).__init__()
        self.n_in = n_in
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv1 = nn.Conv1d(self.n_channels, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, self.n_classes)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.avg_pool1d(x, kernel_size=2)  # We still perform pooling
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc4(x)

        return F.log_softmax(x, 1)

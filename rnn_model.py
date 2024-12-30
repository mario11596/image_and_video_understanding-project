import torch
import torch.nn as nn
import numpy as np

num_classes = 33
input_features = 63
seed_num = 302

np.random.seed(seed_num)
np.random.seed(seed_num)
torch.manual_seed(seed_num)

# in case GPU is  available
torch.cuda.manual_seed(seed_num)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Architecture of Residual neural network model
class ResidualBlockWithBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockWithBatchNorm, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResidualNeuralNetworkModel(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(ResidualNeuralNetworkModel, self).__init__()

        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

        self.layer1 = ResidualBlockWithBatchNorm(in_channels=128, out_channels=128)
        self.layer2 = ResidualBlockWithBatchNorm(in_channels=128, out_channels=128)
        self.layer3 = ResidualBlockWithBatchNorm(in_channels=128, out_channels=64)
        self.layer4 = ResidualBlockWithBatchNorm(in_channels=64, out_channels=64)
        self.layer5 = ResidualBlockWithBatchNorm(in_channels=64, out_channels=32)

        self.fc1 = nn.Linear(32 * 63, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

num_classes = 35
input_features = 63 * 2
num_epochs = 50
batch_size = 32
shuffle = False


class ResidualBlockWithBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockWithBatchNorm, self).__init__()

        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class RNNModel(nn.Module):
    def __init__(self, input_features=input_features, num_classes=num_classes):
        super(RNNModel, self).__init__()

        self.fc1 = nn.Linear(input_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.layer1 = ResidualBlockWithBatchNorm(in_channels=256, out_channels=512)
        self.layer2 = ResidualBlockWithBatchNorm(in_channels=512, out_channels=1024)

        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.fc2(x)
        return x


def train_validation():
    train_avg_losses = []
    val_avg_losses = []

    model.train()
    for epoch in range(num_epochs):
        train_losses = []

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_avg_losses.append( np.mean(train_losses) )

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                data, target = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += F.nll_loss(outputs, target, reduction='sum').item()
                pred = torch.argmax(outputs, dim=1)

        test_loss /= len(validation_loader.dataset)
        val_avg_losses.append(test_loss)

        print(f'\nAverage loss in epoch {epoch + 1}: Train {train_avg_losses[-1]}, Validation {val_avg_losses[-1]}')
    return train_avg_losses, val_avg_losses


def training_and_validation_plot(train_loss_plot, val_loss_plot):
    plt.plot(train_loss_plot, label='Training Loss')
    plt.plot(val_loss_plot, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig('training_validation.jpg')
    plt.close()


train_directory = r'.\dataset\test_data.npy'
train_labels_directory = r'.\dataset\test_labels.npy'
validation_directory = r'.\dataset\validation_data.npy'
validation_labels_directory = r'.\dataset\validation_labels.npy'

train_data = np.load(train_directory)
train_labels = np.load(train_labels_directory)
validation_data = np.load(validation_directory)
validation_labels = np.load(validation_labels_directory)

train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
validation_data_tensor = torch.tensor(validation_data, dtype=torch.float32)
validation_labels_tensor = torch.tensor(validation_labels, dtype=torch.long)

train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
validation_dataset = TensorDataset(validation_data_tensor, validation_labels_tensor)
validation_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss, val_loss = train_validation()
training_and_validation_plot(train_loss, val_loss)







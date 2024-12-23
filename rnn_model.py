import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

num_classes = 35
input_features = 63 * 2
num_epochs = 100
batch_size = 128
shuffle = True
training_validation_mode = False

classes = ('A', 'B', 'C', 'comma', 'D', 'del', 'E', 'exclamation mark', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'minus', 'N', 'nothing', 'O', 'P', 'parentheses', 'period', 'Q', 'question mark', 'R', 'S', 'space',
                'T', 'U', 'V', 'W', 'X', 'Y', 'Z')


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

        self.fc1 = nn.Linear(input_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

        self.layer1 = ResidualBlockWithBatchNorm(in_channels=128, out_channels=128)
        self.layer2 = ResidualBlockWithBatchNorm(in_channels=128, out_channels=128)
        self.layer3 = ResidualBlockWithBatchNorm(in_channels=128, out_channels=64)
        self.layer4 = ResidualBlockWithBatchNorm(in_channels=64, out_channels=64)
        self.layer5 = ResidualBlockWithBatchNorm(in_channels=64, out_channels=32)

        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
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
        train_avg_losses.append(np.mean(train_losses))

        model.eval()
        val_loss = []
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                data, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_loss.append( loss.item())

                pred = torch.argmax(outputs, dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        val_avg_losses.append( np.mean(val_loss) )
        avg_correct = correct / total

        print(f'\nAverage loss in epoch {epoch + 1}: Train {train_avg_losses[-1]}, Validation {val_avg_losses[-1]}, '
              f'Accuracy: {100. * avg_correct:.3f}%')
    return train_avg_losses, val_avg_losses


def training_and_validation_plot(train_loss_plot, val_loss_plot):
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_plot, label='Training Loss')
    plt.plot(val_loss_plot, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig('training_validation_rnn.jpg')
    plt.close()


def eval_classifier():
    model.eval()

    correct = 0
    test_loss = 0
    total = 0
    all_predicted = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = (correct / total) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    cm = confusion_matrix(all_labels, all_predicted)
    plt.figure(figsize=(16, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.jpg')


def set_seed(seed):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # in case GPU is  available
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(302)

train_directory = r'.\dataset\test_data.npy'
train_labels_directory = r'.\dataset\test_labels.npy'
validation_directory = r'.\dataset\validation_data.npy'
validation_labels_directory = r'.\dataset\validation_labels.npy'
test_directory = r'.\dataset\test_data.npy'
test_labels_directory = r'.\dataset\test_labels.npy'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)


if training_validation_mode:
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
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)

    train_loss, val_loss = train_validation()
    training_and_validation_plot(train_loss, val_loss)

    torch.save(model.state_dict(), "sign_language_rnn_model.pth")

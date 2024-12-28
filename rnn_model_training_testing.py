import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import rnn_model

num_classes = 34
input_features = 63 * 2
num_epochs = 2000
batch_size = 128
shuffle = True
training_validation_mode = False
seed_num = 302
learning_rate = 0.001

classes = ('A', 'B', 'C', 'comma', 'D', 'del', 'E', 'exclamation mark', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'minus',
                   'N', 'O', 'P', 'parentheses', 'period', 'Q', 'question mark', 'R', 'S', 'Space', 'T', 'U', 'V', 'W',
                   'X', 'Y', 'Z')


def train_validation():
    train_avg_losses = []
    val_avg_losses = []

    for epoch in range(num_epochs):
        train_losses = []

        model.train()
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

        print(f'\nAverage loss in epoch {epoch + 1}/{num_epochs}: Train {train_avg_losses[-1]}, Validation {val_avg_losses[-1]}, '
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


set_seed(seed_num)

train_directory = r'.\dataset1\train_data.npy'
train_labels_directory = r'.\dataset1\train_labels.npy'
validation_directory = r'.\dataset1\validation_data.npy'
validation_labels_directory = r'.\dataset1\validation_labels.npy'
test_directory = r'.\dataset1\test_data.npy'
test_labels_directory = r'.\dataset1\test_labels.npy'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = rnn_model.RNNModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training/validation parts
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

# Testing parts
else:
    model.load_state_dict(torch.load("sign_language_rnn_model.pth"))

    test_data = np.load(test_directory)
    test_labels = np.load(test_labels_directory)

    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    eval_classifier()
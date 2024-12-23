import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


num_classes = 35
num_epochs = 30
batch_size = 32
learning_rate = 0.001
step_size = 5
gamma = 0.1
seed=17

def set_seed(seed=17):
    
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    #in case GPU is  available
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

class lmk_cnn(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(lmk_cnn, self).__init__()
        self.input_shape = input_shape  # (channels, height, width)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        flat_size = self.calculate_flat_size()

        self.fc1 = nn.Linear(flat_size, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def calculate_flat_size(self):
        with torch.no_grad():
            x = torch.zeros(1, *self.input_shape)  # Create a dummy input
            x = self.pool1(self.bn1(self.conv1(x)))
            x = self.pool2(self.bn2(self.conv2(x)))
            x = self.pool3(self.bn3(self.conv3(x)))
            # x = self.pool4(self.bn4(self.conv4(x)))
            # x = self.pool5(self.bn5(self.conv5(x)))
            return x.numel()

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # x = torch.relu(self.bn4(self.conv4(x)))
        # x = self.pool4(x)

        # x = torch.relu(self.bn5(self.conv5(x)))
        # x = self.pool5(x)

        x = torch.flatten(x, 1)  
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class HandSignDataset(Dataset):
    def __init__(self, features_file, labels_file, input_shape):
        self.features = np.load(features_file)
        self.labels = np.load(labels_file)

        print("Unique labels in training data:", np.unique(self.labels))

        # Normalize the features
        self.features = self.features / np.max(self.features, axis=0)

        # Reshape into 2D and add channel dimension for CNN
        self.features = self.features.reshape(-1, 1, *input_shape)

        # Convert to tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0

    for features, labels in dataloader:
        
        features, labels = features.to(device), labels.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy


def train_model_main(train_features, train_labels, val_features, val_labels, num_classes, input_shape=(9, 14), 
                     batch_size=32, epochs=30, learning_rate=0.0015, step_size=3, gamma=0.15):
    
    train_dataset = HandSignDataset(train_features, train_labels, input_shape)
    val_dataset = HandSignDataset(val_features, val_labels, input_shape)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model, loss function, optimizer, and scheduler
    model = lmk_cnn((1, *input_shape), num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step()

    # Save the trained model
    torch.save(model.state_dict(), "sign_language_cnn_model.pth")
    print("Model saved to sign_language_cnn_model.pth")

    return model  # Return the trained model for testing


def test_model_main(test_features, test_labels, input_shape=(9, 14), batch_size=32, class_names=None, output_path="confusion_matrix.png"):
    
    test_dataset = HandSignDataset(test_features, test_labels, input_shape)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lmk_cnn((1, *input_shape), num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    model.to(device)  

    #reload saved model
    model.load_state_dict(torch.load("sign_language_cnn_model.pth", map_location=device))
    model.eval()


    all_preds = []
    all_labels = []

    total_loss = 0
    correct = 0

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            
            
            # Store predictions and labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
                
            

    # Calculate metrics
    test_loss = total_loss / len(test_loader)
    test_acc = correct / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Optionally display the confusion matrix as a heatmap
    if class_names is None:
        class_names = [str(i) for i in range(conf_matrix.shape[0])]

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save the confusion matrix as an image
    plt.savefig(output_path)
    print(f"Confusion matrix saved as {output_path}")

    # Show the confusion matrix
    plt.show()

    # Optionally display a classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return test_loss, test_acc, conf_matrix  # Return the confusion matrix as well


def test_model_with_threshold(test_features, test_labels, input_shape=(9, 14), batch_size=32, class_names=None, output_path="confusion_matrix.png", threshold=0.5):
    """
    Test the model and discard predictions with softmax probability below a given threshold.
    """
    test_dataset = HandSignDataset(test_features, test_labels, input_shape)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lmk_cnn((1, *input_shape), num_classes).to(device)
    model.load_state_dict(torch.load("sign_language_cnn_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    discarded_count = 0

    total_loss = 0
    correct = 0

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probabilities, dim=1)

            # Discard predictions with max probability below threshold
            for i in range(features.size(0)):
                if max_probs[i] < threshold:
                    discarded_count += 1
                    continue  # Skip this prediction
                
                all_preds.append(preds[i].cpu().item())
                all_labels.append(labels[i].cpu().item())

                if preds[i] == labels[i]:
                    correct += 1

    # Calculate metrics
    total_samples = len(test_loader.dataset)
    valid_samples = total_samples - discarded_count
    accuracy = correct / valid_samples if valid_samples > 0 else 0.0

    print(f"Total samples: {total_samples}")
    print(f"Valid samples (above threshold): {valid_samples}")
    print(f"Discarded samples (below threshold): {discarded_count}")
    print(f"Accuracy: {accuracy:.4f}")

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Optionally display the confusion matrix as a heatmap
    if class_names is None:
        class_names = [str(i) for i in range(conf_matrix.shape[0])]

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save the confusion matrix as an image
    plt.savefig(output_path)
    print(f"Confusion matrix saved as {output_path}")

    # Show the confusion matrix
    plt.show()

    # Optionally display a classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return accuracy, conf_matrix  # Return accuracy and confusion matrix



if __name__ == "__main__":
    set_seed(seed)

    train_features = r".\dataset\train_data_200.npy"
    train_labels = r".\dataset\train_labels_200.npy"
    val_features = r".\dataset\validation_data_200.npy"
    val_labels = r".\dataset\validation_labels_200.npy"
    test_features = r".\dataset\test_data_200.npy"
    test_labels = r".\dataset\test_labels_200.npy"

    # Train the model
    # trained_model = train_model_main(train_features, train_labels, val_features, val_labels, num_classes, (9, 14), 
    #                                   batch_size, num_epochs, learning_rate, step_size, gamma)
    
    class_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L','M', 'N', 'nothing', 'O', 'P', 
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'comma', 'exclamation mark', 'minus', 'nothing', 
        'parentheses', 'period', 'question mark', 'space')
    
    
    #test_model_main(test_features, test_labels, (9, 14), batch_size, class_names, output_path="confusion_matrix_test_2.png")
    test_model_with_threshold(test_features, test_labels, (9, 14), batch_size, class_names, output_path="confusion_matrix_test_2.png")
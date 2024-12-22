import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


num_classes = 35
num_epochs = 30
batch_size = 32
learning_rate = 0.001
step_size = 3
gamma = 0.15
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

def main(train_features, train_labels, val_features, val_labels, test_features, test_labels, 
         num_classes, input_shape=(9,14), batch_size=32, epochs=30, learning_rate=0.0015, step_size=3, gamma=0.15):
    # Create datasets and data loaders
    train_dataset = HandSignDataset(train_features, train_labels, input_shape)
    val_dataset = HandSignDataset(val_features, val_labels, input_shape)
    test_dataset = HandSignDataset(test_features, test_labels, input_shape)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

        if (epoch+1)%5==0:
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
            print("------------------------\nAT ", epoch+1, "EPOCHS")
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
            print(f"Test Acc: {test_acc:.4f}", "Epochs:", epoch+1, "LR:", learning_rate, "Step:", step_size, "Gamma:", gamma)

        scheduler.step()


    #Evaluation on the test set
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}", "Epochs:", num_epochs, "LR:", learning_rate, "Step:", step_size, "Gamma:", gamma)

    # Save the model
    torch.save(model.state_dict(), "sign_language_cnn_model.pth")
    print("Model saved to sign_language_cnn_model.pth")

def filter_nothing():
    
    sets = {
        "train": {
            "features": r".\dataset\train_data.npy",
            "labels": r".\dataset\train_labels.npy"
        },
        "validation": {
            "features": r".\dataset\validation_data.npy",
            "labels": r".\dataset\validation_labels.npy"
        },
        "test": {
            "features": r".\dataset\test_data.npy",
            "labels": r".\dataset\test_labels.npy"
        }
    }

    # Iterate over each dataset (train, validation, test)
    for split, paths in sets.items():
        # Load features and labels
        features = np.load(paths["features"])
        labels = np.load(paths["labels"])

        # Remove the label 18 itself ("nothing")
        valid_indices = labels != 18
        filtered_features = features[valid_indices]
        filtered_labels = labels[valid_indices]
        # Adjust the labels
        filtered_labels = np.where(filtered_labels > 18, filtered_labels - 1, filtered_labels)
        

        # Save the adjusted features and labels with new filenames
        filtered_features_path = paths["features"].replace(".npy", "_filtered.npy")
        filtered_labels_path = paths["labels"].replace(".npy", "_filtered.npy")
        np.save(filtered_features_path, filtered_features)
        np.save(filtered_labels_path, filtered_labels)

        print(f"Adjusted {split} data saved:")
        print(f"  Original size: {features.shape[0]}")
        print(f"  Adjusted size: {filtered_features.shape[0]}")
        print(f"  Adjusted features saved to: {filtered_features_path}")
        print(f"  Adjusted labels saved to: {filtered_labels_path}\n")

if __name__ == "__main__": 
    set_seed(seed)

    train_features = r".\dataset.\train_data.npy"  
    train_labels = r".\dataset.\train_labels.npy"      
    val_features = r".\dataset.\validation_data.npy"      
    val_labels = r".\dataset.\validation_labels.npy"          
    test_features = r".\dataset.\test_data.npy"    
    test_labels = r".\dataset.\test_labels.npy"  
    

    # filter_nothing()
    # train_features = r".\dataset\train_data_filtered.npy"
    # train_labels = r".\dataset\train_labels_filtered.npy"
    # val_features = r".\dataset\validation_data_filtered.npy"
    # val_labels = r".\dataset\validation_labels_filtered.npy"
    # test_features = r".\dataset\test_data_filtered.npy"
    # test_labels = r".\dataset\test_labels_filtered.npy"                          
    
    main(train_features, train_labels, val_features, val_labels, test_features, test_labels, num_classes, (9, 14), batch_size, num_epochs, learning_rate, step_size, gamma)
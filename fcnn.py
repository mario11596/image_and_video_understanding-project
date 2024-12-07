import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


num_classes = 35
num_epochs = 50
batch_size = 32
learning_rate = 0.0015
step_size = 5
gamma = 0.1
seed=17

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

# Fully Connected Neural Network
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # First fully connected layer
        self.dropout1 = nn.Dropout(0.3)       # Dropout for regularization
        self.fc2 = nn.Linear(256, 128)        # Second fully connected layer
        self.dropout2 = nn.Dropout(0.3)       # Dropout for regularization
        self.fc3 = nn.Linear(128, num_classes)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply activation after first layer
        x = self.dropout1(x)         # Apply dropout
        x = torch.relu(self.fc2(x))  # Apply activation after second layer
        x = self.dropout2(x)         # Apply dropout
        x = self.fc3(x)              # Output layer (no activation here; handled by loss)
        return x

# Dataset for Hand Signs
class HandSignDataset(Dataset):
    def __init__(self, features_file, labels_file):
        self.features = np.load(features_file)
        self.labels = np.load(labels_file)

        # Normalize the features (scale between 0 and 1)
        self.features = self.features / np.max(self.features, axis=0)

        # Convert to tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Training function
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

# Evaluation function
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

# Main function
def main(train_features, train_labels, val_features, val_labels, test_features, test_labels, 
         num_classes, input_size, batch_size=32, epochs=30, learning_rate=0.0015, step_size=3, gamma=0.15):
    # Create datasets and data loaders
    train_dataset = HandSignDataset(train_features, train_labels)
    val_dataset = HandSignDataset(val_features, val_labels)
    test_dataset = HandSignDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define device, model, loss function, optimizer, and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FullyConnectedNN(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

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

    # Final evaluation on the test set
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save the model
    torch.save(model.state_dict(), "sign_language_fcn_model.pth")
    print("Model saved to sign_language_fcn_model.pth")

# Main script execution
if __name__ == "__main__": 
    set_seed(seed=17)

    train_features = r".\dataset.\train_data.npy"  
    train_labels = r".\dataset.\train_labels.npy"      
    val_features = r".\dataset.\validation_data.npy"      
    val_labels = r".\dataset.\validation_labels.npy"          
    test_features = r".\dataset.\test_data.npy"    
    test_labels = r".\dataset.\test_labels.npy" 

    # Set the input size to match the length of the feature vectors
    input_size = 126  # Example: 126 features in each vector (adjust based on actual data)

    main(train_features, train_labels, val_features, val_labels, test_features, test_labels, 
         num_classes, 126, batch_size, num_epochs, 
         learning_rate, step_size, gamma)
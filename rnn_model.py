import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets

folder_names = ['A', 'B', 'C', 'comma', 'D', 'del', 'E', 'exclamation mark', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'minus', 'N', 'nothing', 'O', 'P', 'parentheses', 'period', 'Q', 'question mark', 'R', 'S', 'space',
                'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
batch_size = 100
input_size = 28
hidden_size = 16
num_layers = 2
num_classes = len(folder_names)
num_epochs = 2


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='tanh')

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.rnn(x, h0)  # out: (batch_size, seq_length, hidden_size)

        out = self.fc(out[:, -1, :])  # out: (batch_size, num_classes)
        return out


train_directory = r'.\dataset\train'
validation_directory = r'.\dataset\validation'
test_directory = r'.\dataset\test'

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(root=train_directory, transform=transform)
validation_dataset = datasets.ImageFolder(root=validation_directory, transform=transform)
test_dataset = datasets.ImageFolder(root=test_directory, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNModel(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_log = []
accuracy_log = []
val_loss_log = []
val_acc_log = []

print('Training started')
for epoch in range(num_epochs):
    correct_predictions = 0
    total_loss = 0

    for inputs, labels in validation_dataset:
        inputs = inputs.reshape(-1, 28, 28)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predicted = torch.argmax(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_loss += loss.item()

    accuracy = correct_predictions / (len(train_loader) * train_loader.batch_size)
    loss = total_loss / len(train_loader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss:.4f}')

    # Testing loop
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28, 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
print('Training ended')







import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Initialize fc1 as None; will set it later
        self.fc1 = None

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        if self.fc1 is None:
            # Dynamically determine the input features for fc1
            self.fc1 = nn.Linear(x.shape[1], 10).to(x.device)
        x = self.fc1(x)
        return x

# Hyperparameters
batch_size = 64
epochs = 1
learning_rate = 0.01

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(
    './data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    './data', train=False, transform=transform)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Initialize the model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
def train():
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    print(f'\nAverage Training Loss: {avg_loss:.6f}\n')

# Testing loop
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}')
        train()
        test()


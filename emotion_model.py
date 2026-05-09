import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("mps")
print(f"Training on: {device}")

# Data
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.ImageFolder(
    root='fer2013/train', transform=transform)
testset  = torchvision.datasets.ImageFolder(
    root='fer2013/test',  transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = DataLoader(testset,  batch_size=64, shuffle=False)

print(f"Classes: {trainset.classes}")
print(f"Training images: {len(trainset)}")
print(f"Test images:     {len(testset)}")

# Model
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 6 * 6, 512)
        self.fc2   = nn.Linear(512, num_classes)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 48 → 24
        x = self.pool(self.relu(self.conv2(x)))  # 24 → 12
        x = self.pool(self.relu(self.conv3(x)))  # 12 → 6
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

num_classes = len(trainset.classes)
model = EmotionCNN(num_classes).to(device)
print(f"\nModel ready — {num_classes} emotion classes")

# Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nStarting training...")
for epoch in range(15):
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/15 — loss: {running_loss/len(trainloader):.3f}")

# Evaluate
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTest accuracy: {accuracy:.2f}%")

# Save
torch.save(model.state_dict(), 'emotion_cnn.pth')
torch.save(trainset.classes, 'emotion_classes.pth')
print("Model saved to emotion_cnn.pth")
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

device = torch.device("mps")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = torchvision.datasets.ImageFolder(root='fer2013/test', transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
classes = testset.classes

print(f"Classes: {classes}")
print(f"Test images: {len(testset)}")

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
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = EmotionCNN(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load('emotion_cnn.pth', map_location=device))
model.eval()
print("Model loaded.")

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Per-class accuracy
print("\nPer-class accuracy:")
for i, cls in enumerate(classes):
    mask = all_labels == i
    cls_acc = 100 * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
    print(f"  {cls:10s}: {cls_acc:.1f}%")

overall = 100 * (all_preds == all_labels).sum() / len(all_labels)
print(f"\nOverall accuracy: {overall:.2f}%")

# Save results to JSON
results = {
    "overall_accuracy": round(float(overall), 2),
    "per_class_accuracy": {
        cls: round(float(100 * (all_preds[all_labels == i] == i).sum() / (all_labels == i).sum()), 2)
        for i, cls in enumerate(classes)
    }
}
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results.json")

# Confusion matrix plot
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im)
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.set_yticklabels(classes)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Emotion CNN — Confusion Matrix')

for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=9)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("Confusion matrix saved to confusion_matrix.png")
plt.show()
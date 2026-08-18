import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# -----------------------------
# 1. Select device
# -----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# -----------------------------
# 2. Image transformations
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 3. Load dataset
# -----------------------------
train_path = "dataset/food11/train"
test_path = "dataset/food11/test"

train_dataset = datasets.ImageFolder(
    train_path,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    test_path,
    transform=test_transform
)

print("Food classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# -----------------------------
# 4. Load pretrained ResNet18
# -----------------------------
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze the existing layers
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace final layer for 11 food classes
number_of_classes = len(train_dataset.classes)

model.fc = nn.Linear(
    model.fc.in_features,
    number_of_classes
)

model = model.to(device)

# -----------------------------
# 5. Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

# -----------------------------
# 6. Train model
# -----------------------------
epochs = 1

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )

# -----------------------------
# 7. Test model
# -----------------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total

print(f"Test Accuracy: {test_accuracy:.2f}%")

# -----------------------------
# 8. Save model
# -----------------------------
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": train_dataset.classes
    },
    "food_model.pth"
)

print("Model saved as food_model.pth")
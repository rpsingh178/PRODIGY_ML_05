import torch
from calorie_data import calorie_data
from torch import nn
from torchvision import models, transforms
from PIL import Image

# Select device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# Load trained model
checkpoint = torch.load(
    "food_model.pth",
    map_location=device,
    weights_only=False
)

classes = checkpoint["classes"]

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

image = Image.open("test_food.jpg").convert("RGB")

image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)
image_tensor = image_tensor.to(device)

# Make prediction
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)

    confidence, prediction = torch.max(
        probabilities, 1
    )

food_name = classes[prediction.item()]
confidence_value = confidence.item() * 100
calories_per_100g = calorie_data.get(food_name, 0)

serving_size = 100

estimated_calories = (
    calories_per_100g * serving_size / 100
)
print("--------------------------------")
print("FOOD & CALORIE PREDICTION")
print("--------------------------------")
print("Food:", food_name)
print(f"Confidence: {confidence_value:.2f}%")
print(f"Serving size: {serving_size} grams")
print(f"Estimated calories: {estimated_calories:.0f} kcal")
print("--------------------------------")
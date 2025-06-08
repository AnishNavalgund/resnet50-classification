import os
import torch
import torchvision
from torchvision import transforms, datasets, models
import numpy as np
from sklearn.metrics import accuracy_score

val_dir = "data/split/val"  # or use "data/split/test" to evaluate on test set
model_path = "models/resnet50_flower.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(val_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)


model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.classes))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

top1_correct = 0
top5_correct = 0
total = 0

with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Top-5 predictions
        _, top5_preds = outputs.topk(5, dim=1)

        # Top-1
        top1_preds = top5_preds[:, 0]
        top1_correct += (top1_preds == labels).sum().item()

        # Top-5
        for i in range(labels.size(0)):
            if labels[i].item() in top5_preds[i]:
                top5_correct += 1

        total += labels.size(0)

top1_acc = top1_correct / total
top5_acc = top5_correct / total

print(f"Validation Results")
print(f"Top-1 Accuracy: {top1_acc:.4f}")
print(f"Top-5 Accuracy: {top5_acc:.4f}")

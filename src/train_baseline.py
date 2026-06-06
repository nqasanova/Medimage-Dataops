from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms

import medmnist
from medmnist import INFO

# Configuration

DATASET_NAME = "pathmnist"
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CNN Model

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Main

def main():

    output_dir = Path("data/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    info = INFO[DATASET_NAME]
    num_classes = len(info["label"])

    DataClass = getattr(medmnist, info["python_class"])

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = DataClass(
        split="train",
        root="data/raw",
        transform=transform,
        download=False
    )

    test_dataset = DataClass(
        split="test",
        root="data/raw",
        transform=transform,
        download=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = SimpleCNN(num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []

    print(f"Training on: {DEVICE}")

    # Training Loop

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        train_losses.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

    # Evaluation

    model.eval()

    all_preds = []
    all_labels = []

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total

    print(f"\nTest Accuracy: {accuracy:.2f}%")

    # Save Classification Report

    class_names = list(info["label"].values())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names
    )

    report_path = output_dir / "classification_report.txt"

    with open(report_path, "w") as f:
        f.write(report)

    print(f"Saved classification report to {report_path}")

    # Training Loss Plot

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    loss_plot_path = output_dir / "training_loss.png"

    plt.savefig(loss_plot_path, dpi=200)
    plt.close()

    print(f"Saved training plot to {loss_plot_path}")

    # Confusion Matrix

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    cm_path = output_dir / "confusion_matrix.png"

    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
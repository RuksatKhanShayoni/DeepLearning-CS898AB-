import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Training Setup
    DATA_DIR = "/Users/ruksatkhanshayoni/Downloads/tiny-imagenet-output" # Preprocessed Dataset directory
    BATCH_SIZE = 128 # Number of images per training batch
    NUM_EPOCHS = 10 # How many times the model will see the entire training set
    NUM_CLASSES = 100 # Total number of classes in Tiny ImageNet subset
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224), # Image translation: crop 224x224
        transforms.RandomHorizontalFlip(), # Horizontal reflection
        transforms.ToTensor(), # Convert to PyTorch tensor [0,1]
    ])

    # Validation Image Transformation
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    #Load the dataset from folders with transforms applied
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Custom AlexNet
    class CustomAlexNet(nn.Module):
        def __init__(self, num_classes=100):
            super(CustomAlexNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4), # Conv1
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(5),
                nn.MaxPool2d(3, 2),
                nn.Conv2d(96, 256, kernel_size=5, padding=2), # Conv2
                nn.ReLU(inplace=True),
                nn.LocalResponseNorm(5),
                nn.MaxPool2d(3, 2),
                nn.Conv2d(256, 384, kernel_size=3, padding=1), # Conv3
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=3, padding=1), # Conv4
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1), # Conv5
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2)
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.5), # Dropout to prevent overfitting
                nn.Linear(256 * 5 * 5, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes)
            )
            self._init_weights()

        # Weight & Bias Initialization
        def _init_weights(self):  # Initialize weights and biases as described in paper
            i = 0
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        if isinstance(m, nn.Conv2d) and i in [4, 8, 10]:   # Bias=1 for Conv2, 4, 5
                            nn.init.constant_(m.bias, 1)
                        elif isinstance(m, nn.Linear):
                            nn.init.constant_(m.bias, 1)
                        else:
                            nn.init.constant_(m.bias, 0)
                i += 1

        def forward(self, x):
            x = self.features(x) # Pass convolutional layers
            x = torch.flatten(x, 1) # Flatten for fully-connected layers
            x = self.classifier(x)
            return x

    # Training loop
    model = CustomAlexNet(NUM_CLASSES).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005) # Paper's optimizer
    criterion = nn.CrossEntropyLoss() # for multi-class classification
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.1) # LR scheduler

    # Store history
    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    #Top-K Accuracy Calculation
    def top_k_accuracy(output, target, topk=(1, 5)):
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k predictions (shape: [batch_size, maxk])
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)

        # Transpose to shape [maxk, batch_size]
        pred = pred.t()

        # Compare with target expanded to [1, batch_size]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        return [correct[:k].reshape(-1).float().sum(0) / batch_size for k in topk]


    # Training loop
    print("Model training has started...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_correct, total_loss = 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += out.argmax(1).eq(y).sum().item()

        model.eval()
        val_correct, val_total, val_top1, val_top5 = 0, 0, 0, 0
        v_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                v_loss += criterion(out, y).item()
                top1, top5 = top_k_accuracy(out, y)
                val_top1 += top1.item()
                val_top5 += top5.item()

        train_loss.append(total_loss / len(train_loader))
        train_acc.append(100. * total_correct / len(train_dataset))
        val_loss.append(v_loss / len(val_loader))
        val_acc.append(100. * val_top1 / len(val_loader))
        scheduler.step(v_loss)

        print(f"Epoch {epoch+1}: Training Accuracy {train_acc[-1]:.2f}%, Validation Accuracy {val_acc[-1]:.2f}%, Top-1: {val_top1/len(val_loader):.2f}%, Top-5: {val_top5/len(val_loader):.2f}%")

    print("Model training completed successfully!")

    # Plot results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.title("Loss per Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Training Accurarcy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy per Epoch")
    plt.show()

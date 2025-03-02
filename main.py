import torch #PyTorch Library
import torch.nn as nn #neural network layers and loss functions
import torch.optim as optim #optimization algorithms
import torchvision #datasets, transforms, and models
import torchvision.transforms as transforms
import time #measure training time per epoch
import matplotlib.pyplot as plt #plotting graphs

# Define device for training (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(), #Converts images into PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalizes pixel values from (0 to 255) to (-1 to 1)
])

#define batch size
batch_size = 128

#Downloads CIFAR-10 dataset, loads training dataset and transform
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

#Downloads CIFAR-10 dataset,  loads testing dataset  and transform
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# Define CNN Model
class CNN(nn.Module):
    def __init__(self, activation_function):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # Max pooling reduces image size by 2x2
        self.activation = activation_function

        #fully connected layers
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 10)

    #forward pass
    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.pool(self.activation(self.conv3(x)))
        x = self.pool(self.activation(self.conv4(x)))
        x = torch.flatten(x, 1)  # Flatten tensor into a 1D vector
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


# Training Function
def train_model(activation_fn, activation_name):
    model = CNN(activation_fn).to(device) #creates a CNN model with the given activation function
    criterion = nn.CrossEntropyLoss() #loss function for classification.
    optimizer = optim.Adam(model.parameters(), lr=0.001) #update weights using Adam

    epochs = 50  # Upper limit, but we stop at 25% error
    times_per_epoch = [] #Keep track of training time per epoch
    train_errors = [] #stores time per epoch.

    for epoch in range(epochs):
        start_time = time.time()
        correct, total = 0, 0
        running_loss = 0.0

        # This loop iterates the dataset, calculate loss, and update the model
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() # Clears gradients
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) #loss
            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            running_loss += loss.item()  # Accumulate total loss

            # Get predicted labels
            predicted_labels = torch.argmax(outputs, dim=1)

            # Update total number of samples
            total_samples = labels.size(0)
            total += total_samples

            # Count correct predictions
            correct_predictions = (predicted_labels == labels).sum().item()
            correct += correct_predictions

        #Calculate epoch time
        epoch_time = time.time() - start_time
        times_per_epoch.append(epoch_time)

        #Training Error
        train_error = 1 - (correct / total)
        train_errors.append(train_error)

        print(
            f"Epoch {epoch + 1}, Loss: {running_loss:.4f}, Training Error: {train_error:.4f}, Time: {epoch_time:.2f}s")

        if train_error <= 0.25:  # Stop training when error reaches 25%
            break

    return times_per_epoch, train_errors


# Activation Function ReLU, Tanh, Sigmoid
relu_times, relu_errors = train_model(nn.ReLU(), "ReLU")
tanh_times, tanh_errors = train_model(nn.Tanh(), "Tanh")
sigmoid_times, sigmoid_errors = train_model(nn.Sigmoid(), "Sigmoid")

# Graph Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(relu_errors) + 1), relu_errors, label="ReLU", marker='o')
plt.plot(range(1, len(tanh_errors) + 1), tanh_errors, label="Tanh", marker='s')
plt.plot(range(1, len(sigmoid_errors) + 1), sigmoid_errors, label="Sigmoid", marker='^')

plt.xlabel("Epochs")
plt.ylabel("Training Error Rate")
plt.title("Training Time per Epoch for ReLU, Tanh and Sigmoid")
plt.legend()
plt.grid()
plt.show()

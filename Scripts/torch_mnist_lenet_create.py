import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the model using nn.Sequential
net = nn.Sequential(
    nn.Conv2d(1, 20, 5),                # Conv layer: 1 input channel, 20 output channels, 5x5 kernel
    nn.ReLU(),
    nn.MaxPool2d(2),                    # Max pooling layer with 2x2 window
    nn.Conv2d(20, 50, 5),               # Conv layer: 20 input channels, 50 output channels, 5x5 kernel
    nn.ReLU(),
    nn.MaxPool2d(2),                    # Max pooling layer with 2x2 window
    nn.Flatten(),                        # Flatten the output into 1D vector
    nn.ReLU(),
    nn.Linear(50 * 4 * 4, 500),          # Fully connected layer (input size = 50 * 4 * 4)
    nn.ReLU(),
    nn.Linear(500, 10)                   # Final output layer (10 classes for MNIST)
)

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Training the model
epochs = 5
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = net(inputs)  # Forward pass
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

# Save the trained model to a .pth file
torch.save(net.state_dict(), 'custom_cnn_mnist.pth')

print("Model saved as custom_cnn_mnist.pth")

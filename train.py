import os
import math
import time
import linear
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CustomTransform:
    def __init__(self, target_size=(28, 28)):
        self.target_size = target_size

    def __call__(self, img):
        # Resize
        # img = img.resize(self.target_size, Image.ANTIALIAS)
        # Convert to NumPy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(img_array)

        return img_tensor
class MNISTDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data = []

        for label in os.listdir(root):
            label_path = os.path.join(root, label)
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                if(label.isdigit()):
                    self.data.append((img_path, int(label)))
                else:
                    self.data.append((img_path,-1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        # img = Image.open(img_path)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)   # [label]
        return img, label
class CustomDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_idx = 0

        if shuffle:
            self.shuffle_dataset()

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        self.idx_list = list(range(len(self.dataset)))
        self.batch_start_idx = 0
        return self

    def __next__(self):
        if self.batch_start_idx >= len(self.idx_list):
            raise StopIteration

        batch_end_index = min(self.batch_start_idx + self.batch_size, len(self.dataset))
        batch_indices = self.idx_list[self.batch_start_idx:batch_end_index]

        batch = [self.dataset[i] for i in batch_indices]

        # Transpose the list of tuples into a tuple of lists
        batch = list(zip(*batch))

        # Convert the data and labels to tensors
        data_tensor = torch.stack([torch.Tensor(item) for item in batch[0]])
        label_tensor = torch.stack([torch.Tensor(item) for item in batch[1]])

        # self.current_idx += self.batch_size
        self.batch_start_idx = batch_end_index

        return data_tensor, label_tensor

    def shuffle_dataset(self):
        indices = torch.randperm(len(self.dataset))
        self.dataset = [self.dataset[i] for i in indices]
        self.current_idx = 0

# Set the paths for training and testing data
train_data_path = r"MNIST_DATASET/trainingSet/trainingSet" #/trainingSet"
test_data_path = r"MNIST_DATASET/testSet"

# Create the MNISTDataset instance with the custom transformation
custom_transform = CustomTransform()
train_dataset = MNISTDataset(root=train_data_path, transform=custom_transform)
test_dataset = MNISTDataset(root=test_data_path, transform=custom_transform)

# Split the dataset into training, validation, and test sets
total_size = len(train_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.5 * (total_size - train_size))
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(train_dataset, [train_size, val_size, test_size])
print(f"{len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")

# Create custom data loaders for training, validation, and testing
train_dataloader = CustomDataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
val_dataloader = CustomDataLoader(dataset=val_dataset, batch_size=128, shuffle=False)
test_dataloader = CustomDataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

activation_function = "Sigmoid" # [Sigmoid,ReLU]
HeInitialization = False
model = linear.Linear(activation_type=activation_function)
learning_rate = 0.03
num_epochs = 60

train_losses, val_losses, test_losses, train_accuracies, val_accuracies, test_accuracies = [], [], [], [], [], []

print(f"Training model of 5 layers with 32 neurons each for {num_epochs} epochs \n[learning rate is {learning_rate}]\n[Activation Function is {activation_function}]\n[HeInitialization is {HeInitialization}]\n")

starting_time_scratch = time.time()
for epoch in range(num_epochs):
    print(f"Epoch [{str(epoch + 1).zfill(len(str(num_epochs)))}/{num_epochs}], ", end=" ")

    # Training
    train_running_loss, correct_train, total_train = 0.0, 0, 0

    for data, labels in train_dataloader:
        data = data.view(-1, 784)
        labels = labels.squeeze()
        labels_one_hot = F.one_hot(labels, num_classes=10).float()

        y_pred = model.forward(data)

        loss = model.compute_loss(y_pred, labels_one_hot)
        model._backward(data, labels_one_hot, y_pred, learning_rate)

        train_running_loss += loss
        total_train += labels.size(0)
        y_pred = torch.argmax(y_pred, dim=1)
        correct_train += (y_pred == labels).sum().item()

    train_losses.append(train_running_loss / len(train_dataloader))
    train_accuracies.append(100 * correct_train / total_train)

    # Validation
    val_running_loss, correct_val, total_val = 0.0, 0, 0

    with torch.no_grad():
        for data, labels in val_dataloader:
            data = data.view(-1, 784)
            labels = labels.squeeze()
            labels_one_hot = F.one_hot(labels, num_classes=10).float()

            y_pred = model.forward(data)
            loss = model.compute_loss(y_pred, labels_one_hot)

            val_running_loss += loss
            total_val += labels.size(0)
            y_pred = torch.argmax(y_pred, dim=1)
            correct_val += (y_pred == labels).sum().item()

    val_losses.append(val_running_loss / len(val_dataloader))
    val_accuracies.append(100 * correct_val / total_val)

    # Testing
    test_running_loss, correct_test, total_test = 0.0, 0, 0

    with torch.no_grad():
        for data, labels in test_dataloader:
            data = data.view(-1, 784)
            labels = labels.squeeze()
            labels_one_hot = F.one_hot(labels, num_classes=10).float()

            y_pred = model.forward(data)
            loss = model.compute_loss(y_pred, labels_one_hot)

            test_running_loss += loss
            total_test += labels.size(0)
            y_pred = torch.argmax(y_pred, dim=1)
            correct_test += (y_pred == labels).sum().item()

    test_losses.append(test_running_loss / len(test_dataloader))
    test_accuracies.append(100 * correct_test / total_test)

    print(
        f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.2f}%, "
        f"Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.2f}%, "
        f"Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.2f}%"
    )

ending_time_scratch = time.time()
full_run_scratch = ending_time_scratch-starting_time_scratch
print(f'Total Time taken = {full_run_scratch}')
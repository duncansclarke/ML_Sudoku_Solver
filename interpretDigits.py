''' This file uses PyTorch and the MNIST dataset to recognize the digits of the sudoku puzzle, and write the data of the sudoku puzzle to Puzzle.txt
'''

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import cv2
import os

# Define transformation to Torch sensor
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download datasets
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET',
                          download=True, train=True, transform=transform)
testset = datasets.MNIST('PATH_TO_STORE_TESTSET',
                         download=True, train=False, transform=transform)
# Load datasets
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


dataiterator = iter(trainloader)
images, labels = dataiterator.next()

''' Build neural network '''

# Instantiate layer sizes
input_size = 784  # 28x28 pixels = 784 pixels when flattened
hidden_sizes = [128, 64]
output_size = 10

# Wrap layers in network with ReLU activation and LogSoftmax activation for output layer
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

# Instantiate negative log likelihood loss
criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images)
loss = criterion(logps, labels)

# Adjust the weights
loss.backward()

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Convert mnist images to vector of size 784
        images = images.view(images.shape[0], -1)

        # Training
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # Backpropagating
        loss.backward()

        # Optimizing weights
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e,
                                                    running_loss/len(trainloader)))
print("\nTraining Time =", (time()-time0)/60)

images, labels = next(iter(testloader))


# Read number cell images to a list
number_images = []
for filename in os.listdir('NumberCells'):
    img = cv2.imread(os.path.join('NumberCells', filename))
    if img is not None:
        number_images.append(img)

# Apply final preprocessing to each number image
for i in range(len(number_images)):
    number_images[i] = number_images[i].copy()
    number_images[i] = cv2.cvtColor(number_images[i], cv2.COLOR_BGR2GRAY)
    # Apply some thresholding
    number_images[i] = cv2.threshold(
        number_images[i], 40, 255, cv2.THRESH_BINARY)[1]
    # Apply some blurring and filtering to improve accuracy
    number_images[i] = cv2.GaussianBlur(number_images[i], (5, 5), 0)
    number_images[i] = cv2.medianBlur(number_images[i], 5)
    number_images[i] = cv2.bilateralFilter(number_images[i], 9, 125, 125)
    # Resize to 28x28 pixels
    number_images[i] = cv2.resize(number_images[i], (28, 28))
    number_images[i] = number_images[i].astype('float32')
    # Reshape into format for Torch
    number_images[i] = number_images[i].reshape(1, 28, 28, 1)
    number_images[i] /= 255

# Recognize each number image and save them in a list
numbers = []
for img in number_images:
    features = (torch.from_numpy(img))
    features = features.reshape(1, 784)
    with torch.no_grad():
        logps = model(features)
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    prediction = probab.index(max(probab))
    print("Predicted Digit =", prediction, "  Probability = ", max(probab))
    numbers.append(prediction)

# Read puzzle skeleton with blank tile information from txt file
puzzle_file = open("Puzzle.txt", 'r')
# Save the characters in the file in a list
skeleton = list(puzzle_file.read())
puzzle_file.close()
# Loop through characters and replace number entries with the recognized numbers
digit_index = 0
for i in range(len(skeleton)):
    if skeleton[i] == '?':
        # current tile is a number
        skeleton[i] = str(numbers[digit_index])
        digit_index += 1
# Write complete puzzle data to Puzzle.txt
puzzle_file = open("Puzzle.txt", 'w')
puzzle_file.truncate()  # Delete existing contents of txt file
for i in skeleton:
    puzzle_file.write(i)
puzzle_file.close()

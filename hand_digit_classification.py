# required libraries and time module 
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

# transformations applying to the MNIST (making as a tensor and do the normalization)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# load the MNIST - training and validation (test) data
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

# data loaders for batch processing 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# load a batch (training images) and labelling
dataiter = iter(trainloader)
images, labels = next(dataiter)

# num_images, num_channel, x by x size
print(images.shape)  
print(labels.shape)  

plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

# a grid of 60 images from the batch
figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

# a simple feedforward neural network model
input_size = 784  # each image, 28x28 pixels (flattened to a 784-length vector)
hidden_sizes = [128, 64]  # 2 hidden layers with 128 and 64 neurons
output_size = 10  # 10 output classes (digits 0-9)

# sequential neural network architecture = {linear layers, ReLU activation, LogSoftmax output}
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model) 

# negative log-likelihood loss function
criterion = nn.NLLLoss()

# reshape images to 784-length vectors; calculating the amount of loss for a batch
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)
logps = model(images)
loss = criterion(logps, labels)
print('Before backward pass: \n', model[0].weight.grad)  
loss.backward()  # backward pass to calculate gradients
print('After backward pass: \n', model[0].weight.grad) 

# stochastic gradeint descent, with learning rate & momentum
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

# starting time for tracking the time
time0 = time()

# training with 15 epochs
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # flattening images for input to the network
        images = images.view(images.shape[0], -1)
        # reset gradients to zero before each training
        optimizer.zero_grad()
        # forward pass - calculate predictions and loss
        output = model(images)
        loss = criterion(output, labels)
        # backward pass - calculate gradients and update weights
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
        
print("\nTraining Time (in minutes) =", (time() - time0) / 60)

# visualize an image with predicted class probabilities
def view_classify(img, ps):
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

# testing model on a batch of validation images
images, labels = next(iter(valloader))
img = images[0].view(1, 784)
with torch.no_grad():
    logps = model(img)

ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))

view_classify(img.view(1, 28, 28), ps)

# model accuracy evaluation - with validation set
correct_count, all_count = 0, 0
for images, labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)
        
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if true_label == pred_label:
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

torch.save(model, './my_mnist_model.pt')

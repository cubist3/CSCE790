# -*- coding: utf-8 -*-
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn

import math

# Making a custom optimizer based on Adam
class CustomOptimizer(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        # Initialize with parameters passed to the parent Adam class
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        self.weight_decay = weight_decay

    def step(self, closure=None):
        # via weight decay - parameters are updated
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    # exponential MA of gradients
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # exponential MA of squared gradients
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if self.weight_decay != 0:
                    grad = grad.add(p.data, alpha=self.weight_decay)

                # decay the 1st and 2nd moment (mean and variability)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

optimizer = CustomOptimizer(model.parameters(), lr=0.003, weight_decay=0.00001)

start_time = time()
epochs = 15

for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        # custom optimizer - update
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Training loss: {running_loss / len(trainloader)}")

print("\nTotal Training Time (in minutes):", (time() - start_time) / 60)
torch.save(model, './custom_opt_mnist_model.pt')
########################################################################################################################
# This script contains an implementation of the lightning version of model1.                                           #
# Author:               Lukas Radtke, Daniel Schirmacher                                                               #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                                #
# Python Version:       3.11.7                                                                                         #
# PyTorch Version:      2.1.2                                                                                          #
# Lightning Version:    2.1.3                                                                                          #
########################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

#Load in transformed train and test set and the train and test data loader

classes =('Positve', 'Negative')

#Implement ConvNet
class ConvNet(nn.Module):
    def __innit__(self):
        super(ConvNet, self).__innit__()
        self.conv1 = nn.Conv3d(2, 4, 5) # channel size, output size, kernel size
        self.pool = nn.MaxPool3d(2, 2) # kernel size, stride
        self.conv2 = nn.Conv3d(4, 8, 5)
        self.fc1 = nn.Linear(8*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5) #flattens tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameter(), lr=learning_rate)

num_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (image_mask, labels) in enumerate(train_loader):
        image_mask = image_mask.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 = 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{num_total_steps}], Loss: {loss.item():.3f}')

print('Finished Training.')


# Calculates the accuracy
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        n_samples += labels.size(0)
        n_correct += (output == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

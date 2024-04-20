## Adapted from the Pytorch documentation
## (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from LYDIA_optimizer import LYDIA

## Dataset
DATA_PATH = './data'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

## Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()

## Initialize the LYDIA algorithm
criterion = nn.CrossEntropyLoss()
optimizer = LYDIA(model.parameters(), lr=1e-2 ,fstar=0., weight_decay=1e-6) # takes the variable to optimize, a step-size and optionally fstar

## Train
print_every = 1000
list_values_lydia = []

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step(value=loss.item())

        # print statistics
        running_loss += loss.item()
        if i % print_every == print_every-1:    # print every "print_every" mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_every:.3f}')
            list_values_lydia.append(running_loss / print_every)
            running_loss = 0.0

## Plot
fig, ax = plt.subplots(figsize=(12,6.5))

# Plot LYDIA
ax.plot(list_values_lydia, color='dodgerblue', lw=3, label='LYDIA Train loss')

ax.set_ylabel(r'Loss', fontsize=16)
ax.set_xlabel(f'number of iterations (x {print_every})', fontsize=16)
ax.legend(fontsize=16)

fig.show()

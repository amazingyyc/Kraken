# coding=utf-8

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import kraken.pytorch as kk


class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output


def main():
  # connect to training cluster.
  kk.initialize('127.0.0.1:50000')

  # Training settings
  batch_size = 64
  epochs = 14
  lr = 0.01
  seed = 1
  log_interval = 5

  torch.manual_seed(seed)

  device = torch.device("cpu")

  train_kwargs = {'batch_size': batch_size}

  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
  dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
  dataset2 = datasets.MNIST('./data', train=False, transform=transform)
  train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)

  model = Net().to(device)
  # optimizer = optim.Adadelta(model.parameters(), lr=lr)
  optimizer = kk.Optimizer('Mnist', named_parameters=model.named_parameters(), lr=lr, optim=kk.SGD())

  model.train()
  for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      output = model(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()

      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                       len(train_loader.dataset),
                                                                       100. * batch_idx / len(train_loader),
                                                                       loss.item()))


if __name__ == '__main__':
  main()

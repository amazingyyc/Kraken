import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import kraken.pytorch as kk

from model.DeepFM import DeepFM
from data.dataset import CriteoDataset

# load data
train_data = CriteoDataset('./data', train=True)

loader_train = DataLoader(train_data, batch_size=100,
                          sampler=sampler.SubsetRandomSampler(range(800)))
val_data = CriteoDataset('./data', train=True)
loader_val = DataLoader(val_data, batch_size=100,
                        sampler=sampler.SubsetRandomSampler(range(800, 899)))

feature_sizes = np.loadtxt('./data/feature_sizes.txt', delimiter=',')
feature_sizes = [int(x) for x in feature_sizes]
print(feature_sizes)

# initialize Kraken
kk.initialize('127.0.0.1:50000')

model = DeepFM(feature_sizes, use_cuda=False)
# optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
optimizer = kk.Optimizer(model_name='DeepFM', named_parameters=model.named_parameters(), optim=kk.Adam(), lr=1e-3)

model.fit(loader_train, loader_val, optimizer, epochs=1000, verbose=True)

# Stop worker.
kk.stop()

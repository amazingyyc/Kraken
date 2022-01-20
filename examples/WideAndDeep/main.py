# coding=utf-8

import time
import numpy as np
import pandas as pd
import data_utils
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from model import WideDeep, WideDeepLoader
import kraken.pytorch as kk

# initialize Kraken
kk.initialize(addrs='localhost:50000')

path = 'data/adult_data.csv'

# Read the data
DF = pd.read_csv(path)

# target for logistic regression
DF['income_label'] = (DF["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

# Experiment set up
# WIDE PART
wide_cols = [
    'age', 'hours_per_week', 'education', 'relationship', 'workclass', 'occupation', 'native_country', 'gender'
]
crossed_cols = (['education', 'occupation'], ['native_country', 'occupation'])

# DEEP PART
# columns that will be passed as embeddings and their corresponding number of embeddings (optional, see demo3).
embeddings_cols = [('education', 10), ('relationship', 8), ('workclass', 10), ('occupation', 10),
                   ('native_country', 10)]
continuous_cols = ["age", "hours_per_week"]

# TARGET AND METHOD
target = 'income_label'
method = 'logistic'

wd_dataset = data_utils.prepare_data(DF, wide_cols, crossed_cols, embeddings_cols, continuous_cols, target)

wide_dim = wd_dataset['train_dataset'].wide.shape[1]
n_unique = len(np.unique(wd_dataset['train_dataset'].labels))
deep_column_idx = wd_dataset['deep_column_idx']
embeddings_input = wd_dataset['embeddings_input']
encoding_dict = wd_dataset['encoding_dict']
hidden_layers = [100, 50]
dropout = [0.5, 0.2]
n_class = 1

model = WideDeep(wide_dim, embeddings_input, continuous_cols, deep_column_idx, hidden_layers, dropout, encoding_dict,
                 n_class)

optimizer = kk.Optimizer(model_name='Wide_Deep', named_parameters=model.named_parameters(), optim=kk.Adam(), lr=0.001)

criterion = F.binary_cross_entropy

train_dataset = wd_dataset['train_dataset']

n_epochs = 100
batch_size = 64

widedeep_dataset = WideDeepLoader(train_dataset)
train_loader = torch.utils.data.DataLoader(dataset=widedeep_dataset, batch_size=batch_size, shuffle=True)
model.train()

for epoch in range(n_epochs):
  total = 0
  correct = 0

  for i, (X_wide, X_deep, target) in enumerate(train_loader):
    start_t = time.clock()

    X_w = Variable(X_wide)
    X_d = Variable(X_deep)
    y = Variable(target).float().unsqueeze(-1)

    optimizer.zero_grad()
    y_pred = model(X_w, X_d)

    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    y_pred_cat = (y_pred > 0.5).int().flatten()
    y_cat = y.int().flatten()

    count = y.size(0)
    cur_correct = (y_pred_cat == y_cat).sum().item()

    total += count
    correct += cur_correct

    end_t = time.clock()

    print('Cost {:0,.2f} Epoch {} of {}, Loss: {}, accuracy: {}'.format((end_t - start_t) * 1000, epoch + 1, n_epochs,
                                                                        round(loss.item(), 3),
                                                                        round(correct / total, 4)))

# Stop it
kk.stop()

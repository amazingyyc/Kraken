# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class WideDeepLoader(Dataset):
  """Helper to facilitate loading the data to the pytorch models.

    Parameters:
    --------
    data: namedtuple with 3 elements - (wide_input_data, deep_inp_data, target)
    """

  def __init__(self, data):

    self.X_wide = data.wide
    self.X_deep = data.deep
    self.Y = data.labels

  def __getitem__(self, idx):

    xw = self.X_wide[idx]
    xd = self.X_deep[idx]
    y = self.Y[idx]

    return xw, xd, y

  def __len__(self):
    return len(self.Y)


class WideDeep(nn.Module):
  """ Wide and Deep model. As explained in Heng-Tze Cheng et al., 2016, the
    model taked the wide features and the deep features after being passed through
    the hidden layers and connects them to an output neuron. For details, please
    refer to the paper and the corresponding tutorial in the tensorflow site:
    https://www.tensorflow.org/tutorials/wide_and_deep

    Parameters:
    --------
    wide_dim (int) : dim of the wide-side input tensor
    embeddings_input (tuple): 3-elements tuple with the embeddings "set-up" -
    (col_name, unique_values, embeddings dim)
    continuous_cols (list) : list with the name of the continuum columns
    deep_column_idx (dict) : dictionary where the keys are column names and the values
    their corresponding index in the deep-side input tensor
    hidden_layers (list) : list with the number of units per hidden layer
    encoding_dict (dict) : dictionary with the label-encode mapping
    n_class (int) : number of classes. Defaults to 1 if logistic or regression
    dropout (float)
    """

  def __init__(self, wide_dim, embeddings_input, continuous_cols, deep_column_idx, hidden_layers, dropout,
               encoding_dict, n_class):

    super(WideDeep, self).__init__()
    self.wide_dim = wide_dim
    self.deep_column_idx = deep_column_idx
    self.embeddings_input = embeddings_input
    self.continuous_cols = continuous_cols
    self.hidden_layers = hidden_layers
    self.dropout = dropout
    self.encoding_dict = encoding_dict
    self.n_class = n_class

    # Build the embedding layers to be passed through the deep-side
    for col, val, dim in self.embeddings_input:
      setattr(self, 'emb_layer_' + col, nn.Embedding(val, dim))
      # setattr(self, 'emb_layer_' + col, kk.Embedding(dim))

    # Build the deep-side hidden layers with dropout if specified
    input_emb_dim = np.sum([emb[2] for emb in self.embeddings_input])
    self.linear_1 = nn.Linear(input_emb_dim + len(continuous_cols), self.hidden_layers[0])
    if self.dropout:
      self.linear_1_drop = nn.Dropout(self.dropout[0])
    for i, h in enumerate(self.hidden_layers[1:], 1):
      setattr(self, 'linear_' + str(i + 1), nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))
      if self.dropout:
        setattr(self, 'linear_' + str(i + 1) + '_drop', nn.Dropout(self.dropout[i]))

    # Connect the wide- and dee-side of the model to the output neuron(s)
    self.output = nn.Linear(self.hidden_layers[-1] + self.wide_dim, self.n_class)

  def forward(self, X_w, X_d):
    """Implementation of the forward pass.

        Parameters:
        ----------
        X_w (torch.tensor) : wide-side input tensor
        X_d (torch.tensor) : deep-side input tensor

        Returns:
        --------
        out (torch.tensor) : result of the output neuron(s)
        """
    # Deep Side
    emb = [
        getattr(self, 'emb_layer_' + col)(X_d[:, self.deep_column_idx[col]].long())
        for col, _, _ in self.embeddings_input
    ]
    if self.continuous_cols:
      cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
      cont = [X_d[:, cont_idx].float()]
      deep_inp = torch.cat(emb + cont, 1)
    else:
      deep_inp = torch.cat(emb, 1)

    x_deep = F.relu(self.linear_1(deep_inp))
    if self.dropout:
      x_deep = self.linear_1_drop(x_deep)
    for i in range(1, len(self.hidden_layers)):
      x_deep = F.relu(getattr(self, 'linear_' + str(i + 1))(x_deep))
      if self.dropout:
        x_deep = getattr(self, 'linear_' + str(i + 1) + '_drop')(x_deep)

    # Deep + Wide sides
    wide_deep_input = torch.cat([x_deep, X_w.float()], 1)

    return F.sigmoid(self.output(wide_deep_input))

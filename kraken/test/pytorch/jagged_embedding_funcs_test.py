# coding=utf-8

import unittest
import torch
import numpy as np
from kraken.pytorch.jagged_tensor import JaggedTensor
from kraken.pytorch.jagged_embedding_funcs import JaggedEmbeddingSumFunction, JaggedEmbeddingMeanFunction


class JaggedEmbeddingFuncsTest(unittest.TestCase):

  def test_sum(self):
    lengths = (np.random.rand(5) * 10).astype(int) + 1
    values = []

    for l in lengths:
      values.append(torch.rand(l, 20, 30))

    offsets = [0]
    for l in lengths:
      offsets.append(offsets[-1] + l)

    e_values = []
    for v in values:
      v = torch.sum(v, dim=0, keepdim=True)
      e_values.append(v)

    e_value = torch.cat(e_values, dim=0)

    j_tensor = JaggedTensor.from_list_tensors(values)
    r_value = JaggedEmbeddingSumFunction.apply(j_tensor.values(), j_tensor.offsets(), 0)

    torch.testing.assert_close(e_value, r_value)

  def test_mean(self):
    lengths = (np.random.rand(5) * 10).astype(int) + 1
    values = []

    for l in lengths:
      values.append(torch.rand(l, 20, 30))

    offsets = [0]
    for l in lengths:
      offsets.append(offsets[-1] + l)

    e_values = []
    for v in values:
      v = torch.mean(v, dim=0, keepdim=True)
      e_values.append(v)

    e_value = torch.cat(e_values, dim=0)

    j_tensor = JaggedTensor.from_list_tensors(values)
    r_value = JaggedEmbeddingMeanFunction.apply(j_tensor.values(), j_tensor.offsets(), 0)

    torch.testing.assert_close(e_value, r_value)

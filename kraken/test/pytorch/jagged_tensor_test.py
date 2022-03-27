# coding=utf-8

import unittest
import torch
import numpy as np
from kraken.pytorch.jagged_tensor import JaggedTensor

class JaggedTensorTest(unittest.TestCase):
  def test_normal(self):
    lengths = (np.random.rand(5) * 10).astype(int) + 1
    values = []

    for l in lengths:
      values.append(torch.rand(l))
    
    offsets = [0]
    for l in lengths:
      offsets.append(offsets[-1] + l)

    e_values = torch.cat(values, 0)
    e_offsets = torch.from_numpy(np.array(offsets))
    
    j_tensor = JaggedTensor.from_list_tensors(values)

    torch.testing.assert_close(e_values, j_tensor.values())
    torch.testing.assert_close(e_offsets, j_tensor.offsets())

  def test_multi_dimension(self):
    lengths = (np.random.rand(5) * 10).astype(int) + 1
    values = []

    for l in lengths:
      values.append(torch.rand(l, 2, 3))
    
    offsets = [0]
    for l in lengths:
      offsets.append(offsets[-1] + l)

    e_values = torch.cat(values, 0)
    e_offsets = torch.from_numpy(np.array(offsets))
    
    j_tensor = JaggedTensor.from_list_tensors(values)

    torch.testing.assert_close(e_values, j_tensor.values())
    torch.testing.assert_close(e_offsets, j_tensor.offsets())
# coding=utf-8

import logging
import torch
from kraken.pytorch.initializer import Initializer, NormalInitializer
import kraken_native


class SparseTable(torch.nn.Parameter):

  def __new__(cls,
              dimension: int,
              dtype: torch.dtype = torch.float32,
              initializer: Initializer = NormalInitializer(),
              name: str = None):
    self = super(SparseTable, cls).__new__(cls)
    self._dimension = dimension
    self._dtype = dtype
    self._table_id = None
    self._initializer = initializer
    self._name = name

    return self

  def dimension(self):
    return self._dimension

  def dtype(self):
    return self._dtype

  def table_id(self):
    return self._table_id

  def initializer(self):
    return self._initializer

  def name(self):
    return self._name

  def set_table_id(self, table_id: int):
    self._table_id = table_id

  def set_name(self, name: str):
    self._name = name


class EmbeddingFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, table_id, indices):
    ctx.save_for_backward(indices)
    ctx.table_id = table_id

    return kraken_native.pull_sparse_table(table_id, indices)

  @staticmethod
  def backward(ctx, grad):
    indices, = ctx.saved_tensors
    table_id = ctx.table_id

    kraken_native.push_sparse_table(table_id, indices, grad)

    return None, None


class Embedding(torch.nn.Module):

  def __init__(self,
               dimension: int,
               dtype: torch.dtype = torch.float32,
               initializer: Initializer = NormalInitializer(),
               name: str = None):
    super(Embedding, self).__init__()
    # Ok at here we just create a SparseTabel instance.
    # This instance include dimension/dtype/name.
    # We will actually register the SparseTable in server when user call optimizer.
    self.sparse_table = SparseTable(dimension=dimension, dtype=dtype, initializer=initializer, name=name)

  def forward(self, indices):
    return EmbeddingFunction.apply(self.sparse_table.table_id(), indices)

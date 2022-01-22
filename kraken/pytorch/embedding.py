# coding=utf-8

import logging
from math import fabs
from typing import List
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
  def forward(ctx, sparse_table, indices):
    ctx.save_for_backward(sparse_table, indices)
    return kraken_native.pull_sparse_table(sparse_table.table_id(), indices)

  @staticmethod
  def backward(ctx, grad):
    sparse_table, indices, = ctx.saved_tensors

    kraken_native.push_sparse_table(sparse_table.table_id(), indices, grad)

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
    return EmbeddingFunction.apply(self.sparse_table, indices)


# class CombineEmbedding(torch.nn.Module):

#   def __init__(self,
#                dimensions: List[int],
#                dtypes: List[torch.dtype] = None,
#                initializers: List[Initializer] = None,
#                names: List[str] = None):
#     super(CombineEmbedding, self).__init__()

#     if dtypes:
#       assert len(dimensions) == len(dtypes)

#     if initializers:
#       assert len(dimensions) == len(initializers)

#     if names:
#       assert len(dimensions) == len(names)

#     self._dimensions = dimensions
#     self._dtypes = dtypes
#     self._initializers = initializers
#     self._names = names

#     if self._dtypes is None:
#       self._dtypes = [torch.float32] * len(self._dimensions)

#     if self._initializers is None:
#       self._initializers = [NormalInitializer()] * len(self._dimensions)

#     if self._names is None:
#       self._names = [None] * len(self._dimensions)

#     for i in range(len(self._dimensions)):
#       setattr(
#           self, 'sparse_table_' + str(i),
#           SparseTable(dimension=self._dimensions[i],
#                       dtype=self._dtypes[i],
#                       initializer=self._initializers[i],
#                       name=self._names[i]))

#   def forward(self, indices):
#     pass

# coding=utf-8

from typing import List
import torch
from sparse_table import SparseTable
from kraken.pytorch.initializer import Initializer, NormalInitializer
import kraken_native


class CombineEmbeddingFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, sparse_table, indices, val):
    ctx.save_for_backward(sparse_table, indices)
    return val

  @staticmethod
  def backward(ctx, grad):
    sparse_table, indices, = ctx.saved_tensors

    kraken_native.push_sparse_table(sparse_table.table_id(), indices, grad)

    return None, None


class CombineEmbedding(torch.nn.Module):

  def __init__(self,
               dimensions: List[int],
               dtypes: List[torch.dtype] = None,
               initializers: List[Initializer] = None,
               names: List[str] = None):
    super(CombineEmbedding, self).__init__()

    if dtypes:
      assert len(dimensions) == len(dtypes)

    if initializers:
      assert len(dimensions) == len(initializers)

    if names:
      assert len(dimensions) == len(names)

    self._dimensions = dimensions
    self._dtypes = dtypes
    self._initializers = initializers
    self._names = names

    if self._dtypes is None:
      self._dtypes = [torch.float32] * len(self._dimensions)

    if self._initializers is None:
      self._initializers = [NormalInitializer()] * len(self._dimensions)

    if self._names is None:
      self._names = [None] * len(self._dimensions)

    for i in range(len(self._dimensions)):
      setattr(
          self, 'sparse_table_' + str(i),
          SparseTable(dimension=self._dimensions[i],
                      dtype=self._dtypes[i],
                      initializer=self._initializers[i],
                      name=self._names[i]))

    self._sparse_tables = [getattr(self, 'sparse_table_' + str(i)) for i in range(len(self._dimensions))]

  def forward(self, indices: List):
    """This is maybe weired, We get the embedding by call combine_pull_dense_table directly not invoke a special function like Embedding.
    Because I can not find a right way that torch.autograd.Function can accept a list inputs and return a list output with the gradient backward right."""
    assert len(self._sparse_tables) == len(indices)

    table_ids = [t.table_id() for t in self._sparse_tables]

    vals = kraken_native.combine_pull_sparse_table(table_ids, indices)

    assert len(self._sparse_tables) == len(vals)
    """call CombineEmbeddingFunction make sure the gradient canbe backward right."""
    return [
        CombineEmbeddingFunction.apply(self._sparse_tables[i], indices[i], vals[i])
        for i in range(len(self._sparse_tables))
    ]

# coding=utf-8

import torch
from kraken.pytorch.sparse_table import SparseTable
from kraken.pytorch.initializer import Initializer, NormalInitializer
import kraken_native


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
    '''At here we just create a SparseTabel instance.
    This instance include dimension/dtype/name.
    We will actually register the SparseTable in server when user call optimizer.'''
    self.sparse_table = SparseTable(dimension=dimension,
                                    dtype=dtype,
                                    initializer=initializer,
                                    name=name)

  def forward(self, indices):
    return EmbeddingFunction.apply(self.sparse_table, indices)

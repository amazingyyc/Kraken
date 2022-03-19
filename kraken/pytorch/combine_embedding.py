# coding=utf-8

from pickletools import read_unicodestring1
from typing import List, Union
import torch
from kraken.pytorch.combine_sparse_table import CombineSparseTable
from kraken.pytorch.initializer import Initializer
import kraken_native


class CombineEmbeddingFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, combine_sparse_table, *indices):
    ctx.save_for_backward(combine_sparse_table, *indices)

    assert len(combine_sparse_table.table_ids()) == len(indices)

    vals = kraken_native.combine_pull_sparse_table(combine_sparse_table.table_ids(), list(indices))

    return tuple(vals)

  @staticmethod
  def backward(ctx, *grads):
    combine_sparse_table, *indices, = ctx.saved_tensors

    kraken_native.combine_push_sparse_table(combine_sparse_table.table_ids(), list(indices), list(grads))

    return tuple([None] * (1 + len(grads)))


class CombineEmbedding(torch.nn.Module):

  def __init__(self,
               dimensions: List[int],
               dtypes: List[torch.dtype] = None,
               initializers: List[Initializer] = None,
               names: List[str] = None):
    super(CombineEmbedding, self).__init__()

    self.combine_sparse_table = CombineSparseTable(dimensions=dimensions,
                                                   dtypes=dtypes,
                                                   initializers=initializers,
                                                   names=names)

  def forward(self, indices: list):
    vals = CombineEmbeddingFunction.apply(self.combine_sparse_table, *indices)
    return list(vals)

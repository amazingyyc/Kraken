# coding=utf-8

from typing import List
import torch
from kraken.pytorch.initializer import Initializer, NormalInitializer


class CombineSparseTable(torch.nn.Parameter):

  def __new__(cls,
              dimensions: List[int],
              dtypes: List[torch.dtype] = None,
              initializers: List[Initializer] = None,
              names: List[str] = None):
    self = super(CombineSparseTable, cls).__new__(cls)

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
    self._table_ids = None

    if self._dtypes is None:
      self._dtypes = [torch.float32] * len(self._dimensions)

    if self._initializers is None:
      self._initializers = [NormalInitializer()] * len(self._dimensions)

    return self

  def dimensions(self):
    return self._dimensions

  def dtypes(self):
    return self._dtypes

  def initializers(self):
    return self._initializers

  def names(self):
    return self._names

  def table_ids(self):
    return self._table_ids

  def set_names(self, names: List[str]):
    self._names = names

  def set_table_ids(self, table_ids: List[int]):
    self._table_ids = table_ids

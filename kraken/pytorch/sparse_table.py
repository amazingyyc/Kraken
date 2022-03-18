# coding=utf-8

import torch
from kraken.pytorch.initializer import Initializer, NormalInitializer


class SparseTable(torch.nn.Parameter):

  def __new__(cls,
              dimension: int,
              dtype: torch.dtype = torch.float32,
              initializer: Initializer = NormalInitializer(),
              name: str = None):
    self = super(SparseTable, cls).__new__(cls)
    self._dimension = dimension
    self._dtype = dtype
    self._initializer = initializer
    self._name = name
    self._table_id = None

    return self

  def dimension(self):
    return self._dimension

  def dtype(self):
    return self._dtype

  def initializer(self):
    return self._initializer

  def name(self):
    return self._name

  def table_id(self):
    return self._table_id

  def set_name(self, name: str):
    self._name = name

  def set_table_id(self, table_id: int):
    self._table_id = table_id

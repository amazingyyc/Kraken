# coding=utf-8

from typing import Dict, Union
import logging
from kraken.pytorch.embedding import SparseTable
from kraken.pytorch.lr import LR, ConstantLR
from kraken.pytorch.optim import Optim
import kraken_native


class Optimizer:

  def __init__(self, model_name: str, named_parameters, lr: Union[float, LR], optim: Optim):

    self._model_name = model_name

    if isinstance(lr, float):
      self._lr = ConstantLR(lr=lr)
    else:
      self._lr = lr

    self._optim_type = optim.type()
    self._optim_conf = optim.conf()
    self._name_parma = {}
    self._name_table_id = {}

    self._model_id = kraken_native.register_model(self._model_name, self._optim_type, self._optim_conf)
    logging.info(f'Register model:[{self._model_name}], model_id:[{self._model_id}].')

    # We should update learning rate when initialize model.
    kraken_native.update_lr(self._lr.lr())

    for name, param in named_parameters:
      if param.requires_grad:
        self._name_parma[name] = param

        if isinstance(param, SparseTable):
          # Check whether the user has set a name.
          real_name = name
          if param.name() is not None:
            real_name = param.name()

          table_id = kraken_native.register_sparse_table(real_name, param.dimension(), param.dtype())
          param.set_table_id(table_id)
          param.set_name(real_name)

          self._name_table_id[name] = table_id

          logging.info(
              f'Register SparseTable:[{name}], ' \
              f'table id:[{table_id}], ' \
              f'dimension:[{param.dimension()}], ' \
              f'dtype:[{param.dtype()}].'
          )
        else:
          table_id = kraken_native.register_dense_table(name, param.data)
          self._name_table_id[name] = table_id

          logging.info(f'Register DenseTable:[{name}], table id:[{table_id}].')

          # Register a hook func to send gradient to server.
          param.register_hook(self._create_dense_grad_hook(name, table_id))

  def _create_dense_grad_hook(self, name, table_id):
    logging.info(f'Create gradient hook for:[{name}], table_id:[{table_id}].')

    def hook(grad):
      # This is a little tricky, the returned gradient actually is the Tensor value.
      # (TODO) maybe send gradient but not wait receive.
      return kraken_native.push_pull_dense_table(table_id, grad)

    return hook

  def step(self):
    # For now we already pull the DenseTable from server and assign to the param's grad,
    # So just give it to param's data.
    for _, param in self._name_parma.items():
      if not isinstance(param, SparseTable):
        param.data = param.grad
        param.grad = None

    # Update learning rate.
    self._lr.step()
    kraken_native.update_lr(self._lr.lr())

  def zero_grad(self):
    pass

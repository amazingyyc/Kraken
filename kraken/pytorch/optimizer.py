# coding=utf-8

from typing import Dict, Union
import logging
from kraken.pytorch.embedding import SparseTable
from kraken.pytorch.lr import LR, ConstantLR
from kraken.pytorch.optim import Optim
import kraken_native


class Optimizer:

  def __init__(self, model_name: str, named_parameters, lr: Union[float, LR], optim: Optim, dense_async=True):

    self._model_name = model_name

    if isinstance(lr, float):
      self._lr = ConstantLR(lr=lr)
    else:
      self._lr = lr

    self._optim = optim
    self._name_param = {}
    self._name_table_id = {}

    # Whether async send Dense gradient to server.
    self._dense_async = dense_async
    logging.info(f'dense_async is:{self._dense_async}')

    self._model_id = kraken_native.register_model(self._model_name, self._optim.type(), self._optim.conf())
    logging.info(f'Register model:[{self._model_name}], model_id:[{self._model_id}].')

    # We should update learning rate when initialize model.
    kraken_native.update_lr(self._lr.lr())

    for name, param in named_parameters:
      if param.requires_grad:
        self._name_param[name] = param

        if isinstance(param, SparseTable):
          # Check whether the user has set a name.
          real_name = name
          if param.name() is not None:
            real_name = param.name()

          dimension = param.dimension()
          dtype = param.dtype()
          initializer = param.initializer()

          table_id = kraken_native.register_sparse_table(name=real_name,
                                                         dimension=dimension,
                                                         dtype=dtype,
                                                         init_type=initializer.type(),
                                                         init_conf=initializer.conf())

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
      # At here we will check the flag: dense_async, it True we will send gradient to server async and not wait.
      # If False we will send Parameter's Gradient to server and wait fetch the Val.
      if self._dense_async:
        kraken_native.push_dense_table(table_id, grad)
        return None
      else:
        return kraken_native.push_pull_dense_table(table_id, grad)

    return hook

  def step(self):
    for name, param in self._name_param.items():
      if not isinstance(param, SparseTable):
        if self._dense_async:
          param.data = kraken_native.pull_dense_table(self._name_table_id[name])
        else:
          # dense_async is False means the Prameter's value has been assign to the grad.
          param.data = param.grad

      param.grad = None

    # Update learning rate.
    self._lr.step()
    kraken_native.update_lr(self._lr.lr())

  def zero_grad(self):
    pass

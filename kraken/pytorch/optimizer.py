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

    self._optim = optim
    self._name_param = {}
    self._name_table_id = {}

    kraken_native.init_model(self._model_name, self._optim.type(), self._optim.conf())
    logging.info(
        f'Register model:[{self._model_name}], optim_type:{self._optim.type()}, optim_conf:{self._optim.conf()}.')

    # We should update learning rate when initialize model.
    kraken_native.update_lr(self._lr.lr())

    for name, param in named_parameters:
      if param.requires_grad:
        self._name_param[name] = param

        if isinstance(param, SparseTable):
          raise ValueError('Not support SparseTable for now.')
          # # Check whether the user has set a name.
          # real_name = name
          # if param.name() is not None:
          #   real_name = param.name()

          # dimension = param.dimension()
          # dtype = param.dtype()
          # initializer = param.initializer()

          # table_id = kraken_native.register_sparse_table(name=real_name,
          #                                                dimension=dimension,
          #                                                dtype=dtype,
          #                                                init_type=initializer.type(),
          #                                                init_conf=initializer.conf())

          # param.set_table_id(table_id)
          # param.set_name(real_name)

          # self._name_table_id[name] = table_id

          # logging.info(
          #     f'Register SparseTable:[{name}], ' \
          #     f'table id:[{table_id}], ' \
          #     f'dimension:[{param.dimension()}], ' \
          #     f'dtype:[{param.dtype()}].'
          # )
        else:
          table_id = kraken_native.register_dense_table(name, param.data)
          self._name_table_id[name] = table_id

          # Register a hook func to send gradient to server.
          param.register_hook(self._create_dense_grad_hook(name, table_id))

          logging.info(f'Register DenseTable:[{name}], table id:[{table_id}].')

  def _create_dense_grad_hook(self, name, table_id):
    logging.info(f'Create gradient hook for:[{name}], table_id:[{table_id}].')

    def hook(grad):
      # We will send gradient to server async.
      kraken_native.push_dense_table(table_id, grad)
      return grad

    return hook

  def step(self):
    dense_params = []
    dense_table_ids = []

    for name, param in self._name_param.items():
      if not isinstance(param, SparseTable):
        dense_params.append(param)
        dense_table_ids.append(self._name_table_id[name])

      param.grad = None

    dense_table_datas = kraken_native.combine_pull_dense_table(dense_table_ids)

    assert len(dense_params) == len(dense_table_datas)

    # update parameter's data.
    for i in range(len(dense_params)):
      dense_params[i].data = dense_table_datas[i]

    # Update learning rate.
    self._lr.step()
    kraken_native.update_lr(self._lr.lr())

  def zero_grad(self):
    pass

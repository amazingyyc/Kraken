# coding=utf-8

from typing import Union
import logging
from kraken.pytorch.sparse_table import SparseTable
from kraken.pytorch.combine_sparse_table import CombineSparseTable
from kraken.pytorch.lr import LR, ConstantLR
from kraken.pytorch.optim import Optim
import kraken_native

logging.getLogger().setLevel(logging.INFO)


class Optimizer:

  def __init__(self, model_name: str, named_parameters, lr: Union[float, LR],
               optim: Optim):

    self._model_name = model_name

    if isinstance(lr, float):
      self._lr = ConstantLR(lr=lr)
    else:
      self._lr = lr

    self._optim = optim

    # <param, table_id/table_ids>
    self._param_table_id = {}

    kraken_native.init_model(self._model_name, self._optim.type(),
                             self._optim.conf())

    logging.info(
        f'Register model:[{self._model_name}], optim_type:{self._optim.type()}, optim_conf:{self._optim.conf()}.'
    )

    # We should update learning rate when initialize model.
    kraken_native.update_lr(self._lr.lr())

    for name, param in named_parameters:
      if not param.requires_grad:
        continue

      if isinstance(param, SparseTable):
        # Check whether the user has set a name.
        real_name = name
        if param.name() is not None:
          real_name = param.name()

        dimension = param.dimension()
        dtype = param.dtype()
        initializer = param.initializer()

        table_id = kraken_native.register_sparse_table(
            name=real_name,
            dimension=dimension,
            dtype=dtype,
            init_type=initializer.type(),
            init_conf=initializer.conf())

        param.set_table_id(table_id)
        param.set_name(real_name)

        self._param_table_id[param] = table_id

        logging.info(
            f'Register SparseTable:[{real_name}], ' \
            f'table id:[{table_id}], ' \
            f'dimension:[{dimension}], ' \
            f'dtype:[{dtype}], ' \
            f'init_type:[{initializer.type()}], ' \
            f'init_conf:[{initializer.conf()}]'
        )
      elif isinstance(param, CombineSparseTable):
        real_names = []
        table_ids = []

        for i in range(len(param.dimensions())):
          real_name = name + '.' + str(i)

          if param.names() is not None:
            real_name = param.names()[i]

          real_names.append(real_name)

          dimension = param.dimensions()[i]
          dtype = param.dtypes()[i]
          initializer = param.initializers()[i]

          table_id = kraken_native.register_sparse_table(
              name=real_name,
              dimension=dimension,
              dtype=dtype,
              init_type=initializer.type(),
              init_conf=initializer.conf())

          table_ids.append(table_id)

          logging.info(
            f'Register SparseTable:[{real_name}], ' \
            f'table id:[{table_id}], ' \
            f'dimension:[{dimension}], ' \
            f'dtype:[{dtype}], ' \
            f'init_type:[{initializer.type()}], ' \
            f'init_conf:[{initializer.conf()}]'
          )

        param.set_names(real_names)
        param.set_table_ids(table_ids)

        # CombineSparseTable donot has a table id, so we set the table_ids list.
        self._param_table_id[param] = table_ids
      else:
        table_id = kraken_native.register_dense_table(name, param.data)
        self._param_table_id[param] = table_id

        # Register a hook func to send gradient to server.
        param.register_hook(self._create_dense_grad_hook(name, table_id))

        logging.info(f'Register DenseTable:[{name}], table id:[{table_id}].')

    # When finish register SpareTable/DenseTable in Ps we need pull DenseTable at beginning.
    # make sure the the Model is same as stored in Ps.
    dense_params = []
    dense_table_ids = []

    for param, table_id in self._param_table_id.items():
      if not isinstance(param, SparseTable) and not isinstance(
          param, CombineSparseTable):
        dense_params.append(param)
        dense_table_ids.append(table_id)

    dense_table_datas = kraken_native.combine_pull_dense_table(dense_table_ids)

    assert len(dense_params) == len(dense_table_datas)

    for i in range(len(dense_params)):
      dense_params[i].data = dense_table_datas[i]

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

    for param, table_id in self._param_table_id.items():
      if not isinstance(param, SparseTable) and not isinstance(
          param, CombineSparseTable):
        dense_params.append(param)
        dense_table_ids.append(table_id)

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

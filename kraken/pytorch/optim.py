# coding=utf-8

from typing import Dict
import kraken_native


class Optim(object):

  def type(self) -> kraken_native.OptimType:
    raise NotImplementedError

  def conf(self) -> Dict[str, str]:
    raise NotImplementedError


class Adagrad(Optim):

  def __init__(self, weight_decay=None, eps: float = 1e-10):
    super(Adagrad, self).__init__()

    self._weight_decay = weight_decay
    self._eps = eps

  def type(self) -> kraken_native.OptimType:
    return kraken_native.OptimType.kAdagrad

  def conf(self) -> Dict[str, str]:
    conf = dict()

    if self._weight_decay:
      conf['weight_decay'] = str(self._weight_decay)

    conf['eps'] = str(self._eps)

    return conf


class Adam(Optim):

  def __init__(self,
               weight_decay=None,
               beta1: float = 0.9,
               beta2: float = 0.999,
               eps: float = 1e-08,
               amsgrad: bool = False):
    super(Adam, self).__init__()

    self._beta1 = beta1
    self._beta2 = beta2
    self._eps = eps
    self._weight_decay = weight_decay
    self._amsgrad = amsgrad

  def type(self) -> kraken_native.OptimType:
    return kraken_native.OptimType.kAdam

  def conf(self) -> Dict[str, str]:
    conf = dict()
    if self._weight_decay:
      conf['weight_decay'] = str(self._weight_decay)

    conf['beta1'] = str(self._beta1)
    conf['beta2'] = str(self._beta2)
    conf['eps'] = str(self._eps)

    if self._amsgrad:
      conf['amsgrad'] = 'true'
    else:
      conf['amsgrad'] = 'false'

    return conf


class RMSprop(Optim):

  def __init__(self, weight_decay=None, momentum=None, alpha: float = 0.99, eps: float = 1e-8, centered: bool = False):
    super(RMSprop, self).__init__()

    self._weight_decay = weight_decay
    self._momentum = momentum
    self._alpha = alpha
    self._eps = eps
    self._centered = centered

  def type(self) -> kraken_native.OptimType:
    return kraken_native.OptimType.kRMSprop

  def conf(self) -> Dict[str, str]:
    conf = dict()

    if self._weight_decay:
      conf['weight_decay'] = str(self._weight_decay)
    if self._momentum:
      conf['momentum'] = str(self._momentum)

    conf['alpha'] = str(self._alpha)
    conf['eps'] = str(self._eps)

    if self._centered:
      conf['centered'] = 'true'
    else:
      conf['centered'] = 'false'

    return conf


class SGD(Optim):

  def __init__(self, momentum=None, dampening=None, weight_decay=None, nesterov: bool = False):
    super(SGD, self).__init__()

    self._momentum = momentum
    self._dampening = dampening
    self._weight_decay = weight_decay
    self._nesterov = nesterov

  def type(self) -> kraken_native.OptimType:
    return kraken_native.OptimType.kSGD

  def conf(self) -> Dict[str, str]:
    conf = dict()

    if self._momentum:
      conf['momentum'] = str(self._momentum)
    if self._dampening:
      conf['dampening'] = str(self._dampening)
    if self._weight_decay:
      conf['weight_decay'] = str(self._weight_decay)
    if self._nesterov:
      conf['nesterov'] = 'true'
    else:
      conf['nesterov'] = 'false'

    return conf

# coding=utf-8

from typing import Dict
import kraken_native


class Initializer(object):

  def type(self) -> kraken_native.InitializerType:
    raise NotImplementedError

  def conf(self) -> Dict[str, str]:
    raise NotImplementedError


class ConstantInitializer(Initializer):

  def __init__(self, value: float = 0.0):
    super(ConstantInitializer, self).__init__()
    self._value = value

  def type(self) -> kraken_native.InitializerType:
    return kraken_native.InitializerType.kConstant

  def conf(self) -> Dict[str, str]:
    return {'value': str(self._value)}


class UniformInitializer(Initializer):

  def __init__(self, lower: float = 0.0, upper: float = 1.0):
    super(UniformInitializer, self).__init__()
    self._lower = lower
    self._upper = upper

  def type(self) -> kraken_native.InitializerType:
    return kraken_native.InitializerType.kUniform

  def conf(self) -> Dict[str, str]:
    return {'lower': str(self._lower), 'upper': str(self._upper)}


class NormalInitializer(Initializer):

  def __init__(self, mean: float = 0.0, stddev: float = 1.0):
    super(NormalInitializer, self).__init__()
    self._mean = mean
    self._stddev = stddev

  def type(self) -> kraken_native.InitializerType:
    return kraken_native.InitializerType.kNormal

  def conf(self) -> Dict[str, str]:
    return {'mean': str(self._mean), 'stddev': str(self._stddev)}


class XavierUniformInitializer(Initializer):

  def __init__(self, gain: float = 1.0):
    self._gain = gain

  def type(self) -> kraken_native.InitializerType:
    return kraken_native.InitializerType.kXavierUniform

  def conf(self) -> Dict[str, str]:
    return {'gain': str(self._gain)}


class XavierNormalInitializer(Initializer):

  def __init__(self, gain: float = 1.0):
    self._gain = gain

  def type(self) -> kraken_native.InitializerType:
    return kraken_native.InitializerType.kXavierNormal

  def conf(self) -> Dict[str, str]:
    return {'gain': str(self._gain)}

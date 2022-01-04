# coding=utf-8


class LR(object):

  def __init__(self):
    super(LR, self).__init__()

  def step(self):
    raise NotImplementedError

  def lr(self) -> float:
    raise NotImplementedError


class ConstantLR(LR):

  def __init__(self, lr: float):
    super(ConstantLR, self).__init__()
    self._lr = lr

  def step(self):
    pass

  def lr(self) -> float:
    return self._lr

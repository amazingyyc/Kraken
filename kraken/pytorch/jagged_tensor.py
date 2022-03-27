# coding=utf-8

from typing import List, Optional
import torch


class JaggedTensor(object):
  '''JaggedTensor is combined by a list of tensors that has same dimension except the first rank.
  Like a tensor list:[t0, t1, t2 .. tn] we must make sure t0.shape[1:] == t1.shape[1:] ... == tn.shape[1:].
  and the offsets will be: [0, t0.shape[0], t0.shape[0] + t1.shape[0], ... , sum(ti.shape[0])]
  The weights'shape must be:(sum(ti.shape[0])).
  '''

  def __init__(self,
               values: torch.Tensor,
               offsets: torch.Tensor,
               weights: Optional[torch.Tensor] = None):
    self._check_params(values, offsets, weights)

    self._values: torch.Tensor = values
    self._offsets: torch.Tensor = offsets
    self._weights: Optional[torch.Tensor] = weights

  def values(self) -> torch.Tensor:
    return self._values

  def offsets(self) -> torch.Tensor:
    return self._offsets

  def weights(self) -> Optional[torch.Tensor]:
    return self._weights

  def _check_params(self,
                    values: torch.Tensor,
                    offsets: torch.Tensor,
                    weights: Optional[torch.Tensor] = None):
    '''Check whether values/offsets is valid'''
    offsets_l = offsets.tolist()

    if offsets_l[0] != 0 or offsets_l[-1] != values.shape[0]:
      raise ValueError(
          f'JaggedTensor\'s offsets is invalid:{offsets_l}, values shape:{values.shape}'
      )

    for i in range(1, len(offsets_l)):
      if offsets_l[i] < offsets_l[i - 1]:
        raise ValueError(
            f'JaggedTensor\'s offsets is invalid:{offsets_l}, values shape:{values.shape}'
        )

    if weights:
      if weights.shape != values.shape:
        raise ValueError(
            f'JaggedTensor need values and weights has same shape' \
            f'but got weights shape:{weights.shape}, values shape:{values.shape}.'
        )

  @staticmethod
  def from_list_tensors(list_values: List[torch.Tensor],
                        list_weigths: Optional[List[torch.Tensor]] = None):
    assert len(list_values) > 0

    offset_v = [0]
    tail_dims = list_values[0].shape[1:]

    for v in list_values:
      if v.shape[1:] != tail_dims:
        raise ValueError(
            'JaggedTensor need values has tha same dimension except the first rank.'
        )

      offset_v.append(offset_v[-1] + v.shape[0])

    # shape will be:[sum(tensors[i].shape[0]] + tail_dims
    values = torch.cat(list_values, dim=0)
    offsets = torch.as_tensor(offset_v)
    weights = None

    if list_weigths:
      assert len(list_values) == len(list_weigths)

      for i in range(len(list_values)):
        if list_weigths[i].shape != list_values[i].shape:
          raise ValueError(
              'JaggedTensor need values and weights has same shape.')

      weights = torch.cat(list_weigths, 0)

    return JaggedTensor(values=values, offsets=offsets, weights=weights)

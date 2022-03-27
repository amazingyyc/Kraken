# coding=utf-8

from typing import List
import torch
from kraken.pytorch.combine_embedding import CombineEmbedding
from kraken.pytorch.initializer import Initializer
from kraken.pytorch.jagged_tensor import JaggedTensor
from kraken.pytorch.jagged_embedding_funcs import JaggedEmbeddingSumFunction, JaggedEmbeddingMeanFunction


class CombineJaggedEmbedding(torch.nn.Module):

  def __init__(self,
               dimensions: List[int],
               dtypes: List[torch.dtype] = None,
               initializers: List[Initializer] = None,
               names: List[str] = None,
               modes: List[str] = None,
               patch_values: List[float] = None):
    super(CombineJaggedEmbedding, self).__init__()

    if modes:
      assert len(dimensions) == len(modes)

      for mode in modes:
        if mode not in ['sum', 'mean']:
          raise ValueError(
              'CombineJaggedEmbedding\'s mode only support sum/mean.')

    if patch_values:
      assert len(dimensions) == len(patch_values)

    self.combine_embedding = CombineEmbedding(dimensions=dimensions,
                                              dtypes=dtypes,
                                              initializers=initializers,
                                              names=names)

    self._modes = modes
    self._patch_values = patch_values

    if self._modes is None:
      self._modes = ['sum'] * len(dimensions)

    if self._patch_values is None:
      self._patch_values = [0.0] * len(dimensions)

  def forward(self, inputs: List[JaggedTensor]):
    assert len(self._modes) == len(inputs)
    indices = [input.values() for input in inputs]

    list_embeddings = self.combine_embedding(indices)
    assert len(list_embeddings) == len(inputs)

    outputs = []

    for i in range(len(list_embeddings)):
      embeddings = list_embeddings[i]
      offsets = inputs[i].offsets()
      weights = inputs[i].weights()

      if weights:
        embeddings *= weights.unsqueeze(-1)

      if self._modes[i] == 'sum':
        output = JaggedEmbeddingSumFunction.apply(embeddings, offsets,
                                                  self._patch_values[i])
      elif self._modes[i] == 'mean':
        output = JaggedEmbeddingMeanFunction.apply(embeddings, offsets,
                                                   self._patch_values[i])
      else:
        raise ValueError(
            'CombineJaggedEmbedding\'s mode only support sum/mean.')

      outputs.append(output)

    return outputs

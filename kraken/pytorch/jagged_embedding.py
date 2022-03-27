# coding=utf-8

import torch
from kraken.pytorch.embedding import Embedding
from kraken.pytorch.initializer import Initializer, NormalInitializer
from kraken.pytorch.jagged_tensor import JaggedTensor
from kraken.pytorch.jagged_embedding_funcs import JaggedEmbeddingSumFunction, JaggedEmbeddingMeanFunction


class JaggedEmbedding(torch.nn.Module):

  def __init__(self,
               dimension: int,
               dtype: torch.dtype = torch.float32,
               initializer: Initializer = NormalInitializer(),
               name: str = None,
               mode='sum',
               patch_value: float = 0.0):
    super(JaggedEmbedding, self).__init__()

    if mode not in ['sum', 'mean']:
      raise ValueError('JaggedEmbedding\'s mode only support sum/mean.')

    self.embedding = Embedding(dimension=dimension,
                               dtype=dtype,
                               initializer=initializer,
                               name=name)

    self._mode = mode
    self._patch_value = patch_value

  def forward(self, input: JaggedTensor):
    indices = input.values()
    weights = input.weights()
    offsets = input.offsets()

    embeddings = self.embedding(indices)

    if weights:
      '''When weights is not None the weight's shape must same with indices's shape,
      so after get embeddings we need expend the weights's dims
      to make sure the shape is right for mul.'''
      embeddings *= weights.unsqueeze(-1)

    if self._mode == 'sum':
      return JaggedEmbeddingSumFunction.apply(embeddings, offsets,
                                              self._patch_value)
    elif self._mode == 'mean':
      return JaggedEmbeddingMeanFunction.apply(embeddings, offsets,
                                               self._patch_value)
    else:
      raise ValueError('JaggedEmbedding\'s mode only support sum/mean.')

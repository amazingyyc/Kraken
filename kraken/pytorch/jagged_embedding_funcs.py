# coding=utf-8

import torch
import kraken_native


class JaggedEmbeddingSumFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, values, offsets, patch_value):
    ctx.save_for_backward(offsets)

    return kraken_native.jagged_sum_forward(values, offsets, patch_value)

  @staticmethod
  def backward(ctx, grads):
    offsets, = ctx.saved_tensors

    return kraken_native.jagged_sum_backward(offsets, grads)


class JaggedEmbeddingMeanFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, values, offsets, patch_value):
    ctx.save_for_backward(offsets)

    return kraken_native.jagged_mean_forward(values, offsets, patch_value)

  @staticmethod
  def backward(ctx, grads):
    offsets, = ctx.saved_tensors

    return kraken_native.jagged_mean_backward(offsets, grads)

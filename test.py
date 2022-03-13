# coding=utf-8
import kraken.pytorch as kk
import numpy as np
import torch

kk.initialize('127.0.0.1:50000')

kk.init_model('test_model', kk.OptimType.kSGD)

t0 = torch.rand(2, 3)
t1 = torch.rand(2, 3)
t2 = torch.rand(3, 4)
t3 = torch.rand(2, 3)
t4 = torch.rand(3, 4)

kk.register_dense_table('dense_table_0', t0)
kk.register_dense_table('dense_table_1', t1)
kk.register_dense_table('dense_table_2', t2)
kk.register_dense_table('dense_table_3', t3)
kk.register_dense_table('dense_table_4', t4)

kk.stop()
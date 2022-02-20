# coding=utf-8
import kraken_watcher as kw
import numpy as np

watcher = kw.Watcher()
watcher.load('/root/code/Kraken/0')

modle_info = watcher.model_info()
print('modle_info:', modle_info)

dense_table_infos = watcher.dense_table_infos()
print('dense_table_infos:', dense_table_infos)

sparse_table_infos = watcher.sparse_table_infos()
# exist_sparse_table_ids = watcher.exist_sparse_table_ids(1)

# t = watcher.dense_table_val(5)
# t = np.array(t, copy = False)

# t1 = watcher.sparse_table_val(1, 0)
# t1 = np.array(t1, copy = False)

# print('----------------------------')
# print(modle_info)
# print(modle_info.optim_type)
# print(dense_table_infos)
# print(sparse_table_infos)
# print(exist_sparse_table_ids)
# # print(t)
# print(t1)

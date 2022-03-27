# coding=utf-8

import unittest
from jagged_tensor_test import JaggedTensorTest
from jagged_embedding_funcs_test import JaggedEmbeddingFuncsTest

def suite():
    suite = unittest.TestSuite()
    suite.addTest(JaggedTensorTest('test_normal'))
    suite.addTest(JaggedTensorTest('test_multi_dimension'))
    suite.addTest(JaggedEmbeddingFuncsTest('test_sum'))
    suite.addTest(JaggedEmbeddingFuncsTest('test_mean'))
    return suite

if __name__ == '__main__':
  runner = unittest.TextTestRunner()
  runner.run(suite())

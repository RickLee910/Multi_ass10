# -*- coding: utf-8 -*-
# @Time    : 2/2/2021 10:32 PM
# @Author  : RICK LEE
# @FileName: KLT.py
# @Software: PyCharm
import numpy as np
from numpy import array as matrix, zeros, matmul
C = [[1.6944, 1.7361, 1.4028, 1.4167],
     [1.7361, 2.0278, 1.6944, 1.6667],
     [1.4028, 1.6944, 2.1111, 1.9167],
     [1.4167, 1.6667, 1.9167, 2.2500]]
C_test = [[0.1704, 0.6946, 0.2681],
          [0.4808,0.3752,0.3871],
          [0.4828,-0.6138,0.5439],
          [-0.7118,0.0034,0.6946]]
eig_values, eig_vectors = np.linalg.eig(C)
print(eig_values)
print(eig_vectors)


b = matrix([[1],[1],[1],[0]])
b_T = b.transpose()
b1 = zeros((4,1))
for i in range(4):
    b1 += matmul(b_T, eig_vectors[i]) * eig_vectors[:,[i]]
print(b1)
# -*- coding: utf-8 -*-
# @Time    : 2/2/2021 3:52 PM
# @Author  : RICK LEE
# @FileName: test.py
# @Software: PyCharm
######python
from sklearn.decomposition import PCA
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
im = cv.imread('Original-Lena-image-512-512-pixels.png')###导入图片
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)###将彩色图片灰度化
pca = PCA(n_components = 256)###PCA，提取特征为256
p = pca.fit_transform(imgray)###对原始数据进行降维
plt.imshow(im,'gray')
plt.show()###展示图片
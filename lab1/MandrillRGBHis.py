
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 为matplotlib设置后端
import matplotlib
matplotlib.use('TkAgg')

# 读取图像
mandrillRGB = cv.imread('mandrillRGB.jpg')

# 颜色通道
colors = ('b', 'g', 'r')

# 对于每一个颜色通道，计算直方图并绘制
for i, color in enumerate(colors):
    hist = cv.calcHist([mandrillRGB], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)

plt.title('Histogram for color scale picture')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.show()
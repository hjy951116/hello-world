import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
 

matfn=u'./code/uvl+ba.mat'
data=sio.loadmat(matfn)
# print(data)
img = cv.imread('./allframespng/000001.png')
imgcomp = img
xi=np.float32(data['uv'])
print(xi)
#yi=xi
yi=np.float32(xi/42)
for m in range (720):
    for n in range (1280):
        yi[m, n, 1] = - yi[m, n, 1] + n
        yi[m, n, 0] = - yi[m, n, 0] + m
print(yi)
imgcomp = cv.remap(img, yi[..., 1], yi[..., 0], interpolation = cv.INTER_CUBIC)
# imgcomp = cv.remap(img, x, x, interpolation = cv.INTER_CUBIC)
cv.imwrite('2compensation.png',imgcomp)
import cv2
import numpy as np
import matplotlib
 
img = cv2.imread('/Users/jaymichael/Downloads/py_project/tooth_py/test/1113.bmp', -1)
savepath = '/Users/jaymichael/Downloads/py_project/tooth_py/'
# 进行直方图均衡化
equ = cv2.equalizeHist(img)
# CLAHE有限对比适应性直方图均衡化
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))
cl1 = clahe.apply(img)
 
# 将三幅图像拼接在一起
res = np.hstack((img, equ, cl1))

cv2.imwrite(savepath+'/'+str(1000)+'.bmp',cl1)
cv2.imwrite(savepath+'/'+str(1001)+'.bmp',res)
cv2.imshow('img', res)

cv2.waitKey(0)
cv2.destroyAllWindows()
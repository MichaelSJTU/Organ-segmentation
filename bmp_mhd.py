import SimpleITK as sitk
import torch
import numpy
import cv2
import numpy as np
import os
import glob

img = []
img1 = cv2.imread('/Users/jaymichael/Downloads/py_project/tooth_py/test/1000.bmp',-1)
img1 = np.array(img1)
img.append(img1)

width = img1.shape[1]
height = img1.shape[0]
print(width)
print(height)
# 100 width* 75 height * 175 chanel

img_path = sorted(glob.glob(r'/Users/jaymichael/Downloads/py_project/tooth_py/test1/*.bmp'))
print(img_path[0])
print(img_path[3])
chanel = len(img_path)
print(chanel)

img_resize = np.zeros([chanel,height,width],dtype=np.uint32)

for i in range(chanel):
    #print(i)
    img = cv2.imread(img_path[i],-1)
    #print(img_path)
    img_resize[i,0:img1.shape[0],0:img1.shape[1]] = img
    img_resize[i,(height- img1.shape[0]) // 2:(height - img1.shape[0]) // 2 + img1.shape[0],
    (width - img1.shape[1]) // 2:(width - img1.shape[1]) // 2 + img1.shape[1]] = img

img_resize=np.reshape(img_resize,[chanel,height,width])
mhd_data = sitk.GetImageFromArray(img_resize)
#spacing1 = [0.1,0.1,0.1]
image = sitk.ReadImage('/Users/jaymichael/Downloads/py_project/tooth_py/glx1_hei.mhd')
spacing = image.GetSpacing()
print(spacing)
mhd_data.SetSpacing(spacing)
sitk.WriteImage(mhd_data, "/Users/jaymichael/Downloads/py_project/output/yangjiuping5_hei.mhd")
spacing = mhd_data.GetSpacing()
print("Image spacing:", spacing)
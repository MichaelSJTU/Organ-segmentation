import numpy
import cv2
import SimpleITK as sitk
import numpy as np
import numpy

image = sitk.ReadImage('/Users/jaymichael/Downloads/py_project/Croppedgroundtruth/gujiadong2_hei.mhd')
#image = sitk.ReadImage('/Users/jaymichael/Downloads/py_project/croppedtooth3/glx1_hei.mhd')
img_data = sitk.GetArrayFromImage(image) 
height = img_data.shape[1]
weight = img_data.shape[2]
channel = img_data.shape[0]
print(height)
print(weight)
print(channel)
savepath = '/Users/jaymichael/Downloads/py_project/tooth_py/test'

for i in range(channel):
    img = np.zeros((height,weight), dtype=np.uint64)
    img = img_data[i,:,:]
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    cv2.imwrite(savepath+'/'+str(i+1000)+'.bmp',img)
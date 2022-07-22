import cv2
import numpy as np
import numpy
from matplotlib import pyplot as plt
from skimage import morphology
import os
import glob

img_path = sorted(glob.glob(r'/Users/jaymichael/Downloads/py_project/tooth_py/test/*.bmp'))
print(len(img_path))
channel = len(img_path)
savepath = '/Users/jaymichael/Downloads/py_project/tooth_py/test1'
for j in range(len(img_path)):
    #print(i)
    img = cv2.imread(img_path[j],-1)
    #img = cv2.imread('/Users/jaymichael/Downloads/py_project/tooth_py/test/1092.bmp',-1)
    #print (img_path[i])
    # Step1. 加载图像
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Step2.阈值分割，将图像分为黑白两部分
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("thresh", thresh)

    # Step3. 对图像进行腐蚀
    
    thresh = 255 - thresh
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    erosion = cv2.erode(thresh,kernel)
    cv2.imshow("erosion", erosion)

    # Step4. 对“开运算”的结果进行膨胀，得到大部分都是背景的区域
    sure_bg = erosion
    contours, hierarchy = cv2.findContours(sure_bg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    mask = numpy.zeros(sure_bg.shape, numpy.uint8)
    for contour in contours:
        cv2.fillPoly(mask, [contour], 255)
    sure_bg[(mask > 0)] = 255
    #sure_bg = cv2.dilate(sure_bg, kernel, iterations=3)   
    cv2.imshow("sure_bg", sure_bg)
    print("number of contours:%d" % len(contours))
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    min_idx = np.argmin(area)
    print (max_idx)
    for k in range(len(contours)):
        cv2.fillConvexPoly(sure_bg, contours[k], 0)
    cv2.fillConvexPoly(sure_bg, contours[max_idx], 255)
    #show image with max connect components 
    sure_bg1 = 255 - sure_bg
    cv2.imshow("sure_bg1", sure_bg)
    # Step5.通过distanceTransform获取前景区域




    dist_transform = cv2.distanceTransform(sure_bg1, cv2.DIST_L2, 5)  # DIST_L1 DIST_C只能 对应掩膜为3    DIST_L2 可以为3或者5
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    #cv2.imshow("sure_fg", sure_fg)
    sure_fg1 = 255 - sure_fg


    img[(sure_fg1 == 0)] = 0
    cv2.imshow("result", img)

    equ = cv2.equalizeHist(img)
    # CLAHE有限对比适应性直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    cl1 = clahe.apply(img)
    

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(savepath+'/'+str(j+1000)+'.bmp',cl1)
    print("finished")



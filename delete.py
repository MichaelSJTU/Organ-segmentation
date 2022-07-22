import cv2
import SimpleITK as sitk
import numpy as np

def RemoveSmallConnectedCompont(sitk_maskimg, rate=1):
    """
    remove small object
    :param sitk_maskimg:input binary image
    :param rate:size rate
    :return:binary image
    """
    cc = sitk.ConnectedComponent(sitk_maskimg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, sitk_maskimg)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    not_remove = []
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if size > maxsize * rate:
            not_remove.append(l)
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage != maxlabel] = 0
    for i in range(len(not_remove)):
        outmask[labelmaskimage == not_remove[i]] = 255
    outmask_sitk = sitk.GetImageFromArray(outmask)
    outmask_sitk.SetDirection(sitk_maskimg.GetDirection())
    outmask_sitk.SetSpacing(sitk_maskimg.GetSpacing())
    outmask_sitk.SetOrigin(sitk_maskimg.GetOrigin())
    return outmask_sitk

if __name__=='__main__':
    image = sitk.ReadImage('/Users/jaymichael/Downloads/py_project/delete/glx1_label.mhd')
    ss = sitk.GetArrayFromImage(image) 
    RemoveSmallConnectedCompont(ss, rate=0.5)
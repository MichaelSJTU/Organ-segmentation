import SimpleITK as sitk

path_ct = '/Users/jaymichael/Downloads/py_project/tooth_py/tooth_cbct/new/xq_hei.mhd'
#path_ctlabel = '/Users/jaymichael/Downloads/py_project/tooth_py/zhangxin_label.mhd'
image_ct = sitk.ReadImage(path_ct, sitk.sitkInt16)
#image_ctlabel = sitk.ReadImage(path_ctlabel, sitk.sitkInt16)
center = [185,397,
          323,500, 
          260,455]
image_ct = image_ct[center[0]:center[1], center[2]:center[3], center[4]:center[5]]
#image_ctlabel = image_ctlabel[center[0]:center[1], center[2]:center[3], center[4]:center[5]]

sitk.WriteImage(image_ct, "/Users/jaymichael/Downloads/py_project/croppedtooth3/xq4_hei.mhd")
#sitk.WriteImage(image_ctlabel, "/Users/jaymichael/Downloads/py_project/croppedtooth2/gujiadong2_label.mhd")
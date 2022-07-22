path_ct = '/Users/jaymichael/Downloads/py_project/tooth_py/glx.mhd'
image_ct = sitk.ReadImage(path_ct, sitk.sitkInt16)
center = (140, 438, 285)
image_ct = image_ct[center[0]-50:center[0]+51, center[1]-38:center[1]+38, center[2]-70:center[2]+70]
sitk.WriteImage(image_ct, "/Users/jaymichael/Downloads/py_project/tooth_py/glx1.mhd")


path_ct = '/Users/jaymichael/Downloads/py_project/tooth_py/glx.mhd'
image_ct = sitk.ReadImage(path_ct, sitk.sitkInt16)
center = (210, 323, 300)
image_ct = image_ct[center[0]-40:center[0]+40, center[1]-43:center[1]+43, center[2]-90:center[2]+90]
sitk.WriteImage(image_ct, "/Users/jaymichael/Downloads/py_project/croppedtooth/glx2.mhd")

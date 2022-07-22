import SimpleITK as sitk
import matplotlib.pyplot as plt
#ase_path = '/glx.mhd'  
case_path = '/Users/jaymichael/Downloads/py_project/tooth_py/glx.mhd'

itkimage = sitk.ReadImage(case_path)   #这部分给出了关于图像的信息,可以打印处理查看，这里就不在显示了
#print(itkimage)
image = sitk.GetArrayFromImage(itkimage)     #z,y,x
#查看第100张图像
plt.figure()
plt.imshow(image[100,:,:])
plt.show()

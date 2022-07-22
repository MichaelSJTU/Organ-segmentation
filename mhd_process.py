import SimpleITK as sitk
import matplotlib.pyplot as plt
#mhd2rgb
path_address = '/Users/jaymichael/Downloads/py_project/tooth_py/glx.mhd'
image = sitk.ReadImage(path_address)
image_array = sitk.GetArrayFromImage(image)
print(image_array)
# #rgb2mhd
# if i.endswith('.png'):
# img = cv2.imread(os.path.join(rgb_path, i),0)
# # print ("image index:", i[:-4])
# img = np.array(img)
# print(img.shape)
# ept_width = img.shape[1]
# ept_height = img.shape[0]
# ept_chanel = 1
# img_resize = np.zeros([ept_chanel, ept_height, ept_width], dtype=np.uint8)
# img_resize[:,(ept_height- img.shape[0]) // 2:(ept_height - img.shape[0]) // 2 + img.shape[0],
# (ept_width - img.shape[1]) // 2:(ept_width - img.shape[1]) // 2 + img.shape[1]] = img
# img_resize=np.reshape(img_resize,[600,512])
# img_resize[img_resize==255]=1
# mhd_data = sitk.GetImageFromArray(img_resize)
# sitk.WriteImage(mhd_data, mhd_gray_path + i[:-4] + ".mhd")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# 读取彩色图像
rootpath = "/home/jiahua/liuy/hsi_pipeline/final_pic/final_input/"
rgb_savepath = "/home/jiahua/liuy/hsi_pipeline/final_pic/syc/"
# rootpath = "C:\\Users\\xiao\\Desktop\\"
# rootpath = "D:\\Download\\chart_and_stuffed_toy_ms\\"
# rootpath = "D:\\Download\\"
# 读取两张图像

name1 = "scene09_mixture1123.png"
name2 = "scene09_mixturergb.png"

image1 = cv2.imread(rootpath+name1, cv2.IMREAD_COLOR)  # 以灰度模式读取第一张图像
image2 = cv2.imread(rootpath+name2, cv2.IMREAD_COLOR)  # 以灰度模式读取第二张图像
# 转换为灰度图像
image1 = image1.astype(np.float32) / 255.0
image2 = image2.astype(np.float32) / 255.0

# 分离颜色通道
b1, g1, r1 = cv2.split(image1)
b2, g2, r2 = cv2.split(image2)

# 进行傅里叶变换
fft_b1 = np.fft.fft2(b1)
fft_b1 = np.fft.fftshift(fft_b1)

fft_g1 = np.fft.fft2(g1)
fft_r1 = np.fft.fft2(r1)
fft_b2 = np.fft.fft2(b2)
fft_b2 = np.fft.fftshift(fft_b2)
fft_g2 = np.fft.fft2(g2)
fft_r2 = np.fft.fft2(r2)
# rgb_savepath = "C:\\Users\\xiao\\Desktop\\AAAI\\fig\\paper\\spa_amp_pha"

# 计算幅度谱和相位谱
magnitude_spectrum1_b = 20 * np.log(np.abs(fft_b1))
magnitude_spectrum1_g = 20 * np.log(np.abs(fft_g1))
magnitude_spectrum1_r = 20 * np.log(np.abs(fft_r1))
magnitude_spectrum2_b = 20 * np.log(np.abs(fft_b2))
magnitude_spectrum2_g = 20 * np.log(np.abs(fft_g2))
magnitude_spectrum2_r = 20 * np.log(np.abs(fft_r2))
phase_spectrum1_b = np.angle(fft_b1)
phase_spectrum1_g = np.angle(fft_g1)
phase_spectrum1_r = np.angle(fft_r1)
phase_spectrum2_b = np.angle(fft_b2)
phase_spectrum2_g = np.angle(fft_g2)
phase_spectrum2_r = np.angle(fft_r2)

plt.imsave(os.path.join(rgb_savepath, name1[:-4]+"amp"+name1[-4:]),magnitude_spectrum1_b)
plt.imsave(os.path.join(rgb_savepath,name2[:-4]+"amp"+name2[-4:]),magnitude_spectrum2_b)
plt.imsave(os.path.join(rgb_savepath, name1[:-4]+"pha"+name1[-4:]),phase_spectrum1_b)
plt.imsave(os.path.join(rgb_savepath, name2[:-4]+"pha"+name2[-4:]),phase_spectrum2_b)

# 显示幅度谱和相位谱热力图
# plt.subplot(2, 3, 1), plt.imshow(magnitude_spectrum1_b, cmap='hot')
# plt.title('Magnitude Spectrum 1 (Blue)'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 3, 2), plt.imshow(magnitude_spectrum1_g, cmap='hot')
# plt.title('Magnitude Spectrum 1 (Green)'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 3, 3), plt.imshow(magnitude_spectrum1_r, cmap='hot')
# plt.title('Magnitude Spectrum 1 (Red)'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 3, 4), plt.imshow(phase_spectrum1_b, cmap='hot')
# plt.title('Phase Spectrum 1 (Blue)'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 3, 5), plt.imshow(phase_spectrum1_g, cmap='hot')
# plt.title('Phase Spectrum 1 (Green)'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 3, 6), plt.imshow(phase_spectrum1_r, cmap='hot')
# plt.title('Phase Spectrum 1 (Red)'), plt.xticks([]), plt.yticks([])
# plt.show()




# import numpy as np
# import matplotlib.pyplot as plt

# # 假设你有图像的振幅信息（amplitude），相位信息设置为全零数组
# phaseb = phase_spectrum1_b
# phaseg = phase_spectrum1_g
# phaser = phase_spectrum1_r
# amplitude = np.ones_like(phase)
# amplitude = amplitude*0.5
# # 将相位信息设为全零数组，表示所有相位都是零
# # phase = np.ones_like(amplitude)

# # 使用逆傅里叶变换来重建图像
# spectrum_complexb = amplitude * np.exp(1j * phaseb)
# reconstructed_imageb = np.fft.ifft2(spectrum_complexb).real
# spectrum_complexg = amplitude * np.exp(1j * phaseg)
# reconstructed_imageg = np.fft.ifft2(spectrum_complexg).real
# spectrum_complexr = amplitude * np.exp(1j * phaser)
# reconstructed_imager = np.fft.ifft2(spectrum_complexr).real
# # 显示原始图像和重建图像
# reconstructed_image1 = cv2.merge((reconstructed_imageb, reconstructed_imageg, reconstructed_imager)).astype(np.float32)
# plt.subplot(1, 2, 1)

# plt.imshow(reconstructed_imageb,cmap='gray')

# plt.title('Reconstructed Image using Amplitude')
# plt.axis('off')

# plt.show()





# # 从幅度谱和相位谱还原原始图像
# reconstructed_b1 = np.fft.ifft2(np.exp(1j * phase_spectrum1_b) * np.abs(fft_b2)).real
# reconstructed_g1 = np.fft.ifft2(np.exp(1j * phase_spectrum1_g) * np.abs(fft_g2)).real
# reconstructed_r1 = np.fft.ifft2(np.exp(1j * phase_spectrum1_r) * np.abs(fft_r2)).real

# reconstructed_b2 = np.fft.ifft2(np.exp(1j * phase_spectrum2_b) * np.abs(fft_b1)).real
# reconstructed_g2 = np.fft.ifft2(np.exp(1j * phase_spectrum2_g) * np.abs(fft_g1)).real
# reconstructed_r2 = np.fft.ifft2(np.exp(1j * phase_spectrum2_r) * np.abs(fft_r1)).real

# # 合并颜色通道
# reconstructed_image1 = cv2.merge((reconstructed_b1, reconstructed_g1, reconstructed_r1)).astype(np.float32)
# reconstructed_image2 = cv2.merge((reconstructed_b2, reconstructed_g2, reconstructed_r2)).astype(np.float32)

# # reconstructed_image1 = cv2.normalize(reconstructed_image1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# # reconstructed_image2 = cv2.normalize(reconstructed_image2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# # print(reconstructed_image1)

# # 显示原始图像和还原图像
# def truncate_image(image):
#     # 将小于0的值设置为0
#     image[image < 0] = 0
#     # 将大于1的值设置为1
#     image[image > 1] = 1
#     return image

# # plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
# # plt.title('Original Image 1'), plt.xticks([]), plt.yticks([])
# reconstructed_image1 = truncate_image(reconstructed_image1)
# # plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(reconstructed_image1, cv2.COLOR_BGR2RGB))
# # plt.title('Reconstructed Image 1'), plt.xticks([]), plt.yticks([])
# # plt.subplot(2, 2, 3), plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
# # plt.title('Original Image 2'), plt.xticks([]), plt.yticks([])
# reconstructed_image2 = truncate_image(reconstructed_image2)
# # plt.subplot(2, 2, 4), plt.imshow(cv2.cvtColor(reconstructed_image2, cv2.COLOR_BGR2RGB))
# # plt.title('Reconstructed Image 2'), plt.xticks([]), plt.yticks([])
# # plt.show()

# # import os
# plt.imsave(os.path.join(rgb_savepath, "gt.png"),cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
# plt.imsave(os.path.join(rgb_savepath, "syc_noisy.png"),cv2.cvtColor(reconstructed_image1, cv2.COLOR_BGR2RGB))
# plt.imsave(os.path.join(rgb_savepath, "noisy.png"),cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
# plt.imsave(os.path.join(rgb_savepath, "syc_gt.png"),cv2.cvtColor(reconstructed_image2, cv2.COLOR_BGR2RGB))
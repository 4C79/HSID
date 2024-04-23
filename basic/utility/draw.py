import matplotlib.pyplot as plt

import scipy.io as sio

import numpy as np
from scipy.io import loadmat

import cv2
import os

from matplotlib.colors import Normalize

def drawPha():

    basedir = "/home/jiahua/liuy/hsi_pipeline/final_pic/final_input/"

    newdir = "/home/jiahua/liuy/hsi_pipeline/final_pic/final_output/"

    fns = os.listdir(basedir)

    # result_gt = np.zeros([2048,2048], dtype=np.float64)

    for fn in fns:

        image1 = cv2.imread(os.path.join(basedir,fn),0)  # 以灰度模式读取第一张图像


        image1 = cv2.resize(image1, (256, 256))

        image1 = image1.astype(np.float64) / 255.0

        dft = np.fft.fft2(image1)

        fft_fig1 = np.fft.fftshift(dft)

        # 计算频谱幅值
        absF = np.abs(dft)

        # 计算相位
        phi = np.angle(dft)  

        reconst2 = np.fft.ifft2((np.log(1+absF)) * np.exp(1j))
        reconst2 = reconst2.clip(0,1)
        reconst2 = (reconst2 * 255).astype('uint8')

        amplitude = np.abs(dft) 
        norm = Normalize(amplitude.min(), amplitude.max())

        # 将amplitude应用Normalize映射到0-255
        amplitude_scaled = norm(amplitude)*255
        amplitude_scaled = amplitude_scaled.astype(np.uint8)

        # 转换为RGB图像保存
        plt.imsave('amplitude.jpg', amplitude_scaled)


        reconst3 = np.real(np.fft.ifft2(1.*np.exp(1j * phi)))
        reconst3 = reconst3.clip(0,1)
        # result_gt = result_gt + reconst3
        reconst3 = (reconst3 * 255).astype('uint8')

        

        # complex_arr = np.fft.ifft2(fig1_pha).imag

        # complex_arr = complex_arr.astype(np.float32)
    
        # complex_arr /= complex_arr.max()
        # complex_arr *= 255

        # complex_arr[complex_arr < 0] = 0
        # complex_arr = cv2.normalize(complex_arr, None, 0, 255, cv2.NORM_MINMAX)

        # complex_arr = complex_arr.astype(np.uint8)

        # complex_arr = cv2.Canny(complex_arr, 50, 150) 

        # sobelx = cv2.Sobel(complex_arr, cv2.CV_64F, 1, 0, ksize=5)
        # sobely = cv2.Sobel(complex_arr, cv2.CV_64F, 0, 1, ksize=5)
        # sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

        # phapath = fn.split('.')[0] + '_pha.jpg'
        phapath = fn.split('.')[0] + '_fft.jpg'

        # plt.imsave(os.path.join(newdir,phapath), fig1_pha)

        plt.imsave(os.path.join(newdir,phapath), reconst3,cmap="gray")


        amppath = fn.split('.')[0] + '_amp.jpg'
        plt.imsave(os.path.join(newdir,amppath),reconst2)
    # result_gt[result_gt > 1 ] = 1
    # result_gt = (result_gt * 255).astype('uint8')
    # plt.imsave('gt.jpg', reconst3,cmap="gray")

def divide():

    basedir = "/home/jiahua/liuy/hsi_pipeline/final_pic/noisy_sp_output/"

    newdir = "/home/jiahua/liuy/hsi_pipeline/final_pic/noisy_sp_cuted/"

    fns = os.listdir(basedir)

    for fn in fns:

        img = cv2.imread(os.path.join(basedir,fn))  # 以灰度模式读取第一张图像

        new_img = img[365:1567,941:2029,:]

        cv2.imwrite(os.path.join(newdir,fn), new_img)


if __name__ == '__main__':
    # divide()
    drawPha()
    # img = loadmat('/data1/jiahua/ly/kaist_1024_complex/1024_stripe/scene09_reflectance.mat')
    # print(img.keys())
    # data = img['gt'][...]
    # result_gt = np.zeros([2048,2048,3], dtype=np.float64)
    # # for i in range(31):
    # #     result_gt[:,:,i] = data[:,:,i]
    # result_gt[:,:,0] = data[:,:,9]
    # result_gt[:,:,1] = data[:,:,19]
    # result_gt[:,:,2] = data[:,:,29]
    # result_gt = result_gt * 255
    # cv2.imwrite('/home/jiahua/liuy/hsi_pipeline/figure/final_input/scene09_clean.png',result_gt)
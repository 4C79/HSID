import caffe
import scipy.io as io
from pathlib import Path
import os
import cv2
import scipy.io as sio
import skimage.io
import lmdb
import numpy as np
import sys
import scipy
from skimage import img_as_float, img_as_ubyte
import pickle as pl
from scipy.io import loadmat
import h5py
from utility import minmax_normalize

def saveResult(inpath,name,outpath):
    matr =scipy.io.loadmat(inpath)
    print(matr.keys())
    data = matr[name]
    print(data.shape)
    [w,h,band]=data.shape
    for i in range(0,band):
        img = np.squeeze(data[:,:,i])
        cv2.imwrite(outpath + str(i) + ".png",
                      img_as_ubyte(minmax_normalize(img[:, :])))

# inpath = "/home/xx/VDnet_result/Denoise_Result/Urban/3DVSSN/real_denoise.mat"
# #test_path='/home/xx/Codes/Python/VDNet/data3d/Urban_150.mat'
# outpath = "/home/xx/VDnet_result/denoise_bands/Urban/3DVSSN/"
# name ="urban"
# saveResult(inpath,name,outpath)

# inpath = "/home/xx/VDnet_result/Denoise_Result/CAVE/Toy/3DVSSN/mixture.mat"
# #test_path='/home/xx/Codes/Python/VDNet/data3d/Urban_150.mat'
# outpath = "/home/xx/VDnet_result/denoise_bands/CAVE/Toy/gt/"
# test_path='/home/xx/Codes/Python/VDNet/data3d/toy_center/toy_center.mat'
# name ="toy"
# saveResult(test_path,name,outpath)
#
# inpath = "/home/xx/VDnet_result/Denoise_Result/WDC/3DVSSN/50.mat"
# #test_path='/home/xx/Codes/Python/VDNet/data3d/Urban_150.mat'
# outpath = "/home/xx/VDnet_result/denoise_bands/WDC/3DVSSN/50/"
# test_path='/home/xx/Codes/Python/VDNet/data3d/toy_center/toy_center.mat'
# name ="dc"
# saveResult(inpath,name,outpath)
#
# inpath = "/home/xx/VDnet_result/Denoise_Result/WDC/3DVSSN/noiid.mat"
# #test_path='/home/xx/Codes/Python/VDNet/data3d/Urban_150.mat'
# outpath = "/home/xx/VDnet_result/denoise_bands/WDC/3DVSSN/noiid/"
# test_path='/home/xx/Codes/Python/VDNet/data3d/toy_center/toy_center.mat'
# name ="dc"
# saveResult(inpath,name,outpath)

inpath = "/home/xx/VDnet_result/Denoise_Result/WDC/3DVSSN/noiid.mat"
#test_path='/home/xx/Codes/Python/VDNet/data3d/Urban_150.mat'
outpath = "/home/xx/VDnet_result/denoise_bands/WDC/gt/"
test_path='/home/xx/Codes/Python/VDNet/data3d/dc_test_128/dc_test_128.mat'
name ="dc"
saveResult(test_path,name,outpath)
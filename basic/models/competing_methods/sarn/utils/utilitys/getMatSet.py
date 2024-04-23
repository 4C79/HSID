import h5py
from pandas import np
import lmdb
import caffe
import os
from scipy.io import loadmat
from utility.util import *
import torch

print('create Pavia University...')
datadir = '/home/hcx/xx/VDNet/data3d/PaviaU/'
fns = os.listdir(datadir)
fns = [fn.split('.')[0]+'.mat' for fn in fns]


def loadMat(datadir,fns,matKey,crop_sizes):
    mat_data = loadmat(datadir+fns[0])[matKey]
    print(mat_data.shape())
    mat_data= minmax_normalize(mat_data.transpose(2, 0, 1))
    print(mat_data.shape())
    #img_var = torch.from_numpy(img_np).type(dtype)
    #img_noisy_np = get_noisy_image(img_np, sigma_)
    #img_noisy_var = torch.from_numpy(img_noisy_np).type(dtype)

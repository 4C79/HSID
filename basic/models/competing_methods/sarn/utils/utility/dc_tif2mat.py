from util import *
import lmdb
import caffe
import numpy as np
from dataset import *
from utility import *
import scipy.io as sio
import skimage.io
# tif图
imgpath = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/LRMR_gauss55_7.mat'
data = loadmat(imgpath)["im_output"]
# data = data.astype(np.float64)
# data = minmax_normalize(data)


# data= np.transpose(data, (1,2,0))
# data = data.astype(np.float32)
# data = crop_any(data,95,810,200)
# print(data.shape)
# print(data)
# sio.savemat("/home/xiaojiahua/code/RepDnCNN_RRelu/data/DC/dc_200.mat", {'DC': data})
plt.imshow(data[:,:, 15],cmap='gray')
plt.show()
plt.imsave("/home/xiaojiahua/code/RepDnCNN_RRelu/data/0.png",data[:,:, 15],cmap=plt.cm.gray)

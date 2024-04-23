import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import cv2
from torch import nn
import h5py
import os
import random

import threading
from itertools import product
from scipy.io import loadmat
from functools import partial
from scipy.ndimage import zoom
from matplotlib.widgets import Slider
from PIL import Image

def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
   # compute 1 dimension gaussian
   gaussian_1D = np.linspace(-1, 1, k)
   # compute a grid distance from center
   x, y = np.meshgrid(gaussian_1D, gaussian_1D)
   distance = (x ** 2 + y ** 2) ** 0.5

   # compute the 2 dimension gaussian
   gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
   gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

   # normalize part (mathematically)
   if normalize:
       gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
   return gaussian_2D


def get_sobel_kernel(k=3):
   # get range
   range = np.linspace(-(k // 2), k // 2, k)
   # compute a grid the numerator and the axis-distances
   x, y = np.meshgrid(range, range)
   sobel_2D_numerator = x
   sobel_2D_denominator = (x ** 2 + y ** 2)
   sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
   sobel_2D = sobel_2D_numerator / sobel_2D_denominator
   return sobel_2D


def get_thin_kernels(start=0, end=360, step=45):
   k_thin = 3  # actual size of the directional kernel
   # increase for a while to avoid interpolation when rotating
   k_increased = k_thin + 2

   # get 0° angle directional kernel
   thin_kernel_0 = np.zeros((k_increased, k_increased))
   thin_kernel_0[k_increased // 2, k_increased // 2] = 1
   thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

   # rotate the 0° angle directional kernel to get the other ones
   thin_kernels = []
   for angle in range(start, end, step):
       (h, w) = thin_kernel_0.shape
       # get the center to not rotate around the (0, 0) coord point
       center = (w // 2, h // 2)
       # apply rotation
       rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
       kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

       # get the k=3 kerne
       kernel_angle = kernel_angle_increased[1:-1, 1:-1]
       is_diag = (abs(kernel_angle) == 1)  # because of the interpolation
       kernel_angle = kernel_angle * is_diag  # because of the interpolation
       thin_kernels.append(kernel_angle)
   return thin_kernels


class CannyFilter(nn.Module):
   def __init__(self,
                k_gaussian=3,
                mu=0,
                sigma=1,
                k_sobel=3,
                device = 'cuda:0'):
       super(CannyFilter, self).__init__()
       # device
       self.device = device
       # gaussian
       gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
       self.gaussian_filter = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_gaussian,
                                        padding=k_gaussian // 2,
                                        bias=False)
       self.gaussian_filter.weight.data[:,:] = nn.Parameter(torch.from_numpy(gaussian_2D), requires_grad=False)

       # sobel

       sobel_2D = get_sobel_kernel(k_sobel)
       self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                       out_channels=1,
                                       kernel_size=k_sobel,
                                       padding=k_sobel // 2,
                                       bias=False)
       self.sobel_filter_x.weight.data[:,:] = nn.Parameter(torch.from_numpy(sobel_2D), requires_grad=False)

       self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                       out_channels=1,
                                       kernel_size=k_sobel,
                                       padding=k_sobel // 2,
                                       bias=False)
       self.sobel_filter_y.weight.data[:,:] = nn.Parameter(torch.from_numpy(sobel_2D.T), requires_grad=False)

       # thin

       thin_kernels = get_thin_kernels()
       directional_kernels = np.stack(thin_kernels)

       self.directional_filter = nn.Conv2d(in_channels=1,
                                           out_channels=8,
                                           kernel_size=thin_kernels[0].shape,
                                           padding=thin_kernels[0].shape[-1] // 2,
                                           bias=False)
       self.directional_filter.weight.data[:, 0] = nn.Parameter(torch.from_numpy(directional_kernels), requires_grad=False)

       # hysteresis

       hysteresis = np.ones((3, 3)) + 0.25
       self.hysteresis = nn.Conv2d(in_channels=1,
                                   out_channels=1,
                                   kernel_size=3,
                                   padding=1,
                                   bias=False)
       self.hysteresis.weight.data[:,:] = nn.Parameter(torch.from_numpy(hysteresis), requires_grad=False)

   def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=True):
       # set the setps tensors
       B, C, H, W = img.shape
       blurred = torch.zeros((B, C, H, W)).to(self.device)
       grad_x = torch.zeros((B, 1, H, W)).to(self.device)
       grad_y = torch.zeros((B, 1, H, W)).to(self.device)
       grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
       grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

       # gaussian

       for c in range(C):
           blurred[:, c:c + 1,:,:] = self.gaussian_filter(img[:, c:c + 1,:,:])
           grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c + 1,:,:])
           grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c + 1,:,:])

       # thick edges

       grad_x, grad_y = grad_x / C, grad_y / C
       grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
       grad_orientation = torch.atan2(grad_y, grad_x)
       grad_orientation = grad_orientation * (180 / np.pi) + 180  # convert to degree
       grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

       # thin edges

       directional = self.directional_filter(grad_magnitude)
       # get indices of positive and negative directions
       positive_idx = (grad_orientation / 45) % 8
       negative_idx = ((grad_orientation / 45) + 4) % 8
       thin_edges = grad_magnitude.clone()
       # non maximum suppression direction by direction
       for pos_i in range(4):
           neg_i = pos_i + 4
           # get the oriented grad for the angle
           is_oriented_i = (positive_idx == pos_i) * 1
           is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
           pos_directional = directional[:, pos_i]
           neg_directional = directional[:, neg_i]
           selected_direction = torch.stack([pos_directional, neg_directional])

           # get the local maximum pixels for the angle
           # selected_direction.min(dim=0)返回一个列表[0]中包含两者中的小的，[1]包含了小值的索引
           is_max = selected_direction.min(dim=0)[0] > 0.0
           is_max = torch.unsqueeze(is_max, dim=1)

           # apply non maximum suppression
           to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
           thin_edges[to_remove] = 0.0

       # thresholds

       if low_threshold is not None:
           low = thin_edges > low_threshold

           if high_threshold is not None:
               high = thin_edges > high_threshold
               # get black/gray/white only
               thin_edges = low * 0.5 + high * 0.5

               if hysteresis:
                   # get weaks and check if they are high or not
                   weak = (thin_edges == 0.5) * 1
                   weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                   thin_edges = high * 1 + weak_is_high * 1
           else:
               thin_edges = low * 1

       return thin_edges * 255

def Data2Volume(data, ksizes, strides):
    """
    Construct Volumes from Original High Dimensional (D) Data
    """
    dshape = data.shape
    PatNum = lambda l, k, s: (np.floor( (l - k) / s ) + 1)    

    TotalPatNum = 1
    for i in range(len(ksizes)):
        TotalPatNum = TotalPatNum * PatNum(dshape[i], ksizes[i], strides[i])
        
    V = np.zeros([int(TotalPatNum)]+ksizes); # create D+1 dimension volume

    args = [range(kz) for kz in ksizes]
    for s in product(*args):        
        s1 = (slice(None),) + s
        s2 = tuple([slice(key, -ksizes[i]+key+1 or None, strides[i]) for i, key in enumerate(s)])     
        V[s1] = np.reshape(data[s2], (-1,))

    return V

def crop_center(img,cropx,cropy):
    _,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, starty:starty+cropy,startx:startx+cropx]

def crop_any(img,cropx,cropy,size):
    _,y,x = img.shape
    startx = cropx
    starty = cropy
    return img[:, starty:starty+size,startx:startx+size]
def rand_crop(img, cropx, cropy):
    _,y,x = img.shape
    x1 = random.randint(0, x - cropx)
    y1 = random.randint(0, y - cropy)
    return img[:, y1:y1+cropy, x1:x1+cropx]


def sequetial_process(*fns):
    """
    Integerate all process functions
    """
    def processor(data):
        for f in fns:
            data = f(data)
        return data
    return processor


def minmax_normalize(array):    
    amin = np.min(array)
    amax = np.max(array)

    return (array - amin) / (amax - amin)


def frame_diff(frames):
    diff_frames = frames[1:, ...] - frames[:-1, ...]
    return diff_frames


def visualize(filename, matkey, load=loadmat, preprocess=None):
    """
    Visualize a preprecessed hyperspectral image
    """
    if not preprocess:
        preprocess = lambda identity: identity
    mat = load(filename)
    data = preprocess(mat[matkey])
    print(data.shape)
    print(np.max(data), np.min(data))

    data = np.squeeze(data[:,:,:])
    Visualize3D(data)
    # Visualize3D(np.squeeze(data[:,0,:,:]))

def Visualize3D(data, meta=None):
    data = np.squeeze(data)

    for ch in range(data.shape[0]):        
        data[ch, ...] = minmax_normalize(data[ch, ...])
    
    print(np.max(data), np.min(data))

    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    frame = 0
    # l = plt.imshow(data[frame,:,:])
    
    l = plt.imshow(data[frame,:,:], cmap='gray') #shows 256x256 image, i.e. 0th frame
    # plt.colorbar()
    axcolor = 'lightgoldenrodyellow'
    axframe = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    sframe = Slider(axframe, 'Frame', 0, data.shape[0]-1, valinit=0)

    def update(val):
        frame = int(np.around(sframe.val))
        l.set_data(data[frame,:,:])
        if meta is not None:
            axframe.set_title(meta[frame])

    sframe.on_changed(update)

    plt.show()


def data_augmentation(image, mode=None):
    """
    Args:
        image: np.ndarray, shape: C X H X W
    """
    axes = (-2, -1)
    flipud = lambda x: x[:, ::-1, :] 
    
    if mode is None:
        mode = random.randint(0, 7)
    if mode == 0:
        # original
        image = image
    elif mode == 1:
        # flip up and down
        image = flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = np.rot90(image, axes=axes)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image, axes=axes)
        image = flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = np.rot90(image, k=2, axes=axes)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2, axes=axes)
        image = flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = np.rot90(image, k=3, axes=axes)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3, axes=axes)
        image = flipud(image)

    # we apply spectrum reversal for training 3D CNN, e.g. QRNN3D. 
    # disable it when training 2D CNN, e.g. MemNet
    if random.random() < 0.5:
        image = image[::-1, :, :] 
    
    return np.ascontiguousarray(image)


class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def __next__(self):
        self.lock.acquire()
        try:
            return next(self.it)
        finally:
            self.lock.release()


if __name__ == '__main__':
    """Code Usage Example"""
    """ICVL"""
    # hsi_rot = partial(np.rot90, k=-1, axes=(1,2))
    # crop = lambda img: img[:,-1024:, -1024:]
    # zoom_512 = partial(zoom, zoom=[1, 0.5, 0.5])
    # d2v = partial(Data2Volume, ksizes=[31,64,64], strides=[1,28,28])
    # preprocess = sequetial_process(hsi_rot, crop, minmax_normalize, d2v)

    # preprocess = sequetial_process(hsi_rot, crop, minmax_normalize)
    # datadir = 'Data/ICVL/Training/'
    # fns = os.listdir(datadir)
    # mat = h5py.File(os.path.join(datadir, fns[1]))
    # data = preprocess(mat['rad'])
    # data = np.linalg.norm(data, ord=2, axis=(1,2))

    """Common"""
    # print(data)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(data)
    # plt.show()

    # preprocess = sequetial_process(hsi_rot, crop, minmax_normalize, frame_diff)
    # visualize(os.path.join(datadir, fns[0]), 'rad', load=h5py.File, preprocess=preprocess)
    # visualize('Data/BSD/TrainingPatches/imdb_40_128.mat', 'inputs', load=h5py.File, preprocess=None)

    # preprocess = lambda x: np.transpose(x[4][0],(2,0,1))
    # preprocess = lambda x: minmax_normalize(np.transpose(np.array(x,dtype=np.float),(2,0,1)))

    # visualize('/media/kaixuan/DATA/Papers/Code/Data/PIRM18/sample/true_hr', 'hsi', load=loadmat, preprocess=preprocess)
    # visualize('/media/kaixuan/DATA/Papers/Code/Data/PIRM18/sample/img_1', 'true_hr', load=loadmat, preprocess=preprocess)

    # visualize('/media/kaixuan/DATA/Papers/Code/Matlab/ITSReg/code of ITSReg MSI denoising/data/real/new/Indian/Indian_pines.mat', 'hsi', load=loadmat, preprocess=preprocess)
    # visualize('/media/kaixuan/DATA/Papers/Code/Matlab/ECCV2018/Result/Indian/Indian_pines/QRNN3D-f.mat', 'R_hsi', load=loadmat, preprocess=preprocess)
    # visualize('/media/kaixuan/DATA/Papers/Code/Matlab/ECCV2018/Data/Pavia/PaviaU', 'input', load=loadmat, preprocess=preprocess)
    
    pass
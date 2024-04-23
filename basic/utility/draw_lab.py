import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.io import loadmat,savemat
from util import crop_center,crop_custom, Visualize3D, minmax_normalize, rand_crop,BandMinMaxQuantileStateful
import os
import numpy as np
import h5py
from os.path import join, exists
from scipy.io import loadmat, savemat

from util import crop_center,crop_custom, Visualize3D, minmax_normalize, rand_crop,BandMinMaxQuantileStateful
from PIL import Image
import torch
import spectral
from skimage import io
import cv2 
from PIL import Image, ImageDraw

# 由 mat文件开始处理， 通过rgb.三通道 生成原始图， 根目录 数据集/噪声类型/方法/.mat
# 选那种图，09 ，传统方法也要生成图 
# 一个放大细节 ， 左上，右下

def normalize_list(input_list):
    min_val = min(input_list)
    max_val = max(input_list)
    normalized_list = [(x - min_val) / (max_val - min_val) for x in input_list]
    return normalized_list

def get_india_rgb():

    # img = loadmat('/data1/jiahua/ly/india_all_method_mat/fidnet.mat')
    # img = img['output'][...]
    # img = img.transpose((1,0,2))
    # savemat('/data1/jiahua/ly/india_all_method_mat/fidnet.mat',{'output':img})

    save_path = "/home/jiahua/liuy/hsi_pipeline/draw_pic/india/"
    test_path = '/data1/jiahua/ly/india_all_method_mat/'
     
    rect_list = os.listdir(test_path)

    for rect in rect_list:

        img = loadmat(os.path.join(test_path,rect))  
        print(img.keys())
        data = img['output'][...]
        result_gt = np.zeros([128,128,3], dtype=np.float64)
        # for i in range(31):
            # result_gt[:,:,1] = data[:,:,i]
        result_gt[:,:,0] = data[:,:,2]
        result_gt[:,:,1] = data[:,:,24]
        result_gt[:,:,2] = data[:,:,127]
        result_gt = minmax_normalize(result_gt)
        result_gt = result_gt * 255
        cv2.imwrite(save_path+rect[:-4]+'.png',result_gt)
    
def drawRect():

    save_path = "/home/jiahua/liuy/hsi_pipeline/draw_pic/urban_rect/"
    # rgb_path = "/home/jiahua/liuy/hsi_pipeline/figure/scene09_gt/"

    test_path = '/home/jiahua/liuy/hsi_pipeline/draw_pic/urban/'
     
    rect_list = os.listdir(test_path)

    for rect in rect_list:
        # rgb_path_custom = rgb_path+dataset+"\\rgb\\"
        # image_path =  rgb_path_custom + rect+".png"
    #     print(image_path)
        img = cv2.imread(os.path.join(test_path,rect), -1)  #在cv2.imread参数中加入-1，表示不改变读取图像的类型直接读取

        shift =1
        lengh= 32
        lengw = 32
        i = 156 #要放大区域的左上角的 x 坐标（竖轴坐标）
        j = 200 #要放大区域的左上角的 y 坐标（横轴坐标）
        h = i+lengh  #要放大区域的高
        w = j+lengw  #要放大区域的宽
        rate = 4
        h_2 = rate*lengh
        w_2 = rate*lengw

        pt1_1 = (j, i)  # 长方形框左上角坐标
        pt2_1 = (w, h)  # 长方形框右下角坐标 

        patch = img[ i:h,j:w]  # numpy 里先x，后y，x轴沿垂直方向向下，y轴沿水平方向向右
        # plt.imshow(patch)
        patch = cv2.resize(patch, (w_2, h_2))  #(w, h)

        img_h = img.shape[1]
        img_w = img.shape[0]
        rect_h = img_h - h_2
        rect_w = img_w - w_2
        # pt1 = (rect_w-shift,rect_h-shift )  # 长方形框左上角坐标
        # pt2 = (img_w-shift,img_h-shift)  # 长方形框右下角坐标 
        pt1 = (0,0)
        pt2 = (h_2,w_2)


        # img[rect_h-shift:img_h-shift,rect_w-shift:img_w-shift] = patch
        img[0:h_2,0:w_2] = patch

        cv2.rectangle(img, pt1_1, pt2_1, (0,  0,255 ), 3)  # cv2 里也是先x，后y，x轴沿水平方向向右，y轴沿垂直方向向下

        cv2.rectangle(img, pt1, pt2, (0,0,255), 3)  # cv2 里也是先x，后y，x轴沿水平方向向右，y轴沿垂直方向向下
        save_path_custom = save_path + rect
        img = cv2.resize(img,(512,512))
        cv2.imwrite(save_path_custom, img)

# (77,160),(108,191)

def drawwdcRect():

    save_path = "/data1/jiahua/result/wdc_rgb_rect/"
    # rgb_path = "/home/jiahua/liuy/hsi_pipeline/figure/scene09_gt/"

    test_path = '/data1/jiahua/result/wdc_rgb'
     
    rect_list = os.listdir(test_path)

    for rect in rect_list:
        # rgb_path_custom = rgb_path+dataset+"\\rgb\\"
        # image_path =  rgb_path_custom + rect+".png"
    #     print(image_path)
        img = cv2.imread(os.path.join(test_path,rect), -1)  #在cv2.imread参数中加入-1，表示不改变读取图像的类型直接读取

        shift =1
        lengh= 32
        lengw = 32
        i = 159 #要放大区域的左上角的 x 坐标（竖轴坐标）
        j = 77 #要放大区域的左上角的 y 坐标（横轴坐标）
        h = i+lengh  #要放大区域的高
        w = j+lengw  #要放大区域的宽
        rate = 4
        h_2 = rate*lengh
        w_2 = rate*lengw

        pt1_1 = (j, i)  # 长方形框左上角坐标
        pt2_1 = (w, h)  # 长方形框右下角坐标 

        patch = img[ i:h,j:w]  # numpy 里先x，后y，x轴沿垂直方向向下，y轴沿水平方向向右
        # plt.imshow(patch)
        patch = cv2.resize(patch, (w_2, h_2))  #(w, h)

        img_h = img.shape[1]
        img_w = img.shape[0]
        rect_h = img_h - h_2
        rect_w = img_w - w_2
        # pt1 = (rect_w-shift,rect_h-shift )  # 长方形框左上角坐标
        # pt2 = (img_w-shift,img_h-shift)  # 长方形框右下角坐标 
        pt1 = (0,0)
        pt2 = (h_2,w_2)


        # img[rect_h-shift:img_h-shift,rect_w-shift:img_w-shift] = patch
        img[0:h_2,0:w_2] = patch

        cv2.rectangle(img, pt1_1, pt2_1, (0,  0,255 ), 3)  # cv2 里也是先x，后y，x轴沿水平方向向右，y轴沿垂直方向向下

        cv2.rectangle(img, pt1, pt2, (0,0,255), 3)  # cv2 里也是先x，后y，x轴沿水平方向向右，y轴沿垂直方向向下
        save_path_custom = save_path + rect
        img = cv2.resize(img,(512,512))
        cv2.imwrite(save_path_custom, img)

def drawicvlRect():


    save_path = "/home/jiahua/liuy/hsi_pipeline/draw_pic/icvl_rect/"
    # rgb_path = "/home/jiahua/liuy/hsi_pipeline/figure/scene09_gt/"

    test_path = '/home/jiahua/liuy/hsi_pipeline/draw_pic/icvl/'
     
    rect_list = os.listdir(test_path)

    for rect in rect_list:

        img = cv2.imread(os.path.join(test_path,rect), -1)  #在cv2.imread参数中加入-1，表示不改变读取图像的类型直接读取

        shift =1
        lengh= 64
        lengw = 64
        i = 73 #要放大区域的左上角的 x 坐标（竖轴坐标）
        j = 95 #要放大区域的左上角的 y 坐标（横轴坐标）
        h = i+lengh  #要放大区域的高
        w = j+lengw  #要放大区域的宽
        rate = 4

        h_2 = int(rate*lengh)
        w_2 = int(rate*lengw)

        pt1_1 = (j, i)  # 长方形框左上角坐标
        pt2_1 = (w, h)  # 长方形框右下角坐标 

        patch = img[ i:h,j:w]  # numpy 里先x，后y，x轴沿垂直方向向下，y轴沿水平方向向右
        plt.imshow(patch)
        patch = cv2.resize(patch, (w_2, h_2))  #(w, h)
        # bicubic(img, rate, a)
        # cv2.imencode('.png', patch1)[1].tofile(save_path+k)  #也可以将 .png 改为 .jpg。
        # cv2.imwrite(save_path, patch1)


        img_h = img.shape[1]
        img_w = img.shape[0]
        rect_h = img_h - h_2
        rect_w = img_w - w_2
        pt1 = (rect_w-shift,rect_h-shift )  # 长方形框左上角坐标
        pt2 = (img_w-shift,img_h-shift)  # 长方形框右下角坐标 

        img[rect_h-shift:img_h-shift,rect_w-shift:img_w-shift] = patch
        # img[img_h-h_2:img_h,0:w_2] = patch
        cv2.rectangle(img, pt1_1, pt2_1, (0,  0,255 ), 2)  # cv2 里也是先x，后y，x轴沿水平方向向右，y轴沿垂直方向向下

        cv2.rectangle(img, pt1, pt2, (0,0,255), 2)  # cv2 里也是先x，后y，x轴沿水平方向向右，y轴沿垂直方向向下
        
        save_path_custom = save_path + rect
        cv2.imwrite(save_path_custom, img)

def drawcaveRect():


    save_path = "/home/jiahua/liuy/hsi_pipeline/draw_pic/cave_rect/"
    # rgb_path = "/home/jiahua/liuy/hsi_pipeline/figure/scene09_gt/"

    test_path = '/home/jiahua/liuy/hsi_pipeline/draw_pic/cave/'
     
    rect_list = os.listdir(test_path)

    for rect in rect_list:
        # rgb_path_custom = rgb_path+dataset+"\\rgb\\"
        # image_path =  rgb_path_custom + rect+".png"
    #     print(image_path)
        img = cv2.imread(os.path.join(test_path,rect), -1)  #在cv2.imread参数中加入-1，表示不改变读取图像的类型直接读取

        shift =1
        lengh= 64
        lengw = 64
        i = 265 #要放大区域的左上角的 x 坐标（竖轴坐标）
        j = 374 #要放大区域的左上角的 y 坐标（横轴坐标）
        h = i+lengh  #要放大区域的高
        w = j+lengw  #要放大区域的宽
        rate = 4
        h_2 = rate*lengh
        w_2 = rate*lengw

        pt1_1 = (j, i)  # 长方形框左上角坐标
        pt2_1 = (w, h)  # 长方形框右下角坐标 

        patch = img[ i:h,j:w]  # numpy 里先x，后y，x轴沿垂直方向向下，y轴沿水平方向向右
        # plt.imshow(patch)
        patch = cv2.resize(patch, (w_2, h_2))  #(w, h)

        img_h = img.shape[1]
        img_w = img.shape[0]
        rect_h = img_h - h_2
        rect_w = img_w - w_2
        # pt1 = (rect_w-shift,rect_h-shift )  # 长方形框左上角坐标
        # pt2 = (img_w-shift,img_h-shift)  # 长方形框右下角坐标 
        pt1 = (0,0)
        pt2 = (h_2,w_2)


        # img[rect_h-shift:img_h-shift,rect_w-shift:img_w-shift] = patch
        img[0:h_2,0:w_2] = patch

        cv2.rectangle(img, pt1_1, pt2_1, (0,  0,255 ), 3)  # cv2 里也是先x，后y，x轴沿水平方向向右，y轴沿垂直方向向下

        cv2.rectangle(img, pt1, pt2, (0,0,255), 3)  # cv2 里也是先x，后y，x轴沿水平方向向右，y轴沿垂直方向向下
        save_path_custom = save_path + rect
        img = cv2.resize(img,(512,512))
        cv2.imwrite(save_path_custom, img)



def draw_india_rect():

    save_path = "/data1/jiahua/result/indian_rgb_rect/"
    # save_all_path = '/home/jiahua/liuy/hsi_pipeline/draw_pic/india_rect/'

    test_path = '/data1/jiahua/result/indian_rgb/'
    
    rect_list = os.listdir(test_path)

    for rect in rect_list:

        img = cv2.imread(os.path.join(test_path,rect), -1)  #在cv2.imread参数中加入-1，表示不改变读取图像的类型直接读取

        shift =1
        lengh= 24
        lengw = 24
        i = 30 #要放大区域的左上角的 x 坐标（竖轴坐标）
        j = 0 #要放大区域的左上角的 y 坐标（横轴坐标）
        h = i+lengh  #要放大区域的高
        w = j+lengw  #要放大区域的宽
        rate = 2.5

        h_2 = int(rate*lengh)
        w_2 = int(rate*lengw)

        pt1_1 = (j, i)  # 长方形框左上角坐标
        pt2_1 = (w, h)  # 长方形框右下角坐标 

        patch = img[ i:h,j:w]  # numpy 里先x，后y，x轴沿垂直方向向下，y轴沿水平方向向右
        plt.imshow(patch)
        patch = cv2.resize(patch, (w_2, h_2))  #(w, h)
        # bicubic(img, rate, a)
        # cv2.imencode('.png', patch1)[1].tofile(save_path+k)  #也可以将 .png 改为 .jpg。
        # cv2.imwrite(save_path, patch1)


        img_h = img.shape[1]
        img_w = img.shape[0]
        rect_h = img_h - h_2
        rect_w = img_w - w_2
        pt1 = (rect_w-shift,rect_h-shift )  # 长方形框左上角坐标
        pt2 = (img_w-shift,img_h-shift)  # 长方形框右下角坐标 

        img[rect_h-shift:img_h-shift,rect_w-shift:img_w-shift] = patch
        # img[img_h-h_2:img_h,0:w_2] = patch
        cv2.rectangle(img, pt1_1, pt2_1, (0,  0,255 ), 2)  # cv2 里也是先x，后y，x轴沿水平方向向右，y轴沿垂直方向向下

        cv2.rectangle(img, pt1, pt2, (0,0,255), 2)  # cv2 里也是先x，后y，x轴沿水平方向向右，y轴沿垂直方向向下
        
        save_path_custom = save_path + rect
        cv2.imwrite(save_path_custom, img)

def draw_india_rgb():

    mothod = "macnet"
    save_path = "/home/jiahua/liuy/hsi_pipeline/pic/"
    # rgb_path = "/home/jiahua/liuy/hsi_pipeline/figure/scene09_gt/"

    test_path = '/data1/jiahua/ly/wdc_pic_mat/wdc_cut_mat/'
    
    save_path = save_path 
    test_path = test_path + mothod
     
    rect_list = os.listdir(test_path)
    result = np.zeros([3,256,256], dtype=np.float64)
    # cnt = 0

    for rect in rect_list:
        # rgb_path_custom = rgb_path+dataset+"\\rgb\\"
        # image_path =  rgb_path_custom + rect+".png"
    #     print(image_path)

        img = loadmat(os.path.join(test_path,rect))  #在cv2.imread参数中加入-1，表示不改变读取图像的类型直接读取
        img = img['output']

        if rect == "1_31.mat":
            result[0,:,:] = img[10,:,:]
        elif rect == "28_58.mat":
            result[1,:,:] = img[15,:,:]
        elif rect == "61_91.mat":
            result[2,:,:] = img[15,:,:]
    result = result.transpose((1,2,0))
    result = np.array(result*255,np.uint8)
    save_path = save_path + mothod
    cv2.imwrite(save_path + '.jpg', result)
    savemat(save_path + '.mat', {'output': result})
    
def make_erf():
    
    test_path = "/data3/jiahua/ly/test_data/kaist_1024_complex/1024_deadline/scene27_reflectanceNW.mat"
    save_path = '/data3/jiahua/ly/test_data/kaist_1024_complex/erf_test/test'
    
    img = loadmat(test_path)
    gt = img['gt']
    input = img['input']
    print(gt.shape)
    gt = gt[:256,:256,:]
    input = input[:256,:256,:]
    savemat(save_path + '.mat', {'gt':gt,'input': input})

def drawWdcRect():

    save_path = "/home/jiahua/liuy/hsi_pipeline/final_pic/wdc_cuted/"
    test_path = '/home/jiahua/liuy/hsi_pipeline/draw_pic/wdc/'
    
     
    rect_list = os.listdir(test_path)

    for rect in rect_list:
        # rgb_path_custom = rgb_path+dataset+"\\rgb\\"
        # image_path =  rgb_path_custom + rect+".png"
    #     print(image_path)
        if rect[-4:] != ".png":
            continue

        img = cv2.imread(os.path.join(test_path,rect), -1)  #在cv2.imread参数中加入-1，表示不改变读取图像的类型直接读取

        shift =1
        lengh= 32
        lengw = 32
        i = 160 #要放大区域的左上角的 x 坐标（竖轴坐标）
        j = 77 #要放大区域的左上角的 y 坐标（横轴坐标）
        h = i+lengh  #要放大区域的高
        w = j+lengw  #要放大区域的宽
        rate = 4
        h_2 = rate*lengh
        w_2 = rate*lengw

        pt1_1 = (j, i)  # 长方形框左上角坐标
        pt2_1 = (w, h)  # 长方形框右下角坐标 

        patch = img[ i:h,j:w]  # numpy 里先x，后y，x轴沿垂直方向向下，y轴沿水平方向向右
        # plt.imshow(patch)
        patch = cv2.resize(patch, (w_2, h_2))  #(w, h)

        img_h = img.shape[1]
        img_w = img.shape[0]
        rect_h = img_h - h_2
        rect_w = img_w - w_2
        # pt1 = (rect_w-shift,rect_h-shift )  # 长方形框左上角坐标
        # pt2 = (img_w-shift,img_h-shift)  # 长方形框右下角坐标 
        pt1 = (0,0)
        pt2 = (h_2,w_2)


        # img[rect_h-shift:img_h-shift,rect_w-shift:img_w-shift] = patch
        img[0:h_2,0:w_2] = patch

        cv2.rectangle(img, pt1_1, pt2_1, (0,  0,255 ), 3)  # cv2 里也是先x，后y，x轴沿水平方向向右，y轴沿垂直方向向下

        cv2.rectangle(img, pt1, pt2, (0,0,255), 3)  # cv2 里也是先x，后y，x轴沿水平方向向右，y轴沿垂直方向向下
        save_path_custom = save_path + rect
        cv2.imwrite(save_path_custom, img)

def draw_wdc_all():

    mothod = "qrnn3d"
    save_path = "/home/jiahua/liuy/hsi_pipeline/pic/"
    # rgb_path = "/home/jiahua/liuy/hsi_pipeline/figure/scene09_gt/"

    test_path = '/data1/jiahua/ly/wdc_pic_mat/wdc_all_mat/'
    
    save_path = save_path 
    test_path = test_path + mothod
     
    rect_list = os.listdir(test_path)
    result = np.zeros([3,256,256], dtype=np.float64)
    # cnt = 0

    for rect in rect_list:
        # rgb_path_custom = rgb_path+dataset+"\\rgb\\"
        # image_path =  rgb_path_custom + rect+".png"
    #     print(image_path)

        img = loadmat(os.path.join(test_path,rect))  #在cv2.imread参数中加入-1，表示不改变读取图像的类型直接读取
        img = img['output']


        result[0,:,:] = img[10,:,:]
        result[1,:,:] = img[43,:,:]
        result[2,:,:] = img[76,:,:]

    result = result.transpose((1,2,0))
    result = np.array(result*255,np.uint8)
    save_path = save_path + mothod
    cv2.imwrite(save_path + '_all.jpg', result)
    savemat(save_path + '_all.mat', {'output': result})

def montage2048_mat():

    base_path = '/home/jiahua/liuy/hsi_pipeline/cvpr_img/kaist/2048_rect/'
    new_path = '/home/jiahua/HSI-CVPR/hsi_pipeline/figure/kaist/'

    fns = os.listdir(base_path)

    result_gt = np.zeros([2048,2048,3], dtype=np.float64)
    for fn in fns:
        img = loadmat(os.path.join(base_path,fn))
        data = img['output_image'][...]
        if 'NW' in fn:
            result_gt[0:1024,0:1024,0] = data[:,:,9]
            result_gt[0:1024,0:1024,1] = data[:,:,19]
            result_gt[0:1024,0:1024,2] = data[:,:,29]
        elif 'NE' in fn:
            result_gt[0:1024,1024:2048,0] = data[:,:,9]
            result_gt[0:1024,1024:2048,1] = data[:,:,19]
            result_gt[0:1024,1024:2048,2] = data[:,:,29]
        elif 'SW' in fn:
            result_gt[1024:2048,0:1024,0] = data[:,:,9]
            result_gt[1024:2048,0:1024,1] = data[:,:,19]
            result_gt[1024:2048,0:1024,2] = data[:,:,29]
        elif 'SE' in fn:
            result_gt[1024:2048,1024:2048,0] = data[:,:,9]
            result_gt[1024:2048,1024:2048,1] = data[:,:,19]
            result_gt[1024:2048,1024:2048,2] = data[:,:,29]
    tmpdata =  "hsdt.png"
    result_gt = minmax_normalize(result_gt)
    result_gt = result_gt*255
    cv2.imwrite(os.path.join(new_path,tmpdata),result_gt)

def montage2048_rgb():

    base_path = '/home/jiahua/liuy/hsi_pipeline/cvpr_img/kaist/2048_rect/'
    new_path = '/home/jiahua/HSI-CVPR/hsi_pipeline/figure/kaist/'

    fns = os.listdir(base_path)

    result_gt = np.zeros([2048,2048,3], dtype=np.float64)
    for fn in fns:
        data = cv2.imread(os.path.join(base_path,fn))
        if 'NW' in fn:
            result_gt[0:1024,0:1024,:] = data
        elif 'NE' in fn:
            result_gt[0:1024,1024:2048,:] = data
        elif 'SW' in fn:
            result_gt[1024:2048,0:1024,:] = data
        elif 'SE' in fn:
            result_gt[1024:2048,1024:2048,:] = data
    tmpdata =  "hsdt.png"
    result_gt = minmax_normalize(result_gt)
    result_gt = result_gt*255
    cv2.imwrite(os.path.join(new_path,tmpdata),result_gt)
      

def getchannels():
    img = loadmat('/cvlabdata1/ly/hsi_data/kaist/ori_dataset/kaist_2048_complex/512_mixture/scene09_reflectance.mat')
    print(img.keys())
    data = img['input'][...]
    result_gt = np.zeros([2048,2048,3], dtype=np.float64)
    result_gt[:,:,0] = data[:,:,9]
    result_gt[:,:,1] = data[:,:,19]
    result_gt[:,:,2] = data[:,:,29]
    result_gt = result_gt * 255
    cv2.imwrite('/home/liuy/projects/hsi_pipeline/montage_imgs/final_input//scene09_mixture_g.png',result_gt[:,:,1])

def getnoise_only():
    gt = cv2.imread("/home/liuy/projects/hsi_pipeline/montage_imgs/final_input/scene09_11121_sp1.png")
    input = cv2.imread("/home/liuy/projects/hsi_pipeline/montage_imgs/final_input/scene09_mixture1123.png")
    noise = input - gt 
    cv2.imwrite("/home/liuy/projects/hsi_pipeline/montage_imgs/noisy_only_mixture/scene09_1123_mixtureonly.png",noise)

def make_india_mat():

    img = loadmat('/data1/jiahua/ly/india/india.mat')

    mothod = "grn_net"
    save_path = "/data1/jiahua/ly/india_all_method_mat/"

    test_path = '/data1/jiahua/ly/india_output_mat/'
    
    save_path = save_path 
    test_path = test_path + mothod
     
    rect_list = os.listdir(test_path)
    result = np.zeros([200,128,128], dtype=np.float64)
    # cnt = 0

    for rect in rect_list:

        img = loadmat(os.path.join(test_path,rect))  
        data = img['output']
        if rect == "0.mat":
            result[0:31:,:] = data
        elif rect == "31.mat":
            result[31:62,:,:] = data
        elif rect == "62.mat":
            result[62:93,:,:] = data
        elif rect == "93.mat":
            result[93:124,:,:] = data
        elif rect == "124.mat":
            result[124:155,:,:] = data
        elif rect == "155.mat":
            result[155:186,:,:] = data
        elif rect == "168.mat":
            result[186:200,:,:] = data[12:26,:,:]
    result = result.transpose((1,2,0))
    result = np.array(result*255,np.uint8)
    save_path = save_path + mothod
    # cv2.imwrite(save_path + '.jpg', result)
    savemat(save_path + '.mat', {'output': result})

def get_india_rgb():

    save_path = "/home/jiahua/liuy/hsi_pipeline/draw_pic/india/"
    test_path = '/data1/jiahua/ly/india_all_method_mat/'
     
    rect_list = os.listdir(test_path)

    for rect in rect_list:

        img = loadmat(os.path.join(test_path,rect))  
        print(img.keys())
        data = img['output'][...]
        result_gt = np.zeros([128,128,3], dtype=np.float64)
        # for i in range(31):
            # result_gt[:,:,1] = data[:,:,i]
        result_gt[:,:,0] = data[:,:,2]
        result_gt[:,:,1] = data[:,:,24]
        result_gt[:,:,2] = data[:,:,127]
        result_gt = minmax_normalize(result_gt)
        result_gt = result_gt * 255
        cv2.imwrite(save_path+rect[:-4]+'.png',result_gt)
        

def get_urban_rgb():

    # method = "fidnet"

    # save_path = "/home/jiahua/liuy/hsi_pipeline/draw_pic/urban/"
    # test_path = '/data1/jiahua/ly/urban_output_mat/'

    # test_path = test_path+method
     
    # rect_list = os.listdir(test_path)

    # for rect in rect_list:

    #     img = loadmat(os.path.join(test_path,rect))  
    #     # print(img.keys())
    #     data = img['output'][...]
    #     result_gt = np.zeros([256,256,3], dtype=np.float64)

    #     for rect in rect_list:

    #         img = loadmat(os.path.join(test_path,rect))  
    #         data = img['output']
            
    #         data = data.transpose((2,1,0))

    #         data = data.transpose((1,0,2))

    #         data = minmax_normalize(data)

    #         if rect == "0.mat":
    #             result_gt[:,:,0] = data[:,:,15]
    #         elif rect == "93.mat":
    #             result_gt[:,:,1] = data[:,:,12]
    #         elif rect == "179.mat":
    #             result_gt[:,:,2] = data[:,:,16]

    #     result_gt = result_gt * 255
    #     cv2.imwrite(save_path+ method + '.png',result_gt)

    save_path = "/home/jiahua/liuy/hsi_pipeline/draw_pic/urban/"
    test_path = '/data1/jiahua/ly/urban/'

    # test_path = test_path+method
     
    rect_list = os.listdir(test_path)

    for rect in rect_list:

        img = loadmat(os.path.join(test_path,rect))  
        print(img.keys())
        data = img['data'][...]
        # data = data.transpose((2,1,0))
        # data = data.transpose((1,0,2))

        result_gt = np.zeros([256,256,3], dtype=np.float64)

        tmp = data[:,:,0:31]
        tmp = minmax_normalize(tmp)
        result_gt[:,:,0] = tmp[:,:,15]
        tmp = data[:,:,93:124]
        tmp = minmax_normalize(tmp)
        result_gt[:,:,1] = tmp[:,:,12]
        tmp = data[:,:,179:210]
        tmp = minmax_normalize(tmp)
        result_gt[:,:,2] = tmp[:,:,16]

        result_gt = result_gt * 255
        # cv2.imwrite(save_path+rect[:-4]+'.png',result_gt)
        cv2.imwrite(save_path+'gt.png',result_gt)

def draw_cmp():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import ScalarFormatter
    from matplotlib.lines import Line2D
    import math

    n_points = 7
    psnr = [36.55,28.44,28.53,33.61,34.89,35.86,37.60,33.82,27.12]
    params = [0.86,41.44,0.43,0.83,4.10,1.91,0.05,0,0]
    flops = [2513.7,610.7,-1,-1,2082.4,1018.9,456.11,0,0]
    times = [0.720,0.466,2.709,0.758,2.265,0.872,1.03,4.1,4.8]
    labels = ['QRNN3D', 'GRNNET', 'MACNet', 'T3SC', 'SST','SERT','LRTDTV','LLRGTV' 'Ours']

    # 创建彩色圆圈标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#e377c2', '#9467bd', '#8c564b', 'red','c','c']
    # areas = [0.5,1,1.1,4,2.5,1.5,1.6]
    result = [86 , 2200 , 43 ,83 ,210 ,191 ,5 ,0 ,0]
    # result_refine = [ i * 50 for i in result]
            
    # 绘制散点图
    plt.figure(figsize=(6.75, 5))

    plt.scatter(times[:-3], psnr[:-3], s=result[:-3], c=colors[:-3], alpha=0.5)
    plt.scatter(times[-3], psnr[-3], s=result[-3], c=colors[-3])
    # plt.scatter(times[:-2], psnr[:-2], s=areas_in_sqft100[:-2], c=colors[:-2])
    plt.scatter(times[-2:], psnr[-2:],marker='^', c=colors[-2:])
    
    # for i, txt in enumerate(labels):
    #     plt.annotate(txt, (times[i], psnr[i]), fontsize=12, ha='center')
    
    # 创建圆圈图例
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', label='${-\ \ \ M}$', markerfacecolor='#403d39', markersize=10,alpha=0.7),
        Line2D([0], [0], marker='o', color='w', label='$0.1\ M$', markerfacecolor='#403d39', markersize=5,alpha=0.7),
        Line2D([0], [0], marker='o', color='w', label='$1\ \ \ \ M$', markerfacecolor='#403d39', markersize=9,alpha=0.7),
        Line2D([0], [0], marker='o', color='w', label='$5\ \ \ M$', markerfacecolor='#403d39', markersize=15,alpha=0.7)
    ]

    # 添加图例
    plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper right')
    # plt.legend(handles=legend_elements, loc='upper right')

    
    ax = plt.gca()
    ax.set_yticks([28,29,30,31,32,33,34,35,36,37,37.5,38])  # 设置标签的位置
    ax.set_xticks([0.1,0.6,1.0,1.5,2.0,2.5,3,3.5,4.25,5])  # 设置标签的位置
    ax.set_xticklabels([0, 0.6, 1.0, 1.5,2.0,2.5,5, '${10^{2}}$','${1.5×10^{2}}$','${2×10^{2}}$'])  # 设置标签的文本
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
        
    plt.grid(color='gray', linestyle='dashdot', linewidth=1)   

    # 添加轴标题
    # plt.xlabel(r'$\mathbf{Running\ time\ (sec)}$', fontsize=12)
    # plt.ylabel(r'$\mathbf{PSNR\ (dB)}$', fontsize=12)
    # plt.title(r'$\mathbf{FPS VS Times}$', fontsize=12)


    # plt.ylim(66, 82)
    # plt.xlim(0, 120)

    # 显示图
    plt.savefig('test.png')

def draw_cmp_fig():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import ScalarFormatter
    from matplotlib.lines import Line2D
    import math

    n_points = 7
    psnr = [36.55,28.44,28.53,33.61,34.89,35.86,35.98,37.65,39.03,33.82,27.12,24.59,23.20]
    params = [0.86,41.44,0.43,0.83,4.10,1.91,14.28,0.05,0.21,0,0,0,0]
    times = [40.055535,2.657156205,4.945263171,20.73094599,35.72392147,18.69055651,75.58,7.57,13.06,193.3764,101.5178,171.08,807.07]
    times_cpu = [40.06,2.66,4.95,20.73,35.72,18.69,75.58,7.57,13.06,97 ,86,101.08,127.07] # ,171.08,807.07
    labels = ['QRNN3D', 'GRNNET', 'MACNet', 'T3SC', 'SST','SERT','GRUNET', 'Ours','Ours-L','LRTDTV','LLRGTV','NGMEET','LLRT']

    # 创建彩色圆圈标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#e377c2', '#9467bd', '#8c564b','#ACD24E', 'red','red','c','c','c','c']
    # areas = [0.5,1,1.1,4,2.5,1.5,1.6]
    result = [86 , 2200 , 43 ,83 ,210 ,191,1100 ,5,10,0,0,0,0]
    result_refine = [ 5 for i in result]
            
    # 绘制散点图
    plt.figure(figsize=(6.75, 4),dpi=400)

    plt.scatter(times_cpu[:-5], psnr[:-5], s=result[:-5], c=colors[:-5], alpha=0.5)
    plt.scatter(times_cpu[:-5], psnr[:-5], s=result_refine[:-5], c=colors[:-5],alpha=0.6)
    plt.scatter(times_cpu[-5], psnr[-5], s=result[-5], c=colors[-5])
    # plt.scatter(times[:-2], psnr[:-2], s=areas_in_sqft100[:-2], c=colors[:-2])
    plt.scatter(times_cpu[-4:], psnr[-4:],marker='^', c=colors[-4:])
    
    # for i, txt in enumerate(labels):
    #     plt.annotate(txt, (times[i], psnr[i]), fontsize=12, ha='center')
    
    # 创建圆圈图例
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', label='${-\ \ \ M}$', markerfacecolor='#403d39', markersize=10,alpha=0.7),
        Line2D([0], [0], marker='o', color='w', label='$0.1\ M$', markerfacecolor='#403d39', markersize=5,alpha=0.7),
        Line2D([0], [0], marker='o', color='w', label='$1\ \ \ \ M$', markerfacecolor='#403d39', markersize=9,alpha=0.7),
        Line2D([0], [0], marker='o', color='w', label='$5\ \ \ M$', markerfacecolor='#403d39', markersize=15,alpha=0.7)
    ]

    # 添加图例
    plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper right')
    # plt.legend(handles=legend_elements, loc='upper right')

    
    ax = plt.gca()
    ax.set_yticks([29,30,31,32,33,34,35,36,37,38,39])  # 设置标签的位置
    ax.set_xticks([0,10,20,30,40,50,60,70,80,100])  # 设置标签的位置
    ax.set_xticklabels([1, 10, 20,30,40,50,60,70,80,200])  # 设置标签的文本

    x1_label = ax.get_xticklabels() 
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels() 
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
        
    # plt.grid(color='gray', linestyle='dashdot', linewidth=1)   

    # 添加轴标题
    # plt.xlabel(r'$\mathbf{Running\ time\ (sec)}$', fontsize=12)
    # plt.ylabel(r'$\mathbf{PSNR\ (dB)}$', fontsize=12)
    # plt.title(r'$\mathbf{FPS VS Times}$', fontsize=12)


    # plt.ylim(66, 82)
    # plt.xlim(0, 120)

    # 显示图
    plt.savefig('test_r.png')

def draw_cmp_fig2():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import ScalarFormatter
    from matplotlib.lines import Line2D
    import math

    psnr = [36.55,28.44,34.89,35.86,37.03]
    times = [0.720,0.466,2.709,0.758,2.265,0.872,1.03,4.1,4.8]
    flops = [2513.7,610.7,2082.4,1018.9,957.0]
    labels = ['QRNN3D', 'GRNNET', 'SST','SERT','SACT']

    times2 = [97.65,171.72,86.03,807.21,18.92]
    psnr2 = [33.82,27.12,23.20,24.59,37.03]
    labels2 = ['LRTDTV', 'LLRGTV', 'NGMeet','LLRT','SACT']      
            
    # 绘制散点图
    plt.figure(figsize=(6.75, 5))

    plt.grid(color='gray', linestyle='dashdot', linewidth=1)   
    ax = plt.gca()
    
    # plt.scatter(times2[-1], psnr2[-1], c='red')
    # plt.scatter(times2[:-1], psnr2[:-1],c='black')
    # ax.set_yticks([22,24,26,28,30,32,34,36,38])  # 设置标签的位置
    # ax.set_xticks([10,100,200,400,600,800,1000])  # 设置标签的位置
    

    flops = [ i - 500 for i in flops]
    plt.scatter(flops[-1], psnr[-1], c='red')
    plt.scatter(flops[:-1], psnr[:-1],c='black')
    ax.set_yticks([28,29,30,31,32,33,34,35,36,37,38])  # 设置标签的位置
    ax.set_xticks([0,200,400,600,1000,1500,2000,2600])  # 设置标签的位置
    
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
        


    # 显示图
    
    plt.savefig('test2.png',dpi=1200)

# CVPR 时序图

def draw_2():
    import random
    img = loadmat("/data1/jiahua/ly/test_data/wdc_complex/512_mixture/wdc.mat")
    print(img.keys())
    data = img['gt'][...]
    # for i in range(10):
    #     spectral.save_rgb("/home/jiahua/HSI-CVPR/hsi_pipeline/result/wdc/sequential/"+str(i)+".png",data,bands=(19*i,19*i+9,19*i+18))
    normalize_img = minmax_normalize(data)
    # normalize_img = data * 255
    point_1 = normalize_img[162,87,:150]
    
    point_2_1 = normalize_img[12,150,:150]
    point_2_2 = normalize_img[25,158,:150] 
    
    point_7_1 = normalize_img[194,38,:150]
    point_7_2 = normalize_img[222,58,:150] 
    
    point_3 = normalize_img[34,118,:150]
    point_4 = normalize_img[115,34,:150]
    
    point_5 = normalize_img[116,1,:150]
    point_6 = normalize_img[150,152,:150]
    
    # max_index = np.unravel_index(np.argmax(normalize_img), normalize_img.shape)
    # print(max_index)
    
    Bands = np.arange(150)
    # 创建一个折线图
    plt.figure(figsize=(8, 2),dpi=400)
    plt.rcParams['font.sans-serif'] = 'times new roman'
    
    plt.plot(Bands, point_7_1 , label='Position C' , color='#FF0000' , marker = 'o' ,markersize=1)
    plt.plot(Bands, point_7_2 , label='Position D' , color='#0070C0' , marker = 'o' ,markersize=1)
    from matplotlib.font_manager import FontProperties
    font = FontProperties()
    font.set_weight('bold')

    # 添加图例，并设置字体大小和加粗
    # plt.legend()
    # plt.plot(Bands, point_3 , label='point_3')

    ax = plt.gca()
    x1_label = ax.get_xticklabels() 
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels() 
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

    
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    # xticks = [10, 20, 30]
    # plt.xticks(xticks)
    
    # plt.ylim(0, 1)
    # 添加标题和标签
    # plt.xlabel(r'$\mathbf{Band}$')
    plt.ylabel(r'$\mathbf{Normalize\ pixel}$')
    plt.ylabel(r'$\mathbf{Normalize\ pixel}$')
    
    plt.legend(loc='upper right',fontsize='32', prop=font)

    # 保存图片为PNG文件
    plt.savefig('/home/jiahua/HSI-CVPR/hsi_pipeline/result/wdc/point_bands/agg_2.png')
    
    return 0
    
def get_real_rgb():

    # img = loadmat("")
    # print(img.keys())
    # data = img['data'][...]
    # print(data.shape)
    # spectral.save_rgb("/home/jiahua/HSI-CVPR/hsi_pipeline/result/urban/s2s.png",data,bands=(102, 138, 202))

    image_A = cv2.imread('/home/jiahua/HSI-CVPR/hsi_pipeline/result/urban/noisy.png')
    image_B = cv2.imread('/home/jiahua/HSI-CVPR/hsi_pipeline/result/urban/s2s.png')

    print(image_A.shape)
    w, h ,c = image_A.shape

    result_gt = np.zeros([224,224,3], dtype=np.float64)

    for i in range(w):
        result_gt[i,0:w-i] = image_A[i,0:w-i]
        result_gt[i,w-i:w] = image_B[i,w-i:w]

    cv2.imwrite('/home/jiahua/HSI-CVPR/hsi_pipeline/result/urban/fake.png',result_gt)


def draw_flops():
    # 样本数据
    flops = [912.2,1018.9,2082.4,3095.86,610.7,2513.7]
    # flops = flops[::-1]
    psnr = [37.65,35.86,34.89,35.98,28.44,36.55]
    # psnr = psnr[::-1]
    psnr_refine = [(i *100) for i in psnr] 
    psnr_refine1 = [ (i+(i-30)) for i in psnr] 
    flops_norm = normalize_list(flops)
    flops_refine = [ (i * 10)-5 for i in flops_norm] 
    print(psnr_refine)
    # y轴刻度的标签
    plt.figure(figsize=(8, 2))
    labels = ['RAS2S', 'SERT', 'SST', 'GRUNet', 'GRNet', 'QRNN3D']
    # labels = labels[::-1]
    ax = plt.gca()

    # 创建一个水平柱状图
    # plt.barh(labels, psnr_refine1,color='#5B8FA9',height=0.5)
    # plt.barh(labels, flops,color='#ACD24E',height=0.5)
    psnr_refine[0] = psnr_refine[0]+5
    plt.bar(labels, psnr_refine,color='#5B8FA9')
    plt.bar(labels, flops,color='#ACD24E')
    plt.yticks([])
    # plt.xticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)



    # 可以设置其他参数，如颜色、标题、坐标轴标签等

    # 设置x轴标签
    # plt.xlabel('Values')

    # 设置图的标题
    # plt.title('Horizontal Bar Chart')

    # 显示图
    plt.savefig('flops.png')



if __name__ == '__main__':
    # montage2048_rgb()
    # drawRect()
    # drawWdcRect()
    # draw_wdc_cut()
    # draw_wdc_all()
    # getnoise_only()
    # getchannels()
    # make_india_mat()
    # get_india_rgb()
    # draw_india_rgb()
    # draw_wdc_cut()
    # get_urban_rgb()
    # drawRect()
    # draw_india_rect()
    # drawicvlRect()
    # drawcaveRect()
    # draw_cmp_fig()
    # draw_cmp_fig2()
    make_erf()
    # draw_2()
    # get_real_rgb()
    # draw_flops()

    # img = loadmat("/data1/jiahua/result/urban/hsdt_s.mat")
    # print(img.keys())
    # data = img['output'][...]
    # data = data.transpose((2,1,0))
    # data = data.transpose((1,0,2))
    # print(data.shape)
    # spectral.save_rgb("/home/jiahua/HSI-CVPR/hsi_pipeline/figure/urban/hsdt.png",data,bands=(102,138 , 20))


    # img = loadmat('/data1/jiahua/result/wdc/wdc_LLRGTV.mat')
    # print(img.keys())
    # data = img['output_image'][...]
    # savemat('/data1/jiahua/result/wdc/wdc_LLRGTV.mat', {'data': data})
    # test_path= "/home/jiahua/HSI-CVPR/hsi_pipeline/figure/wdc_res/"
     
    # rect_list = os.listdir(test_path)

    # for rect in rect_list:
    #     data = cv2.imread(test_path+rect)
    #     data = cv2.resize(data,(256,256))
    #     cv2.imwrite(test_path+rect,data)
        # print(data.shape)
    
    # savemat('/data1/jiahua/result/urban/LRTDTV.mat', {'data': data})
    
    # data = img['gt'][...]
    # result_gt = np.zeros([2048,2048,3], dtype=np.float64)
    # result_gt[:,:,0] = data[:,:,0]
    # # result_gt[:,:,0] = data[:,:,9]
    # # result_gt[:,:,1] = data[:,:,19]
    # # result_gt[:,:,2] = data[:,:,29]
    # result_gt = result_gt * 255
    # cv2.imwrite('/home/liuy/projects/hsi_pipeline/montage_imgs/09_test/scene09_reflectance_gt_0.png',result_gt[:,:,0])
    # cv2.imwrite('/home/liuy/projects/hsi_pipeline/montage_imgs/09_test/scene09_reflectance_mixture_9.png',result_gt[:,:,0])
    # cv2.imwrite('/home/liuy/projects/hsi_pipeline/montage_imgs/09_test/scene09_reflectance_mixture_19.png',result_gt[:,:,1])
    # cv2.imwrite('/home/liuy/projects/hsi_pipeline/montage_imgs/09_test/scene09_reflectance_mixture_29.png',result_gt[:,:,2])



    # img = loadmat('/cvlabdata1/ly/hsi_data/kaist/ori_dataset/kaist_2048_complex/512_stripe/scene09_reflectance.mat')
    # print(img.keys())
    # data = img['input'][...]
    # result_gt = np.zeros([2048,2048,3], dtype=np.float64)
    # for i in range(31):
    #     result_gt[:,:,1] = data[:,:,i]
    # # result_gt[:,:,0] = data[:,:,5]
    # # result_gt[:,:,1] = data[:,:,15]
    # # result_gt[:,:,2] = data[:,:,20]
    #     result_gt = result_gt * 255
    #     cv2.imwrite('/home/liuy/projects/hsi_pipeline/montage_imgs/noisy_sp/scene09_noise_'+str(i)+'.png',result_gt)

    
    # img = loadmat('/data1/jiahua/test_data/indian_mat/indian_pines.mat')
    # print(img.keys())
    # data = img['pavia'][...]
    # data = data[0:128,0:128,:]
    # savemat("/data1/jiahua/ly/india/india1.mat",{'data':data})
    # # data = data.transpose((2,1,0))
    # result_gt = np.zeros([128,128,3], dtype=np.float64)
    # # for i in range(31):
    # #     result_gt[:,:,1] = data[:,:,i]
    # test1 = data[:,:,0:30]
    # test2 = data[:,:,124:155]
    # test1 = minmax_normalize(test1)
    # test2 = minmax_normalize(test2)
    # result_gt[:,:,0] = test1[:,:,2]
    # result_gt[:,:,1] = test1[:,:,24]
    # result_gt[:,:,2] = test2[:,:,3]
    # result_gt = result_gt * 255
    # # result_gt = np.rot90(result_gt)
    # # result_gt = result_gt[::-1]
    # cv2.imwrite('/home/jiahua/liuy/hsi_pipeline/draw_pic/india/india.png',result_gt)


    # data = loadmat("/data1/jiahua/ly/cave_test_complex/512_mixture/chart_and_stuffed_toy_ms.mat")
    # data = data['input']
    # result_gt = np.zeros([512,512,3], dtype=np.float64)
    # # for i in range(31):
    # #     result_gt[:,:,1] = inputs[:,:,i]
    # result_gt[:,:,0] = data[:,:,9]
    # result_gt[:,:,1] = data[:,:,19]
    # result_gt[:,:,2] = data[:,:,29]
    # result_gt = result_gt * 255
    # cv2.imwrite(os.path.join('/home/jiahua/liuy/hsi_pipeline/draw_pic/cave','noisy.png'),result_gt)
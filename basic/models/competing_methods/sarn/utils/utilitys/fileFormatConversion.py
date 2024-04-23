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
#将tif文件改成mat
import scipy
from skimage import img_as_float, img_as_ubyte
import pickle as pl
from scipy.io import loadmat
import h5py
from utility import minmax_normalize


def mat2npy(inpath,outpath,name):
    matr =scipy.io.loadmat(inpath)
    print(matr.keys())
    data = matr[name]
    print(data.shape)
    numpy_data = data.transpose((1,2,3,0))
    print(numpy_data.shape)
    # 保存为numpy数组文件（.npy文件）io.savemat(outpath + name  + '.mat', {name: result}, '-v7.3')
    io.savemat(outpath, {name: numpy_data})
    #np.save(outpath, numpy_data)

# inpath="/home/xx/Codes/Python/VDNet/data3d/icvl_test.mat"
# outpath="/home/xx/Codes/Python/VDNet/data3d/icvl_test1.mat"
# name="icvl_test"
# mat2npy(inpath,outpath,name)

def dif2mat(inpath,outpath,name):
    # to open a tiff file for reading:
    dc = skimage.io.imread(inpath+name+".tif")
    print(type(dc[0,0,0]))
    print(sys.getsizeof(dc[0,0,0]))
    sio.savemat(outpath+name+".mat", {'dc': dc})
    print('success')


def selectLmdb(path,result_path):
    env1 = lmdb.open(path)
    #print(env.map_size)
    #map_size=209715200  200G
    print(env1.stat())  # 状态
    txn1 = env1.begin()
    database1 = txn1.cursor()
    #将数据保存在另外一个数据集中
    env0=lmdb.open(result_path)
    env0.set_mapsize(1099511627776)  # 扩大映射范围，才可以追加 1T
    txn=env0.begin(write=True)
    count = 0
    k=0
    # 遍历数据库
    for (key, value) in database1:
        print(key)
        str_id = '{:08}'.format(k)
        k += 1
        # txn.put(str_id.encode('ascii'), datum.SerializeToString())
        txn.put(str_id.encode('ascii'), value)
        count += 1
        if (count % 1000 == 0):
            txn.commit()
            txn = env0.begin(write=True)
        if(count==4896):
            txn.commit()
            count = 0
            break
    # if (count % 1000 != 0):
    #     txn.commit()
        #count=0
    txn = env0.begin(write=True)
    print(txn.stat()['entries'])
    count=0

    if (count % 1000 != 0):
        txn.commit()
        count = 0
        txn = env0.begin(write=True)
        print(txn.stat()['entries'])

    # 查询合并前后的lmdb的数据，确认合并成功
    print(txn1.stat()['entries'])
    print(txn.stat()['entries'])
    # 关闭lmdb
    env0.close()
    env1.close()
    print("success")

# train="/home/hcx/xx/VDNet/DataSet/train_db/TrainDataCave.db"
# result="/home/hcx/xx/VDNet/DataSet/train_db/TrainDataCaveSelected.db"
#
#
# selectLmdb(train,result)
# env1=lmdb.open(result)
# txn=env1.begin()
# print(txn.stat()['entries'])


#合并多个lmdb数据库
def mergeLmdb(path1,path2,result_path):
    env1 = lmdb.open(path1)
    env2 = lmdb.open(path2)
    #print(env.map_size)
    #map_size=209715200  200G
    print(env1.stat())  # 状态
    print(env2.stat())
    txn1 = env1.begin()
    txn2 = env2.begin()
    database1 = txn1.cursor()
    database2 = txn2.cursor()
    #将数据保存在另外一个数据集中
    env0=lmdb.open(result_path)
    env0.set_mapsize(1099511627776)  # 扩大映射范围，才可以追加 1T
    txn=env0.begin(write=True)
    count = 0
    k=0
    # 遍历数据库
    for (key, value) in database2:
        print(key)
        str_id = '{:08}'.format(k)
        k += 1
        # txn.put(str_id.encode('ascii'), datum.SerializeToString())
        txn.put(str_id.encode('ascii'), value)
        count += 1
        if (count % 1000 == 0):
            txn.commit()
            count = 0
            txn = env0.begin(write=True)
    if (count % 1000 != 0):
        txn.commit()
        #count=0
    txn = env0.begin(write=True)
    print(txn.stat()['entries'])
    count=0
    for (key, value) in database1:
        print(key)
        str_id = '{:08}'.format(k)
        k += 1
        # txn.put(str_id.encode('ascii'), datum.SerializeToString())
        txn.put(str_id.encode('ascii'), value)
        count+=1
        if (count % 1000 == 0):
            # 将数据写入数据库，必须的，否则数据不会写入到数据库中
            txn.commit()
            count = 0
            txn = env0.begin(write=True)
            # print(txn.stat()['entries'])

    if (count % 1000 != 0):
        txn.commit()
        count = 0
        txn = env0.begin(write=True)
        print(txn.stat()['entries'])

    # 查询合并前后的lmdb的数据，确认合并成功
    print(txn1.stat()['entries'])
    print(txn2.stat()['entries'])
    print(txn.stat()['entries'])
    # 关闭lmdb
    env0.close()
    env1.close()
    env2.close()

    # for (key, value) in database1:
    #     database.put(key, value)
    # for (key, value) in database2:
    #     database.put(key, value)
    #print(env.map_size)
    print("success")

# PaviaC="/home/xx/Codes/Python/VDNet/DataSet/train_db/TrainDataLast1.db"
# PaviaU="/home/xx/Codes/Python/VDNet/DataSet/train_db/TrainDataLast9.db"
# train_database="/home/xx/Codes/Python/VDNet/DataSet/train_db/TrainDataLast2.db"
#
# #mergeLmdb(PaviaC,PaviaU,train_database)
# env1=lmdb.open(train_database)
# txn=env1.begin()
# print(txn.stat()['entries'])

def insert(env, sid, name):
	txn = env.begin(write = True)
	txn.put(str(sid).encode(), name.encode())
	txn.commit()

#从多个不同的数据集中各取部分数据来生成小训练集
def createMiNiTrainDATA(path1,path2,result_path):
    env1 = lmdb.open(path1)
    env2 = lmdb.open(path2)
    # print(env.map_size)
    # map_size=209715200  200G
    print(env1.stat())  # 状态
    print(env2.stat())
    txn1 = env1.begin()
    txn2 = env2.begin()
    database1 = txn1.cursor()
    database2 = txn2.cursor()
    # 将数据保存在另外一个数据集中
    env0 = lmdb.open(result_path)
    env0.set_mapsize(1099511627776)  # 扩大映射范围，才可以追加 1T
    txn = env0.begin(write=True)
    k=0
    count = 0
    # 遍历数据库
    for (key, value) in database2:
        print(key)
        str_id = '{:08}'.format(k)
        k += 1
        #txn.put(str_id.encode('ascii'), datum.SerializeToString())
        txn.put(str_id.encode('ascii'), value)
        count += 1
        if (count==5):
            txn.commit()
            count = 0
            break
    if (count % 1000 != 0):
        txn.commit()
        # count=0
    txn = env0.begin(write=True)
    print(txn.stat()['entries'])
    count = 0
    for (key, value) in database1:
        print(key)
        str_id = '{:08}'.format(k)
        k += 1
        #txn.put(str_id.encode('ascii'), datum.SerializeToString())
        txn.put(str_id.encode('ascii'), value)
       # txn.put(key, value)
        count += 1
        if (count==5):
            # 将数据写入数据库，必须的，否则数据不会写入到数据库中
            txn.commit()
            count = 0
            break;
    txn = env0.begin(write=True)
    print(txn.stat()['entries'])
    print(k)
    # 关闭lmdb
    env0.close()
    env1.close()
    env2.close()

#从单个数据集中取部分数据来生成小训练集
def createMiNiTrainDATA1(path1,result_path):
    env1 = lmdb.open(path1)
    # print(env.map_size)
    # map_size=209715200  200G
    print(env1.stat())  # 状态
    txn1 = env1.begin()
    database1 = txn1.cursor()
    # 将数据保存在另外一个数据集中
    env0 = lmdb.open(result_path)
    env0.set_mapsize(1099511627776)  # 扩大映射范围，才可以追加 1T
    txn = env0.begin(write=True)
    k=0
    count = 0
    for (key, value) in database1:
        print(key)
        str_id = '{:08}'.format(k)
        k += 1
        #txn.put(str_id.encode('ascii'), datum.SerializeToString())
        txn.put(str_id.encode('ascii'), value)
       # txn.put(key, value)
        count += 1
        if (count==1):
            # 将数据写入数据库，必须的，否则数据不会写入到数据库中
            txn.commit()
            count = 0
            break
    txn = env0.begin(write=True)
    print(txn.stat()['entries'])
    print(k)
    # 关闭lmdb
    env0.close()
    env1.close()


def createMiNiDB(path1,path2):
    env1 = lmdb.open(path1)
    env2 = lmdb.open(path2)
    for i in range(1500):
        insert(env1, i, "xuxiang")
    for i in range(3500):
        insert(env2, i + 1500, "haha")
    print(env1.begin().stat()['entries'])
    print(env2.begin())

#将多张二维图像转换成三维mat格式数据
def img2dtoimg3d(inpath,outpath,name,k):
    dir = Path(inpath)
    train_im_list = list(dir.glob('*.jpg')) + list(dir.glob('*.png')) + \
                                                                    list(dir.glob('*.bmp'))
    im_list = sorted([str(x) for x in train_im_list])
    print(im_list)
    #以原通道读取
    im_ori= cv2.imread(im_list[0],-1)[:, :]
    data= img_as_float(im_ori)
    print(data)
    data=np.reshape(data,(1,512,512))
    print(data.shape)
    print(data)
    for i in range(1,len(im_list)):
        im_ori = cv2.imread(im_list[i],-1)[:, :]
        im_gt = img_as_float(im_ori)
        im_gt= np.reshape(im_gt, (1, 512, 512))
        data=np.vstack((data,im_gt))
    print(data.shape)
    io.savemat(outpath+name+str(k)+'.mat', {name: data},'-v7.3')#以h5的yle save

def db2mat(inputpath,outpath,name):
    env0 = lmdb.open(inputpath)
    txn = env0.begin()
    database = txn.cursor()
    result_list=[]
    for (key, value) in database:
        print(key)
        print(value.hape)
        result_list.append(value)
    result=np.array(result_list)
    env0.close()
    io.savemat(outpath + name  + '.mat', {name: result}, '-v7.3')

# inpath="/home/xx/Codes/Python/VDNet/DataSet/IcvlL.db"
# outpath="/home/xx/Codes/Python/QRNN3D/data3d/"
# name="icvl_test"
# db2mat(inpath,outpath,name)


def mat2db(datadir,matkey,fns,name,load=h5py.File):
    if os.path.exists(name+'.db'):
        raise Exception('database already exist!')
    #若没有name这个文件或者文件夹，则生成一个name.db的文件
    env = lmdb.open(name+'.db', map_size=209715200, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        #注意这里的键值  不仅影响数据的合并，并且影响后面数据加载器读取数据
        k =0
        for i, fn in enumerate(fns):
            try:
                X = load(datadir + fn)[matkey]
                X = X.transpose((2, 0, 1))
                X = X[np.newaxis,:,:,:]
            except:
                print('loading', datadir+fn, 'fail')
                continue
            print(X.dtype)
            X = X.astype(np.float32)  # Cave/ICVL/paviaU数据集需要将原本的float64转换成float32
            #X = minmax_normalize(X)
            # X = crop_center(X, crop_sizes[0], crop_sizes[1])
            # print("旋转之前{0}".format(X.shape))
            #X = rotAugmentation(X)
            # print("旋转之后{0}".format(X.shape))

            N = X.shape[0]
            print(X.shape)
            print("N的值{0}".format(N))
            for j in range(N):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = X.shape[1]
                datum.height = X.shape[2]
                datum.width = X.shape[3]
                datum.data = X[j].tobytes()
                str_id = '{:08}'.format(k)
                k += 1
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
            print('load mat (%d/%d): %s' % (i, len(fns), fn))
        print("K的值{0}".format(k))
        print('done')

# datadir='/home/xx/Codes/Python/VDNet/data3d/toy_center/'
# fns = os.listdir(datadir)
# fns = [fn.split('.')[0]+'.mat' for fn in fns]
# matkey='toy'
# outPath='/home/xx/Codes/Python/VDNet/DataSet/toy_center'
# mat2db(datadir,matkey,fns,outPath,load=loadmat)

# inpath="/home/xx/Codes/Python/VDNet/data2d/thread_spools_ms"
# outpath="/home/xx/Codes/Python/QRNN3D/data3d/Cave/"
# name="train"
# img2dtoimg3d(inpath,outpath,name,15)


# test1="/home/hcx/xx/VDNet/DataSet/test1.db"
# test2="/home/hcx/xx/VDNet/DataSet/test2.db"
# # createMiNiDB(test1,test2)
# train_database="/home/hcx/xx/VDNet/DataSet/result1.db"
# mergeLmdb(test1,test2,train_database)


#WTDC和PaviaU这两个数据集合并到TrainData.db文件中
# PaviaC="/home/hcx/xx/VDNet/DataSet/PaviaCentre.db"
# PaviaU="/home/hcx/xx/VDNet/DataSet/PaviaU.db"
# train_database="/home/hcx/xx/VDNet/DataSet/train_db/testTrainData.db"
# createMiNiTrainDATA(PaviaC,PaviaU,train_database)
# #mergeLmdb(PaviaC,PaviaU,train_database)
# env1=lmdb.open(train_database)
# txn=env1.begin()
# print(txn.stat()['entries'])

# PaviaU='/home/xx/Codes/Python/VDNet/DataSet/IcvlL.db'
# train_database='/home/xx/Codes/Python/VDNet/DataSet/test1.db'
# createMiNiTrainDATA1(PaviaU,train_database)
# train_path = "/home/xx/Codes/Python/VDNet/DataSet/TrainLast.db"
# env1=lmdb.open(train_path)
# txn=env1.begin()
# print(txn.stat()['entries'])

#生成dc.mat文件
# inpath="/home/hcx/xx/VDNet/data3d/Washington DC Mall/"
# name="dc"
# outpath="/home/hcx/xx/VDNet/data3d/Washington DC Mall/"
# dif2mat(inpath,outpath,name)

# inpath="/home/hcx/xx/VDNet/data3d/Washington DC Mall/dc.mat"
# name="dc"
# outpath="/home/hcx/xx/VDNet/data3d/Washington DC Mall/dc.npy"
# mat2npy(inpath,outpath,name)
#
# data = skimage.io.imread("/home/hcx/xx/VDNet/data3d/Washington DC Mall/dc.tif")
# print(data.dtype)
# print(np.min(data))
# print(np.max(data))
#
# inpath="/home/xx/Codes/Python/VDNet/data3d/Icvl/4cam_0411-1640-1.mat"
# path="/home/hcx/xx/VDNet/data3d/PaviaU/PaviaU.mat"
#
# matr = h5py.File(inpath)
# print(matr.keys())
# data = matr["train"]
#print(data.shape)

# #中心裁剪
def crop_center(img,cropx,cropy):
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx+50:startx+cropx+50, :]

#rocess *,mat
#inpath='/home/xx/Codes/Python/VDNet/data3d/Cave/train1.mat'
#inpath = '/home/xx/VDnet_result/Denoise_Result/WDC/3DVSSN/30.mat'
#outpath ='/home/xx/Codes/Python/VDNet/data3d/IndianPines_145.mat'
# inpath='/home/xx/Codes/Python/VDNet/data3d/Washington DC Mall/dc.mat'
# outpath ='/home/xx/Codes/Python/VDNet/data3d/dc_test_128_last.mat'
#
#inpath = '/home/xx/Codes/Python/VDNet/data3d/Toy/toy.mat'
inpath  = '/home/xx/VDnet_result/Denoise_Result/WDC/3DVSSN/stripe.mat'
outpath ='/home/xx/Codes/Python/VDNet/data3d/toy_center.mat'
matr = scipy.io.loadmat(inpath)
print(matr.keys())
data = matr['dc']
print(data.shape)
# data =  data.transpose((1,2,0)).astype(np.float32)
# data = minmax_normalize(data)
# show=[10, 20, 30]
# cv2.imwrite("/home/xx/VDnet_result/CAVE/noise"+str(0)+".png", img_as_ubyte(minmax_normalize(data[:, :, show])))
# # # print(data.shape)
# data=crop_center(data,256,256)
# print(data.shape)
# scipy.io.savemat(outpath, {'toy': data})
# show=[10, 20, 30]
# cv2.imwrite("/home/xx/VDnet_result/CAVE/noise"+str(1)+".png", img_as_ubyte(minmax_normalize(data[:, :, show])))
# print(data.shape)

# datadir='/home/xx/VDnet_result/Denoise_Result/WDC/FastHyDe/'
# outpath='/home/xx/VDnet_result/Denoise_Result/WDCM/FastHyDe/'
# fns = os.listdir(datadir)
# fns = [fn.split('.')[0]+'.mat' for fn in fns]
# for i, fn in enumerate(fns):
#     try:
#         # matr = scipy.io.loadmat(datadir+fn)
#         # print(matr.keys())
#         data = loadmat(datadir + fn)['dc']
#         # data = data.transpose((1,2,0)).astype(np.float32)
#         # data = crop_center(data, 256, 256)
#         data = minmax_normalize(data)
#         print(data.shape)
#         scipy.io.savemat(outpath+ fn, {'dc': data})
#     except:
#         print('loading', datadir + fn, 'fail')
#         continue


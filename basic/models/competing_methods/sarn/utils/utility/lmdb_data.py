"""Create lmdb dataset"""
from util import *
import lmdb
import caffe
import numpy as np
from dataset import *


def create_lmdb_train(
    datadir, fns, name, matkey,
    crop_sizes, scales, ksizes, strides,
    load=h5py.File, augment=True,
    seed=2017):
    """
    Create Augmented Dataset
    """
    def preprocess(data):
        new_data = []
        # data = minmax_normalize(data)
        # data = np.rot90(data, k=2, axes=(1,2)) # ICVL
        data = minmax_normalize(data.transpose((2,0,1))) # for Remote Sensing
        # Visualize3D(data)
        if crop_sizes is not None:
            data = crop_center(data, crop_sizes[0], crop_sizes[1])        
        
        for i in range(len(scales)):
            if scales[i] != 1:
                temp = zoom(data, zoom=(1, scales[i], scales[i]))
            else:
                temp = data
            temp = Data2Volume(temp, ksizes=ksizes, strides=list(strides[i]))            
            new_data.append(temp)
        new_data = np.concatenate(new_data, axis=0)
        print(new_data.shape)
        if augment:
             for i in range(new_data.shape[0]):
                 for j in range(8):
                    new_data[i,...] = data_augmentation(new_data[i, ...], mode= j )
                
        return new_data.astype(np.float32)

    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)        
    assert len(scales) == len(strides)
    # calculate the shape of dataset
    data = load(datadir + fns[0])[matkey]
    print(data.shape)
    data = preprocess(data)
    N = data.shape[0]
    print(N)
    
    print(data.shape)
    map_size = data.nbytes * len(fns) * 1.2
    print('map size (GB):', map_size / 1024 / 1024 / 1024)
    
    # import ipdb; ipdb.set_trace()
    if os.path.exists(name+'.db'):
        raise Exception('database already exist!')
    env = lmdb.open(name+'.db', map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        k = 0
        for i, fn in enumerate(fns):
            try:
                X = load(datadir + fn)[matkey]
            except:
                print('loading', datadir+fn, 'fail')
                continue
            X = preprocess(X)        
            N = X.shape[0]
            for j in range(N):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = X.shape[1]
                datum.height = X.shape[2]
                datum.width = X.shape[3]
                datum.data = X[j].tobytes()
                str_id = '{:08}'.format(k)
                k += 1
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
            print('load mat (%d/%d): %s' %(i,len(fns),fn))

        print('done')
def create_lmdb_test_center(
        datadir, fns, name, matkey,
        crop_sizes, scales, ksizes, strides,
        load=h5py.File, augment=True,
        seed=2017):

    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)
    assert len(scales) == len(strides)
    # calculate the shape of dataset
    data = load(datadir + fns[0])[matkey]
    data = minmax_normalize(data)
    data= np.transpose(data, (2, 0, 1))#BHW
    data = data.astype(np.float32)  # Cave/ICVL数据集需要将原本的float64转换成float32
    # print(data.shape)


    data = crop_center(data, crop_sizes[0], crop_sizes[1]) #change preprocess to crop_cnter
    #data = preprocess(data)
    N = data.shape[0]
    # print(data.shape)
    # data=rotAugmentation(data)
    # print("旋转数据增强之后的数据{0}".format(data.shape))
    map_size = data.nbytes * len(fns) * 1.2
    print('map size (GB):', map_size / 1024 / 1024 / 1024)
    del data
    # import ipdb; ipdb.set_trace()

    # 若没有name这个文件或者文件夹，则生成一个name.db的文件
    if not os.path.exists(name + '.db'):
        os.mkdir(name + '.db')
    env = lmdb.open(name + '.db', map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        # 注意这里的键值  不仅影响数据的合并，并且影响后面数据加载器读取数据
        k = 0
        for i, fn in enumerate(fns):
            try:
                X = load(datadir + fn)[matkey]
            except:
                print('loading', datadir + fn, 'fail')
                continue
            #X = preprocess(X)

            X = minmax_normalize(X)
            X = X.astype(np.float32)  # Cave/ICVL数据集需要将原本的float64转换成float32
            # X = np.transpose(X,(1,2,0))
            X = np.transpose(X, (2, 0, 1))  # BHW
            X = crop_center(X, crop_sizes[0], crop_sizes[1])
            # print("旋转之前{0}".format(X.shape))
            # X = rotAugmentation(X)
            # print("旋转之后{0}".format(X.shape))
            X = X[np.newaxis,:,:,:]
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





# Create Pavia Centre dataset 
def create_PaviaCentre():
    print('create Pavia Centre...')
    datadir = './data/PaviaCentre/'
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]

    create_lmdb_train(
        datadir, fns, '/home/kaixuan/Dataset/PaviaCentre', 'hsi',  # your own dataset address
        crop_sizes=None,
        scales=(1,),
        ksizes=(101, 64, 64),
        strides=[(101, 32, 32)],
        load=loadmat, augment=True,
    )

# Create ICVL training dataset
def create_icvl64_31():
    print('create icvl64_31...')
    datadir = '/home/jiahua/HSI-Group/HSI-dataset/train_data/train_mat/' # your own data address
    fns = os.listdir(datadir) 
    fns = [fn.split('.')[0]+'.mat' for fn in fns]
    
    create_lmdb_train(
        datadir, fns, '/home/jiahua/HSI-Group/HSI-dataset/train_data/train_ICVL', 'rad',  # your own dataset address
        crop_sizes=(1024, 1024),
        scales=(1, 0.5, 0.25),
        ksizes=(31, 64, 64),
        strides=[(31, 64, 64), (31, 32, 32), (31, 32, 32)],        
        load=h5py.File, augment=True,
    )


def create_cave64_31():
    print('create cave64_31...')
    datadir = '/home/jiahua/HSI-Group/HSI-dataset/train_data/train_cave_mat/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_train(
        datadir, fns, '/home/jiahua/HSI-Group/HSI-dataset/train_data/train_CAVE_smallsamll', 'msi',  # your own dataset address
        crop_sizes=(512, 512),
        scales=(1, 0.5, 0.25),
        ksizes=(31, 40, 40),
        strides=[(31, 100, 100), (31, 100, 100), (31, 100, 100)],
        load=loadmat, augment=True,
    )

def create_icvl_test():
    print('create icvl test...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/test_icvl_complex/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_test_center(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/ICVL_complex_test', 'rad',  # your own dataset address
        crop_sizes=(512, 512),
        scales=(1, ),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=h5py.File, augment=True,
    )
def create_pavia_test():
    print('create pavia test...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/pavia/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_test_center(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/pavia_complex_test_360', 'pavia',  # your own dataset address
        crop_sizes=(360, 360),
        scales=(1, ),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,
    )
def create_icvl_test_gauss_30():
    print('create icvl test_gauss_30...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/test_icvl/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_test_center_gauss(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/ICVL_gauss_test_gauss_30', 'rad',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=h5py.File, augment=True,Addnoise=train_transform_sigma_30
    )
def create_icvl_test_gauss_50():
    print('create icvl test_gauss_50...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/test_icvl/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_test_center_gauss(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/ICVL_gauss_test_gauss_50', 'rad',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=h5py.File, augment=True,Addnoise=train_transform_sigma_50
    )
def create_icvl_test_gauss_70():
    print('create icvl test_gauss_70...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/test_icvl/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_test_center_gauss(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/ICVL_gauss_test_gauss_70', 'rad',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=h5py.File, augment=True,Addnoise=train_transform_sigma_70
    )
def create_icvl_test_gauss_blind():
    print('create icvl test_gauss_blind...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/test_icvl/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_test_center_gauss(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/ICVL_gauss_test_gauss_blind', 'rad',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=h5py.File, augment=True,Addnoise=train_transform_0
    )



# def create_icvl_test():
#     print('create icvl test...')
#     datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/test_icvl/'  # your own data address
#     fns = os.listdir(datadir)
#     fns = [fn.split('.')[0] + '.mat' for fn in fns]
#
#     create_lmdb_test_center(
#         datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/ICVL_gauss_256_test', 'rad',  # your own dataset address
#         crop_sizes=(256, 256),
#         scales=(1, ),
#         ksizes=(31, 64, 64),
#         strides=[(31, 32, 32)],
#         load=h5py.File, augment=True,
#     )
def download_ICVL():
    print('download icvl64_31...')
    import urllib.request
    base_url = "http://icvl.cs.bgu.ac.il/img/hs_pub/"
    path = "/home/jiahua/HSI-Group/HSI-dataset/train_data/train_mat/"
    import sys
    result = []
    with open('/home/jiahua/HSI-Group/HSI-dataset/train_data/ICVL_train.txt', 'r') as f:
        for line in f:
            result.append(line.strip('\n'))

    if not os.path.exists(path):
        print("Selected folder not exist, try to create it.")
        os.makedirs(path)
    for filename in result:
        url = base_url + filename
        print("Try downloading file: {}".format(url))
        filepath = path  + filename
        if os.path.exists(filepath):
            print("File have already exist. skip")
        else:
            try:
                urllib.request.urlretrieve(url, filename=filepath)
            except Exception as e:
                print("Error occurred when downloading file, error message:")
                print(e)
def download_ICVL_Test():
    print('download icvl64_31 test...')
    import urllib.request
    base_url = "http://icvl.cs.bgu.ac.il/img/hs_pub/"
    path = "/home/xiaojiahua/code/RepDnCNN_RRelu/data/test_icvl/"
    import sys
    result = []
    with open('/home/xiaojiahua/code/QRNN3D/ICVL_test_gauss.txt', 'r') as f:
        for line in f:
            result.append(line.strip('\n'))

    if not os.path.exists(path):
        print("Selected folder not exist, try to create it.")
        os.makedirs(path)
    for filename in result:
        url = base_url + filename
        print("Try downloading file: {}".format(url))
        filepath = path  + filename
        if os.path.exists(filepath):
            print("File have already exist. skip")
        else:
            try:
                urllib.request.urlretrieve(url, filename=filepath)
            except Exception as e:
                print("Error occurred when downloading file, error message:")
                print(e)
def download_ICVL_Test_complex():
    print('download icvl64_31 test...')
    import urllib.request
    base_url = "http://icvl.cs.bgu.ac.il/img/hs_pub/"
    path = "/home/xiaojiahua/code/RepDnCNN_RRelu/data/test_icvl_complex/"
    import sys
    result = []
    with open('/home/xiaojiahua/code/QRNN3D/ICVL_test_complex.txt', 'r') as f:
        for line in f:
            result.append(line.strip('\n'))

    if not os.path.exists(path):
        print("Selected folder not exist, try to create it.")
        os.makedirs(path)
    for filename in result:
        url = base_url + filename
        print("Try downloading file: {}".format(url))
        filepath = path  + filename
        if os.path.exists(filepath):
            print("File have already exist. skip")
        else:
            try:
                urllib.request.urlretrieve(url, filename=filepath)
            except Exception as e:
                print("Error occurred when downloading file, error message:")
                print(e)
def Addnoise(img,sigma,pch_size):
    sigma_ratio = sigma / 255
    noise = np.random.randn(*img.shape) * sigma_ratio
    noise = noise.astype(np.float32)
    im_noisy = img + noise

    return im_noisy
train_transform_sigma_30 = Compose([
        AddNoise(30)
    ])
train_transform_sigma_50 = Compose([
        AddNoise(50)
    ])
train_transform_sigma_70 = Compose([
        AddNoise(70)
    ])
train_transform_sigma_5 = Compose([
        AddNoise(5)
    ])
train_transform_sigma_25 = Compose([
        AddNoise(25)
    ])
train_transform_sigma_55 = Compose([
        AddNoise(55)
    ])
train_transform_sigma_75 = Compose([
        AddNoise(75)
    ])
train_transform_sigma_95 = Compose([
        AddNoise(95)
    ])
sigmas = [10,30,50,70]
train_transform_0 = Compose([
        AddNoiseBlindv2(50,100)
    ])



train_transform_1 = Compose([AddNoiseNoniid(sigmas)])

#case 2  no-iid guass + stripe noise
train_transform_2=Compose([AddNoiseNoniid(sigmas),
                           AddNoiseStripe()])
#case 3  no-iid guass + deadline
train_transform_3=Compose([AddNoiseNoniid(sigmas),
                           AddNoiseDeadline()])
# train_transform_5=Compose([
#         AddNoiseNoniid(sigmas),
#         AddNoiseComplex()])
# train_transform_7=Compose([
#         AddNoiseNoniid(sigmas),
#         AddNoiseComplex_2()])
#case 4  no-iid guass + impluse
train_transform_4=Compose([AddNoiseNoniid(sigmas),
                           AddNoiseImpulse()])
add_noniid_noise = Compose([
    SequentialSelect(
        transforms=[
            lambda x: x,
            AddNoiseImpulse(),
            AddNoiseStripe(),
            AddNoiseDeadline()
        ]
    )
])

train_transform_5=Compose([
        AddNoiseNoniid(sigmas),
        AddNoiseComplex()])

# train_transform_5 = Compose([
#     add_noniid_noise,
#     AddNoiseNoniid(sigmas)
#     # HSI2Tensor()
# ])
#case 5 Mixtrue noise
# train_transform_5=Compose([
#         AddNoiseNoniid(sigmas),
#         AddNoiseComplex()])
# train_transform_6=Compose([AddNoiseDeadline()])

def create_lmdb_test_center_gauss(
        datadir, fns, name, matkey,
        crop_sizes, scales, ksizes, strides,
        load=h5py.File, augment=True,
        seed=2017,Addnoise = train_transform_0):

    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)
    assert len(scales) == len(strides)
    # calculate the shape of dataset
    data = load(datadir + fns[0])
    data=data[matkey]
    data = minmax_normalize(data)
    data = data.astype(np.float32)  # Cave/ICVL数据集需要将原本的float64转换成float32


    data = crop_center(data, crop_sizes[0], crop_sizes[1]) #change preprocess to crop_cnter
    #data = preprocess(data)
    N = data.shape[0]
    # print(data.shape)
    # data=rotAugmentation(data)
    # print("旋转数据增强之后的数据{0}".format(data.shape))
    map_size = data.nbytes * len(fns) * 1.2
    print('map size (GB):', map_size / 1024 / 1024 / 1024)
    del data
    # import ipdb; ipdb.set_trace()

    # 若没有name这个文件或者文件夹，则生成一个name.db的文件
    if not os.path.exists(name + '.db'):
        os.mkdir(name + '.db')
    env = lmdb.open(name + '.db', map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        # 注意这里的键值  不仅影响数据的合并，并且影响后面数据加载器读取数据
        k = 0
        for i, fn in enumerate(fns):
            try:
                X = load(datadir + fn)[matkey]
            except:
                print('loading', datadir + fn, 'fail')
                continue
            #X = preprocess(X)

            X = minmax_normalize(X)
            X = X.astype(np.float32)  # Cave/ICVL数据集需要将原本的float64转换成float32
            # X = np.transpose(X,(2,0,1))
            X = crop_center(X, crop_sizes[0], crop_sizes[1])
            print(X.nbytes)
            X = Addnoise(X)
            print(X.nbytes)
            # print("旋转之前{0}".format(X.shape))
            # X = rotAugmentation(X)
            # print("旋转之后{0}".format(X.shape))
            X = X[np.newaxis,:,:,:]
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
def create_lmdb_test_center_complex(
        datadir, fns, name, matkey,
        crop_sizes, scales, ksizes, strides,
        load=h5py.File, augment=True,
        seed=2017, Addcomplex = train_transform_1):

    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)
    assert len(scales) == len(strides)
    # calculate the shape of dataset
    data = load(datadir + fns[0])[matkey]
    data = minmax_normalize(data)
    data = data.astype(np.float32)  # Cave/ICVL数据集需要将原本的float64转换成float32

    data = crop_center(data, crop_sizes[0], crop_sizes[1])  # change preprocess to crop_cnter
    # data = preprocess(data)
    N = data.shape[0]
    # print(data.shape)
    # data=rotAugmentation(data)
    # print("旋转数据增强之后的数据{0}".format(data.shape))
    map_size = data.nbytes * len(fns) * 1.2
    print('map size (GB):', map_size / 1024 / 1024 / 1024)
    del data
    # import ipdb; ipdb.set_trace()

    # 若没有name这个文件或者文件夹，则生成一个name.db的文件
    if not os.path.exists(name + '.db'):
        os.mkdir(name + '.db')
    env = lmdb.open(name + '.db', map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        # 注意这里的键值  不仅影响数据的合并，并且影响后面数据加载器读取数据
        k = 0
        for i, fn in enumerate(fns):
            try:
                X = load(datadir + fn)[matkey]
            except:
                print('loading', datadir + fn, 'fail')
                continue
            # X = preprocess(X)

            X = minmax_normalize(X)
            X = X.astype(np.float32)  # Cave/ICVL数据集需要将原本的float64转换成float32
            X = np.transpose(X,(2,0,1)) #cave需要转换
            X = crop_center(X, crop_sizes[0], crop_sizes[1])
            print(X.nbytes/ 1024 / 1024 / 1024)
            # print(X.shape)
            # X = X[0]

            X = Addcomplex(X)
            X = X.astype(np.float32)
            print(X.nbytes/ 1024 / 1024 / 1024)
            # print("旋转之前{0}".format(X.shape))
            # X = rotAugmentation(X)
            # print("旋转之后{0}".format(X.shape))
            X = X[np.newaxis, :, :, :]
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
def create_icvl_test_complex_iid():
    print('create icvl test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/test_icvl_complex/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_1

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/ICVL_gauss_test_complex_iid', 'rad',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=h5py.File, augment=True,Addcomplex=Addcomplex
    )
def create_icvl_test_complex_iid_Stripe():
    print('create icvl test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/test_icvl_complex/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_2

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/ICVL_gauss_test_complex_iid_Stripe', 'rad',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=h5py.File, augment=True,Addcomplex=Addcomplex
    )
def create_icvl_test_complex_iid_Deadline():
    print('create icvl test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/test_icvl_complex/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_3

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/ICVL_gauss_test_complex_iid_Deadline', 'rad',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=h5py.File, augment=True,Addcomplex=Addcomplex
    )
def create_icvl_test_complex_Deadline():
    print('create icvl test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/test_icvl_complex/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_6

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/ICVL_gauss_test_complex_Deadline', 'rad',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=h5py.File, augment=True,Addcomplex=Addcomplex
    )
def create_icvl_test_complex_iid_Impulse():
    print('create icvl test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/test_icvl_complex/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_4

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/ICVL_gauss_test_complex_iid_Impulse', 'rad',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=h5py.File, augment=True,Addcomplex=Addcomplex
    )
def create_icvl_test_complex_Mix():
    print('create icvl test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/test_icvl_complex/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_5

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/ICVL_gauss_test_complex_Mix', 'rad',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=h5py.File, augment=True,Addcomplex=Addcomplex
    )
def create_pavia_center_test_complex_Mix():
    print('create pabia test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/pavia/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_5

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/pavia_complex_Mix_360', 'pavia',
        # your own dataset address
        crop_sizes=(360, 360),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addcomplex=Addcomplex
    )
def create_pavia_center_test_complex_iid():
    print('create pabia test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/pavia/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_1

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/pavia_complex_iid_360', 'pavia',
        # your own dataset address
        crop_sizes=(360, 360),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addcomplex=Addcomplex
    )
def create_pavia_center_test_complex_iid_Stripe():
    print('create pabia test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/pavia/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_2

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/pavia_complex_Stripe_360', 'pavia',
        # your own dataset address
        crop_sizes=(360, 360),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addcomplex=Addcomplex
    )
def create_pavia_center_test_complex_Deadline():
    print('create pabia test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/pavia/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_3

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/pavia_complex_Deadline_360', 'pavia',
        # your own dataset address
        crop_sizes=(360, 360),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addcomplex=Addcomplex
    )
def create_pavia_center_test_complex_Impulse():
    print('create pabia test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/pavia/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_4

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/pavia_complex_Impulse_360', 'pavia',
        # your own dataset address
        crop_sizes=(360, 360),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addcomplex=Addcomplex
    )

def create_cave_test():
    print('create cave test...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/cave_complex_gt/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_test_center(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/CAVE_complex_test', 'cave_ms_double',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,
    )
def create_urban_test():
    print('create urban test...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data_3d/Urban/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_test_center(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/urban', 'urban',
        # your own dataset address
        crop_sizes=(228, 228),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,
    )
def create_indiapine_test():
    print('create urban test...')
    datadir = '/home/jiahua/HSI-Group/HSI-dataset/test_data/dc_200_mat/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_test_center(
        datadir, fns, '/home/jiahua/HSI-Group/HSI-dataset/test_data/dc_200', 'dc',
        # your own dataset address
        crop_sizes=(200, 200),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,
    )

def create_indianpine_trian():
    print('create urban test...')
    datadir = '/home/jiahua/HSI-Group/HSI-dataset/test_data/indian_pine_orgin_mat/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_train(
        datadir, fns, '/home/jiahua/HSI-Group/HSI-dataset/train_data/train_indian', 'indian_pines',
        # your own dataset address
        crop_sizes=(144, 144),
        scales=(1,),
        ksizes=(31, 32, 32),
        strides=[(31, 8, 8)],
        load=loadmat, augment=True,
    )
def create_CAVE_test_complex_Mix():
    print('create cave test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/cave/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_5

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/CAVE_test_complex_Mix', 'cave_ms_double',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addcomplex=Addcomplex
    )
def create_CAVE_test_gauss_5():
    print('create CAVE test_gauss_50...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/cave_gauss_gt/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_test_center_gauss(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/CAVE_gauss_test_gauss_5', 'cave_ms_double',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addnoise=train_transform_sigma_5
    )
def create_CAVE_test_gauss_25():
    print('create CAVE test_gauss_50...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/cave_gauss_gt/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_test_center_gauss(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/CAVE_gauss_test_gauss_25', 'cave_ms_double',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addnoise=train_transform_sigma_25
    )
def create_CAVE_test_gauss_55():
    print('create CAVE test_gauss_50...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/cave_gauss_gt/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_test_center_gauss(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/CAVE_gauss_test_gauss_55', 'cave_ms_double',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addnoise=train_transform_sigma_55
    )
def create_CAVE_test_gauss_75():
    print('create CAVE test_gauss_50...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/cave_gauss_gt/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_test_center_gauss(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/CAVE_gauss_test_gauss_75', 'cave_ms_double',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addnoise=train_transform_sigma_75
    )

def create_CAVE_test_gauss_95():
    print('create CAVE test_gauss_50...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/cave_gauss_gt/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_test_center_gauss(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/CAVE_gauss_test_gauss_95', 'cave_ms_double',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True, Addnoise=train_transform_sigma_95
    )
def create_CAVE_test_complex_iid():
    print('create icvl test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/cave_complex_gt/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_1

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/CAVE_test_complex_iid', 'cave_ms_double',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addcomplex=Addcomplex
    )
def create_CAVE_test_complex_iid_Stripe():
    print('create icvl test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/cave_complex_gt/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_2

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/CAVE_test_complex_iid_Stripe', 'cave_ms_double',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addcomplex=Addcomplex
    )
def create_CAVE_test_complex_iid_Deadline():
    print('create icvl test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/cave_complex_gt/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_3

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/CAVE_test_complex_iid_Deadline', 'cave_ms_double',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addcomplex=Addcomplex
    )

def create_CAVE_test_complex_iid_Impulse():
    print('create icvl test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/cave_complex_gt/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_4

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/CAVE_test_complex_iid_Impulse', 'cave_ms_double',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addcomplex=Addcomplex
    )
def create_CAVE_test_complex_iid_stripe_deadline():
    print('create icvl test_complex iid...')
    datadir = '/home/xiaojiahua/code/RepDnCNN_RRelu/data/cave_complex_gt/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_7

    create_lmdb_test_center_complex(
        datadir, fns, '/home/xiaojiahua/code/RepDnCNN_RRelu/DataSet/CAVE_test_complex_iid_Stripe_Deadline', 'cave_ms_double',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addcomplex=Addcomplex
    )
def create_DC_test_complex_Mix():
    print('create DCtest_complex iid...')
    datadir = '/home/jiahua/HSI-Group/HSI-dataset/test_data/dc_200_mat/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_5

    create_lmdb_test_center_complex(
        datadir, fns, '/home/jiahua/HSI-Group/HSI-dataset/test_data/dc_Mix', 'dc',
        # your own dataset address
        crop_sizes=(200, 200),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addcomplex=Addcomplex
    )
def create_DC_test():
    print('create DC test...')
    datadir = '/home/jiahua/HSI-Group/HSI-dataset/test_data/dc_200_mat/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    Addcomplex = train_transform_5
    create_lmdb_test_center_complex(
        datadir, fns, '/home/jiahua/HSI-Group/HSI-dataset/test_data/dc_gauss_mix', 'dc',
        # your own dataset address
        crop_sizes=(200, 200),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addcomplex= Addcomplex
    )

def create_UA_icvl_test():
    print('create icvl test...')
    datadir = '/home/jiahua/Diffusion_model/ASSNet/Test_data/ICVL_Complex_mat/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]

    create_lmdb_test_center(
        datadir, fns, '/home/jiahua/Diffusion_model/ASSNet/Test_data/ICVL_Complex', 'rad',  # your own dataset address
        crop_sizes=(512, 512),
        scales=(1, ),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,
    )
def create_ICVL_UA_test_complex_Mix():
    print('create UA_test Mix...')
    datadir = '/home/jiahua/Diffusion_model/ASSNet/Test_data/ICVL_Complex_mat/'  # your own data address
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0] + '.mat' for fn in fns]
    print(fns)
    Addcomplex = train_transform_5
    # Addcomplex = lambda x: x
    create_lmdb_test_center_complex(
        datadir, fns, '/home/jiahua/Diffusion_model/ASSNet/Test_data/ICVL_Mix', 'rad',
        # your own dataset address
        crop_sizes=(512, 512),
        scales=(1,),
        ksizes=(31, 64, 64),
        strides=[(31, 32, 32)],
        load=loadmat, augment=True,Addcomplex=Addcomplex
    )
if __name__ == '__main__':
    create_ICVL_UA_test_complex_Mix()
    # create_UA_icvl_test()
    # create_DC_test_complex_Mix()
    # create_DC_test()
    # create_indianpine_trian()
    # create_CAVE_test_gauss_5()
    # create_CAVE_test_gauss_25()
    # create_CAVE_test_gauss_55()
    # create_CAVE_test_gauss_75()
    # create_CAVE_test_gauss_95()
    # create_urban_test()
    # create_indiapine_test()
    # create_pavia_center_test_complex_iid()
    # create_pavia_center_test_complex_iid_Stripe()
    # create_pavia_center_test_complex_Deadline()
    # create_pavia_center_test_complex_Impulse()
    # create_pavia_test()

    # create_cave_test()
    # create_CAVE_test_complex_iid()
    # create_CAVE_test_complex_iid_Stripe()
    # create_CAVE_test_complex_iid_Deadline()
    # create_CAVE_test_complex_iid_stripe_deadline()
    # # download_ICVL_Test()
    # create_icvl64_31()
    # create_cave64_31()
    # download_ICVL()
    # create_PaviaCentre()
    # create_cave_test()
    # create_CAVE_test_complex_Mix()

    pass

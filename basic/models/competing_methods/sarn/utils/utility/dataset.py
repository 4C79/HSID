# There are functions for creating a train and validation iterator.
import torch
import torchvision
import random
import cv2
from utils.utilitys.data_tools import *

from .util import *


from utils.utilitys import *

from torchvision.transforms import Compose, ToPILImage, ToTensor, RandomHorizontalFlip, RandomChoice
from torch.utils.data import DataLoader, Dataset
from torchnet.dataset import TransformDataset, SplitDataset, TensorDataset, ResampleDataset

from PIL import Image
from skimage.util import random_noise
from scipy.ndimage.filters import gaussian_filter


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# Define Transforms
class RandomGeometricTransform(object):
    def __call__(self, img):
        """
        Args:
            img (np.mdarray): Image to be geometric transformed.

        Returns:
            np.ndarray: Randomly geometric transformed image.
        """
        if random.random() < 0.25:
            return data_augmentation(img)
        return img


class RandomCrop(object):
    """For HSI (c x h x w)"""

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        img = rand_crop(img, self.crop_size, self.crop_size)
        return img


class SequentialSelect(object):
    def __pos(self, n):
        i = 0
        while True:
            # print(i)
            yield i
            i = (i + 1) % n

    def __init__(self, transforms):
        self.transforms = transforms
        self.pos = LockedIterator(self.__pos(len(transforms)))

    def __call__(self, img):
        out = self.transforms[next(self.pos)](img)
        return out


# 非iid高斯噪声
class AddIidNoise(object):
    """add gaussian noise to the given numpy array (B,H,W)"""

    def __init__(self, sigma_min=0, sigma_max=75, pch_size=64, radius=5):
        self.win = 2 * radius + 1
        self.pch_size = pch_size
        self.sigma_spatial = radius
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, im_gt):
        pch_size = self.pch_size
        C = im_gt.shape[0]
        # generate sigmaMap
        sigma_map = self.generate_sigma()
        # print("sigma分布{0}".format(sigma_map))
        for i in range(C - 1):
            sigma = self.generate_sigma()
            sigma_map = np.vstack((sigma_map, sigma))
        # print("sigma_map的形状{0}".format(sigma_map.shape))
        # print("sigma_map{0}".format(sigma_map))
        # generate noise
        noise = torch.randn(im_gt.shape).numpy() * sigma_map
        # noise = torch.randn(im_gt.shape).numpy() * (25 / 255.0)
        # print("sigma_map的形状{0}".format(noise.shape))
        im_noisy = im_gt + noise.astype(np.float32)
        # im_gt, im_noisy, sigma_map = random_augmentation(im_gt, im_noisy, sigma_map)
        sigma2_map_gt = np.square(sigma_map)
        sigma2_map_gt = np.where(sigma2_map_gt < 1e-10, 1e-10, sigma2_map_gt)
        # if self.noise_estimate:
        # sigma2_map_est = sigma_estimate(im_noisy, im_gt, self.win, self.sigma_spatial)  # CxHxW
        # sigma2_map_est = torch.from_numpy(sigma2_map_est)
        # sigma2_map_gt = np.tile(np.square(sigma_map), (C,1, 1))
        # sigma2_map_gt =np.square(sigma_map)
        # #print("sigma2_map_gt{0}".format(sigma2_map_gt.shape))
        # #print("sigma2_map_gt{0}".format(sigma2_map_gt))
        # sigma2_map_gt = np.where(sigma2_map_gt < 1e-10, 1e-10, sigma2_map_gt)
        # print("噪声分布形状3{0}".format(sigma2_map_gt.shape))
        # sigma2_map_gt = torch.from_numpy(sigma2_map_gt)
        # im_gt = torch.from_numpy(im_gt)
        # im_noisy = torch.from_numpy(im_noisy)

        return im_noisy, sigma2_map_gt

    def generate_sigma(self):
        pch_size = self.pch_size
        # 随机一个核心
        center = [random.uniform(0, pch_size), random.uniform(0, pch_size)]
        scale = random.uniform(pch_size / 4, pch_size / 4 * 3)
        kernel = gaussian_kernel(pch_size, pch_size, center, scale)
        # print("高斯核的形状{0}".format(kernel.shape))
        up = random.uniform(self.sigma_min / 255.0, self.sigma_max / 255.0)
        down = random.uniform(self.sigma_min / 255.0, self.sigma_max / 255.0)
        if up < down:
            up, down = down, up
        up += 5 / 255.0
        sigma_map = down + (kernel - kernel.min()) / (kernel.max() - kernel.min()) * (up - down)
        sigma_map = sigma_map.astype(np.float32)

        return sigma_map[np.newaxis, :, :]


# 非iid高斯噪声-test
class AddIidNoiseTest(object):
    """add gaussian noise to the given numpy array (B,H,W)"""

    def __init__(self, sigma_min=0, sigma_max=75, pch_size=64, radius=5):
        self.win = 2 * radius + 1
        self.pch_size = pch_size
        self.sigma_spatial = radius
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, im_gt):
        pch_size = self.pch_size
        C = im_gt.shape[0]
        # generate sigmaMap
        sigma_map = self.generate_sigma()
        # print("sigma分布{0}".format(sigma_map))
        for i in range(C - 1):
            sigma = self.generate_sigma()
            sigma_map = np.vstack((sigma_map, sigma))

        # print("sigma_map的形状{0}".format(sigma_map.shape))
        # print("sigma_map{0}".format(sigma_map))
        # generate noise
        noise = torch.randn(im_gt.shape).numpy() * sigma_map
        # print("sigma_map的形状{0}".format(noise.shape))
        im_noisy = im_gt + noise.astype(np.float32)
        # im_gt, im_noisy, sigma_map = random_augmentation(im_gt, im_noisy, sigma_map)
        return im_noisy

    def generate_sigma(self):
        pch_size = self.pch_size
        # 随机一个核心
        center = [random.uniform(0, pch_size), random.uniform(0, pch_size)]
        scale = random.uniform(pch_size / 4, pch_size / 4 * 3)
        kernel = gaussian_kernel(pch_size, pch_size, center, scale)
        # print("高斯核的形状{0}".format(kernel.shape))
        up = random.uniform(self.sigma_min / 255.0, self.sigma_max / 255.0)
        down = random.uniform(self.sigma_min / 255.0, self.sigma_max / 255.0)
        if up < down:
            up, down = down, up
        up += 5 / 255.0
        sigma_map = down + (kernel - kernel.min()) / (kernel.max() - kernel.min()) * (up - down)
        sigma_map = sigma_map.astype(np.float32)

        return sigma_map[np.newaxis, :, :]


class AddNoise(object):
    """add gaussian noise to the given numpy array (B,H,W)"""

    def __init__(self, sigma, pch_size=128):
        self.sigma_ratio = sigma / 255
        self.sigma = sigma
        self.pch_size = pch_size

    def __call__(self, img):
        # print("传入图像的shape")
        # print(img.shape)
        noise = np.random.randn(*img.shape) * self.sigma_ratio
        noise_map = np.ones(img.shape) * self.sigma_ratio
        noise_map = noise_map.astype(np.float32)
        # sigma_map = sigma_map.astype(np.float32)
        im_noisy = img + noise
        return im_noisy, noise_map


# class AddNoise(object):
#     """add gaussian noise to the given numpy array (B,H,W)"""
#     def __init__(self, sigma):
#         self.sigma_ratio = sigma / 255.
#
#     def __call__(self, img):
#         noise = np.random.randn(*img.shape) * self.sigma_ratio
#         noise = noise.astype(np.float32)
#         return img + noise

class AddNoiseGaussianWhite(object):
    def __init__(self, snr, seed):
        self.seed = seed
        self.snr = snr

    def __call__(self, img):
        '''
           加入高斯白噪声 Additive White Gaussian Noise
           :param x: 原始信号
           :param snr: 信噪比
           :return: 加入噪声后的信号
           '''
        np.random.seed(self.seed)  # 设置随机种子
        snr = 10 ** (self.snr / 10.0)
        xpower = np.sum(img ** 2) / len(img)
        npower = xpower / snr
        noise = np.random.randn(*img.shape) * np.sqrt(npower)
        return img + noise


class AddNoiseBlind(object):
    """add blind gaussian noise to the given numpy array (B,H,W)"""

    def __pos(self, n):
        i = 0
        while True:
            yield i
            i = (i + 1) % n

    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas) / 255.
        self.pos = LockedIterator(self.__pos(len(sigmas)))

    def __call__(self, img):
        sigma = self.sigmas[next(self.pos)]
        noise = np.random.randn(*img.shape) * sigma
        noise = noise.astype(np.float32)
        noise_map = np.ones(img.shape) * sigma
        noise_map = noise_map.astype(np.float32)
        # print("noise shape{0}".format(noise.shape))
        return img + noise,noise_map


class AddNoiseBlindv2(object):
    """add blind gaussian noise to the given numpy array (B,H,W)"""

    def __init__(self, min_sigma, max_sigma):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img):
        noise = np.random.randn(*img.shape) * np.random.uniform(self.min_sigma, self.max_sigma) / 255
        noise = noise.astype(np.float32)
        return img + noise


class AddNoiseNoniid(object):
    """add non-iid gaussian noise to the given numpy array (B,H,W)"""

    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas) / 255.

    def __call__(self, img):

        bwsigmas = np.reshape(self.sigmas[np.random.randint(0, len(self.sigmas), img.shape[0])], (-1, 1, 1))
        noise = np.random.randn(*img.shape) * bwsigmas
        noise_map = np.ones(img.shape) * bwsigmas
        noise_map = noise_map.astype(np.float32)
        return img + noise,noise_map


class AddNoiseMixed(object):
    """add mixed noise to the given numpy array (B,H,W)
    Args:
        noise_bank: list of noise maker (e.g. AddNoiseImpulse)
        num_bands: list of number of band which is corrupted by each item in noise_bank"""

    def __init__(self, noise_bank, num_bands):
        assert len(noise_bank) == len(num_bands)
        self.noise_bank = noise_bank
        self.num_bands = num_bands

    def __call__(self, img):
        # img = img[0]
        B, H, W = img.shape
        all_bands = np.random.permutation(range(B))
        pos = 0
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            if 0 < num_band <= 1:
                num_band = int(np.floor(num_band * B))
            bands = all_bands[pos:pos + num_band]
            pos += num_band
            img = noise_maker(img, bands)
        return img


class _AddNoiseImpulse(object):
    """add impulse noise to the given numpy array (B,H,W)"""

    def __init__(self, amounts, s_vs_p=0.5):
        self.amounts = np.array(amounts)
        self.s_vs_p = s_vs_p

    def __call__(self, img, bands):
        # bands = np.random.permutation(range(img.shape[0]))[:self.num_band]
        bwamounts = self.amounts[np.random.randint(0, len(self.amounts), len(bands))]
        for i, amount in zip(bands, bwamounts):
            # print("impulse---{0}".format(i))
            self.add_noise(img[i, ...], amount=amount, salt_vs_pepper=self.s_vs_p)
        return img

    def add_noise(self, image, amount, salt_vs_pepper):
        # out = image.copy()
        out = image
        p = amount
        q = salt_vs_pepper
        flipped = np.random.choice([True, False], size=image.shape,
                                   p=[p, 1 - p])
        salted = np.random.choice([True, False], size=image.shape,
                                  p=[q, 1 - q])
        peppered = ~salted
        out[flipped & salted] = 1
        out[flipped & peppered] = 0
        return out


class _AddNoiseStripe(object):
    """add stripe noise to the given numpy array (B,H,W)"""

    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img, bands):
        B, H, W = img.shape
        # print("-----strip noise-----")
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_stripe = np.random.randint(np.floor(self.min_amount * W), np.floor(self.max_amount * W), len(bands))
        for i, n in zip(bands, num_stripe):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            stripe = np.random.uniform(0, 1, size=(len(loc),)) * 0.5 - 0.25
            # print("stripe---{0}".format(stripe.shape))
            img[i, :, loc] -= np.reshape(stripe, (-1, 1))
            # print("stripe reshape ---{0}".format(np.reshape(stripe, (-1, 1)).shape))
            # print("img----{0}".format(img[i, :, loc].shape))
        return img


class _AddNoiseDeadline(object):
    """add deadline noise to the given numpy array (B,H,W)"""

    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img, bands):
        # print("----deadline----")
        # print(img.shape)
        B, H, W = img.shape
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_deadline = np.random.randint(np.ceil(self.min_amount * W), np.ceil(self.max_amount * W), len(bands))
        for i, n in zip(bands, num_deadline):
            # print("deadline----{0}".format(i))
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            img[i, :, loc] = 0
        return img


class AddNoiseImpulse(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])]
        self.num_bands = [1 / 3]


class AddNoiseStripe(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseStripe(0.05, 0.15)]
        self.num_bands = [1 / 3]


class AddNoiseDeadline(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseDeadline(0.05, 0.15)]
        self.num_bands = [1 / 3]


class AddNoiseComplex(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [
            _AddNoiseStripe(0.05, 0.15),
            _AddNoiseDeadline(0.05, 0.15),
            _AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])
        ]
        self.num_bands = [1 / 3, 1 / 3, 1 / 3]


class HSI2Tensor2(object):
    """
    Transform a numpy array with shape (C, H, W)
    into torch 4D Tensor (1, C, H, W) or (C, H, W)
    """

    def __init__(self, use_2dconv):
        self.use_2dconv = use_2dconv

    def __call__(self, hsi, sigma):
        print("进入numpy转换tensor环节")
        print(type(hsi))
        hsi = np.array(hsi, dtype=float)
        sigma = np.array(sigma, dtype=float)
        print(type(hsi))
        if self.use_2dconv:
            img = torch.from_numpy(hsi)
            sigma = torch.from_numpy(sigma)
        else:
            # img = torch.from_numpy(hsi)
            img = torch.from_numpy(hsi[np.newaxis, :, :, :])
            sigma = torch.from_numpy(sigma[np.newaxis, :, :, :])
        # for ch in range(hsi.shape[0]):
        #     hsi[ch, ...] = minmax_normalize(hsi[ch, ...])
        # img = torch.from_numpy(hsi)
        return img.float(), sigma.float()


def HSI2TensorFun(hsi):
    img = torch.from_numpy(hsi[None])
    # img = torch.from_numpy(hsi[0])
    # print("噪声级噪声图像形状{0}".format(img.shape))
    return img.float()


class HSI2Tensor(object):
    """
    Transform a numpy array with shape (C, H, W)
    into torch 4D Tensor (1, C, H, W) or (C, H, W)
    """

    def __init__(self, use_2dconv):
        self.use_2dconv = use_2dconv

    def __call__(self, hsi):
        # print(type(hsi))
        # hsi=np.array(hsi, dtype=float)
        if self.use_2dconv:
            img = torch.from_numpy(hsi)
        else:
            # img = torch.from_numpy(hsi)
            img = torch.from_numpy(hsi[np.newaxis, :, :, :])
        # for ch in range(hsi.shape[0]):
        #     hsi[ch, ...] = minmax_normalize(hsi[ch, ...])
        # img = torch.from_numpy(hsi)
        return img.float()


class LoadMatHSI(object):
    def __init__(self, input_key, gt_key, transform=None):
        self.gt_key = gt_key
        self.input_key = input_key
        self.transform = transform

    def __call__(self, mat):
        if self.transform:
            input = self.transform(mat[self.input_key][:].transpose((2, 0, 1)))
            gt = self.transform(mat[self.gt_key][:].transpose((2, 0, 1)))
        else:
            input = mat[self.input_key][:].transpose((2, 0, 1))
            gt = mat[self.gt_key][:].transpose((2, 0, 1))
        # input = torch.from_numpy(input[None]).float()
        input = torch.from_numpy(input).float()
        # gt = torch.from_numpy(gt[None]).float()  # for 3D net
        gt = torch.from_numpy(gt).float()

        return input, gt


class LoadMatKey(object):
    def __init__(self, key):
        self.key = key

    def __call__(self, mat):
        item = mat[self.key][:].transpose((2, 0, 1))
        return item.astype(np.float32)


# Define Datasets
class DatasetFromFolder(Dataset):
    """Wrap data from image folder"""

    def __init__(self, data_dir, suffix='png'):
        super(DatasetFromFolder, self).__init__()
        self.filenames = [
            os.path.join(data_dir, fn)
            for fn in os.listdir(data_dir)
            if fn.endswith(suffix)
        ]

    def __getitem__(self, index):
        img = Image.open(self.filenames[index]).convert('L')
        return img

    def __len__(self):
        return len(self.filenames)


# class MatDataFromFolder(Dataset):
#     """Wrap mat data from folder"""
#     def __init__(self, data_dir, load=loadmat, suffix='mat', fns=None, size=None):
#         super(MatDataFromFolder, self).__init__()
#         if fns is not None:
#             self.filenames = [
#                 os.path.join(data_dir, fn) for fn in fns
#             ]
#         else:
#             self.filenames = [
#                 os.path.join(data_dir, fn)
#                 for fn in os.listdir(data_dir)
#                 if fn.endswith(suffix)
#             ]
#
#         self.load = load
#
#         if size and size <= len(self.filenames):
#             self.filenames = self.filenames[:size]
#
#         # self.filenames = self.filenames[5:]
#
#     def __getitem__(self, index):
#         # print(self.filenames[index])
#         mat = self.load(self.filenames[index])
#         return mat
#
#     def __len__(self):
#         return len(self.filenames)
#
#
# def get_train_valid_loader(dataset,
#                            batch_size,
#                            train_transform=None,
#                            valid_transform=None,
#                            valid_size=None,
#                            shuffle=True,
#                            verbose=False,
#                            num_workers=1,
#                            pin_memory=False):
#     """
#     Utility function for loading and returning train and valid
#     multi-process iterators over any pytorch dataset. A sample
#     of the images can be optionally displayed.
#     If using CUDA, num_workers should be set to 1 and pin_memory to True.
#     Params
#     ------
#     - dataset: full dataset which contains training and validation data
#     - batch_size: how many samples per batch to load. (train, val)
#     - train_transform/valid_transform: callable function
#       applied to each sample of dataset. default: transforms.ToTensor().
#     - valid_size: should be a integer in the range [1, len(dataset)].
#     - shuffle: whether to shuffle the train/validation indices.
#     - verbose: display the verbose information of dataset.
#     - num_workers: number of subprocesses to use when loading the dataset.
#     - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
#       True if using GPU.
#     Returns
#     -------
#     - train_loader: training set iterator.
#     - valid_loader: validation set iterator.
#     """
#     error_msg = "[!] valid_size should be an integer in the range [1, %d]." %(len(dataset))
#     if not valid_size:
#         valid_size = int(0.1 * len(dataset))
#     if not isinstance(valid_size, int) or valid_size < 1 or valid_size > len(dataset):
#         raise TypeError(error_msg)
#
#
#     # define transform
#     default_transform = lambda item: item  # identity maping
#     train_transform = train_transform or default_transform
#     valid_transform = valid_transform or default_transform
#
#     # generate train/val datasets
#     partitions = {'Train': len(dataset)-valid_size, 'Valid':valid_size}
#
#     train_dataset = TransformDataset(
#         SplitDataset(dataset, partitions, initial_partition='Train'),
#         train_transform
#     )
#
#     valid_dataset = TransformDataset(
#         SplitDataset(dataset, partitions, initial_partition='Valid'),
#         valid_transform
#     )
#
#     train_loader = DataLoader(train_dataset,
#                     batch_size=batch_size[0], shuffle=True,
#                     num_workers=num_workers, pin_memory=pin_memory)
#
#     valid_loader = DataLoader(valid_dataset,
#                     batch_size=batch_size[1], shuffle=False,
#                     num_workers=num_workers, pin_memory=pin_memory)
#
#     return (train_loader, valid_loader)
#
#
# def get_train_valid_dataset(dataset, valid_size=None):
#     error_msg = "[!] valid_size should be an integer in the range [1, %d]." %(len(dataset))
#     if not valid_size:
#         valid_size = int(0.1 * len(dataset))
#     if not isinstance(valid_size, int) or valid_size < 1 or valid_size > len(dataset):
#         raise TypeError(error_msg)
#
#     # generate train/val datasets
#     partitions = {'Train': len(dataset)-valid_size, 'Valid':valid_size}
#
#     train_dataset = SplitDataset(dataset, partitions, initial_partition='Train')
#     valid_dataset = SplitDataset(dataset, partitions, initial_partition='Valid')
#
#     return (train_dataset, valid_dataset)

class TransformAugmentationDataset(Dataset):
    def __init__(self, dataset, transform):
        super(NoTargetImageTransformDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img


class ImageTransformDataset_HSID(Dataset):
    def __init__(self, dataset, transform, target_transform=None, mark=False):
        super(ImageTransformDataset_HSID, self).__init__()
        # 训练为true,测试为false
        self.mark = mark
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.dataset[idx]
        # print(img.shape)
        target = img.copy()

        if self.transform is not None:
            if self.mark == False:
                img = self.transform(img)
                img = self.target_transform(img)
            else:
                img, sigma2_map_gt = self.transform(img)
                img = self.target_transform(img)
                sigma2_map_gt = self.target_transform(sigma2_map_gt)
        if self.target_transform is not None:
            target = self.target_transform(target)

        input_im = []
        input_vol = []
        target_im = []
        # print(img.shape)
        for index in range(img.shape[0]):
            noise_im = img[index]
            orig_im = target[index]
            noise_im = noise_im[np.newaxis, :, :]
            orig_im = orig_im[np.newaxis, :, :]

            if index < 12:
                noise_vol = img[0:24, :, :]
            elif index > 18:
                noise_vol = img[7:31, :, :]
            else:
                noise_vol_1 = img[index - 12:index, :, :].cpu()
                noise_vol_2 = img[index + 1:index + 13, :, :].cpu()
                noise_vol = np.concatenate((noise_vol_1, noise_vol_2), axis=0)
                noise_vol = torch.Tensor(noise_vol)
            # print(orig_im.shape)
            input_vol.append(noise_vol)
            input_im.append(noise_im)
            target_im.append(orig_im)
        # print(target_im)
        target_im = torch.stack(target_im, 0)
        input_vol = torch.stack(input_vol, 0)
        input_im = torch.stack(input_im, 0)
        if self.mark == False:
            return input_im, input_vol, target_im
        else:
            return input_im, input_vol, target_im


class ImageTransformDataset(Dataset):
    def __init__(self, dataset, transform, target_transform=None, mark=False):
        super(ImageTransformDataset, self).__init__()
        # 训练为true,测试为false
        self.mark = mark
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.dataset[idx]
        target = img.copy()
        if self.transform is not None:
            if self.mark == False:
                img = self.transform(img)
                img = self.target_transform(img)

            else:
                img, sigma2_map_gt = self.transform(img)
                img = self.target_transform(img)
                sigma2_map_gt = self.target_transform(sigma2_map_gt)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.mark == False:
            return {'GT': target, 'NO': img}
            # return img, target
        else:
            return {'GT': target, 'NO': img, 'SM': sigma2_map_gt}
            # return img, target, sigma2_map_gt


class NoTargetImageTransformDataset(Dataset):
    def __init__(self, dataset, transform):
        super(NoTargetImageTransformDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img


if __name__ == '__main__':
    """Mat dataset test"""
    dataset = MatDataFromFolder('/media/kaixuan/DATA/Papers/Code/Matlab/ECCV2018/ECCVResult/Indian/Indian_pines/')
    # mat = dataset[0]
    # hsi = mat['R_hsi'].transpose((2,0,1))
    # Visualize3D(hsi)
    print("测试")
    pass

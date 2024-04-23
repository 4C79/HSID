import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import os
def make_layer(block, num_layers, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_layers):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)
def get_adjacent(input_fea, pos, range):
    c = input_fea.shape[1]
    if pos+range <= c:
        adjacent = input_fea[:,pos:pos+range]
    else:
        adjacent = input_fea[:,c-range:c]
    return adjacent

def get_back(input_fea, pos, range):
    c = len(input_fea)
    if pos <= range:
        back = input_fea[0: pos]
    elif pos > range and pos < c - range:
        back = input_fea[pos - range: pos]
    else:
        back = input_fea[c - 2 * range - 1: pos]
    back = torch.stack([torch.as_tensor(item) for item in back],dim=1)
    return back

def get_forth(input_fea, pos, range):
    c = len(input_fea)
    if pos == c-1:
        return None
    if pos <= range:
        forth = input_fea[ pos + 1: 2 * range + 1]
    elif pos > range and pos < c - range:
        forth = input_fea[pos + 1: pos + range + 1]
    else:
        forth = input_fea[pos + 1: c]
    forth = torch.stack([torch.as_tensor(item) for item in forth],dim=1)
    return forth

spa_number = 0
spec_number = 0

def draw_numbers(x):
    c,h,w = x.shape
    result = x.argmax(dim=0)
    # result = torch.zeros(h, w)

    # for i in range(c):
    #     for j in range(h):
    #         for k in range(w):
    #             element = x[i, j, k]
    #             if element == 1 :
    #                 result[j,k] = i

    result = result.to('cpu')
    result = result.numpy().astype(int)
    # 统计元素出现的次数
    unique_elements, counts = np.unique(result, return_counts=True)

    # 获取按照元素出现次数降序排列的索引
    sorted_indices = np.argsort(-counts)

    # 创建一个映射字典，将原始值映射到新值
    mapping_dict = {unique_elements[sorted_indices[i]]: i + 1 for i in range(len(unique_elements))}
    mapped_matrix = np.vectorize(mapping_dict.get)(result)
    return mapped_matrix


def draw_featrue(x,name):

    # numlize_img = draw_numbers(x)

    # 绘制矩阵图

    # fig = plt.figure(x.size)
    # 添加颜色条
    global spa_number,spec_number
    # if spa_number == 30:
    #     plt.colorbar()
    # if spec_number == 30:
    # plt.colorbar()

    book = 'test'

    # 保存图形
    if name == "spa_mask":
        # 创建颜色反转的Colormap
        cmap_reversed = sns.color_palette("viridis", as_cmap=True)
        cmap_reversed = cmap_reversed.reversed()
        # 创建heatmap并应用颜色反转的Colormap
        sns.heatmap(x, cmap="viridis",annot=False, fmt='f',cbar=False,xticklabels=False, yticklabels=False)
        # sns.heatmap(x, cmap='viridis',annot=False, fmt='f',cbar=False,xticklabels=False, yticklabels=False)
        plt.gca().set_aspect('equal', adjustable='box')
        fig = plt.gcf() 
        spa_number = spa_number+1
        # plt.imshow(x, cbar=False, xticklabels=False, yticklabels=False)

        dir = "/home/jiahua/HSI-CVPR/hsi_pipeline/result/icvl/drconv_r_"+name+"_"+book+"/"

        if not os.path.exists(dir):
            os.mkdir(dir)

        fig.savefig(dir+str(spa_number)+'.png',bbox_inches='tight', pad_inches=0,dpi=300)

        print(spa_number)

        # plt.figure(figsize=(6, 6))
        # sns.heatmap(result[:,:,1], annot=False, fmt='f', cmap='viridis',cbar=False,xticklabels=False, yticklabels=False)    
        # plt.gca().set_aspect('equal', adjustable='box')
        # fig = plt.gcf() 
        # # fig.colorbar = False
        # fig.savefig(save_path+rect[:-4]+".png",bbox_inches='tight', pad_inches=0,dpi=300)

    elif name == "spec_mask":
        cmap_reversed = sns.color_palette("viridis", as_cmap=True)
        cmap_reversed = cmap_reversed.reversed()
        # ax = sns.heatmap(x,  cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
        sns.heatmap(x, cmap="viridis",annot=False, fmt='f',cbar=False,xticklabels=False, yticklabels=False)
        plt.gca().set_aspect('equal', adjustable='box')
        fig = plt.gcf() 
        spec_number = spec_number +1

        dir = "/home/jiahua/HSI-CVPR/hsi_pipeline/result/icvl/drconv_r_"+name+"_"+book+"/"
        if not os.path.exists(dir):
            os.mkdir(dir)

        plt.savefig(dir+str(spec_number)+'.png',bbox_inches='tight', pad_inches=0,dpi=300)
    
    # plt.close(fig)

    return None
from os import path as osp
from collections import OrderedDict
import yaml
import argparse
import random
import numpy as np

def zigzag_path(N):

    # print("zigzag_sub_v1", N)
    assert N % 2 == 0, "N should be even"

    def zigzag_path_lr(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for i in range(N):
            for j in range(N):
                # If the row number is even, move right; otherwise, move left
                col = j if i % 2 == 0 else N - 1 - j
                path.append((start_row + dir_row * i) * N + start_col + dir_col * col)
        return path

    def zigzag_path_tb(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for j in range(N):
            for i in range(N):
                # If the column number is even, move down; otherwise, move up
                row = i if j % 2 == 0 else N - 1 - i
                path.append((start_row + dir_row * row) * N + start_col + dir_col * j)
        return path

    paths = []
    for start_row, start_col, dir_row, dir_col in [
        (0, 0, 1, 1),
        (0, N - 1, 1, -1),
        (N - 1, 0, -1, 1),
        (N - 1, N - 1, -1, -1),
    ]:
        paths.append(zigzag_path_lr(N, start_row, start_col, dir_row, dir_col))
        paths.append(zigzag_path_tb(N, start_row, start_col, dir_row, dir_col))

    for _index, _p in enumerate(paths):
        paths[_index] = np.array(_p)
    return paths

def reverse_permut_np(permutation):
    n = len(permutation)
    reverse = np.array([0] * n)
    for i in range(n):
        reverse[permutation[i]] = i
    return reverse

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def  parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='train', help='train or test')
    parser.add_argument('-method', type=str, default='mamba', help='which method to use')
    parser.add_argument('-type', type=str, default='None', help='which method\'s config to use')
    
    args = parser.parse_args()
    
    if args.type == "None":
        opt_path = '/home/jiahua/HSID/options/' + args.method + '_hsid.yml'
    else:
        opt_path = '/home/jiahua/HSID/options/' + args.method + '_' + args.type + '_hsid.yml'
    
    # parse yml to dict
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    opt['mode'] = args.mode

    return opt

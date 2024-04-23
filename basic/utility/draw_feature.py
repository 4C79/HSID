from os import mkdir
import torch
import torchvision
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

def test():
    import matplotlib.pyplot as plt
    import numpy as np

    # 创建一个示例矩阵
    matrix = np.array([[1, 2], [3, 4]])

    # 绘制矩阵图
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')

    # 添加颜色条
    plt.colorbar()

    # 保存图形
    plt.savefig('matrix_plot.png')

def draw_numbers(x):
    c,h,w = x.shape
    result = torch.zeros(h, w)
    for i in range(c):
        for j in range(h):
            for k in range(w):
                element = custom_tensor[i, j, k]
                if element == 1 :
                    result[j,k] = i
    
    result = result.numpy().astype(int)
    return result


def draw_featrue(x):
    
    numlize_img = draw_numbers(x)

    # 绘制矩阵图
    plt.imshow(numlize_img, cmap='viridis', interpolation='nearest')

    # 添加颜色条
    plt.colorbar()

    # 保存图形
    plt.savefig('test.png')
    
    return None


if __name__ == '__main__':
    custom_tensor = torch.tensor([[[0,1,0], [1,0,0],[1,0,0]],
                                  [[1,0,0], [0,1,0],[0,1,0]],
                                  [[0,0,1], [0,0,1],[0,0,1]]])
    draw_featrue(custom_tensor)
    # test()
    
    
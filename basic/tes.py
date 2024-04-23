import torch

# def rearrange_and_merge(array, directions):
#   """
#   按照指定方向重新排序数组元素并将它们合并

#   Args:
#     array: 输入数组，torch 张量
#     directions: 方向数组，每个方向代表一种排序方式

#   Returns:
#     合并后的数组，torch 张量
#   """

#   # 创建一个新张量来存储合并后的结果
#   merged_array = torch.zeros((len(directions), array.shape[0]), device=array.device)

#   # 遍历每个方向
#   for i, direction in enumerate(directions):
#     # 按照当前方向对数组进行排序
#     sorted_array, _ = torch.sort(array[:, direction], dim=1)

#     # 将排序后的数组存储到合并后的张量中
#     merged_array[i] = sorted_array

#   # 返回合并后的张量
#   return merged_array

# # 示例
# array = torch.tensor([1, 2, 3, 4])
# directions = torch.tensor([
#   [0, 1, 3, 2],
#   [0, 2, 3, 1],
#   [1, 0, 2, 3],
#   [1, 3, 2, 0],
#   [2, 3, 1, 0],
#   [2, 0, 1, 3],
#   [3, 2, 0, 1],
#   [3, 1, 0, 2],
# ])

# merged_array = rearrange_and_merge(array, directions)

# print(merged_array)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    # 生成干净的曲线数据
    x = np.linspace(0, 10, 1000)
    y_clean = np.sin(x)

    # 添加噪声
    noise = np.random.normal(0, 0.1, y_clean.shape)
    y_noisy = y_clean + noise

    # 计算傅里叶变换
    fft_clean = np.fft.fft(y_clean)
    fft_noisy = np.fft.fft(y_noisy)

    # 计算频谱的幅度和相位
    amplitude_clean = np.abs(fft_clean)
    phase_clean = np.angle(fft_clean)
    amplitude_noisy = np.abs(fft_noisy)
    phase_noisy = np.angle(fft_noisy)

    # 绘制幅度和相位曲线
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.plot(amplitude_clean)
    plt.title('Amplitude (Clean)')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 2)
    plt.plot(phase_clean)
    plt.title('Phase (Clean)')
    plt.xlabel('Frequency')
    plt.ylabel('Phase')

    plt.subplot(2, 2, 3)
    plt.plot(amplitude_noisy)
    plt.title('Amplitude (Noisy)')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 4)
    plt.plot(phase_noisy)
    plt.title('Phase (Noisy)')
    plt.xlabel('Frequency')
    plt.ylabel('Phase')

    plt.tight_layout()
    plt.savefig('test.png')

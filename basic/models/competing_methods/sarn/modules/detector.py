import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
class Detector(nn.Module):
    """Deep residual network with spatial attention for face SR.
    # Arguments:
        - n_ch: base convolution channels
        - down_steps: how many times to downsample in the encoder
        - res_depth: depth of residual layers in the main body
        - up_res_depth: depth of residual layers in each upsample block

    """

    def __init__(
            self,
            channels = 64,
            opt=None,
    ):
        super(Detector, self).__init__()
        self.opt = opt
        self.in_channels = self.opt.n_range*2
        # self.in_channels = 12
        self.detector = nn.Sequential()
        self.detector.append(nn.Conv2d(self.in_channels, channels, 3, 1, 1))
        self.detector.append(nn.ReLU())
        # self.detector.append(nn.Conv2d(channels, channels, 3, 1, 1))
        # self.detector.append(nn.ReLU())
        self.detector.append(nn.Conv2d(channels, channels, 3, 1, 1))
        self.detector.append(nn.AdaptiveAvgPool2d((1,1)))

        self.l1 = nn.Linear(channels,  self.in_channels)



    def forward(self, input):
        b, c, h, w = input.shape
        # device = input.device
        # x_n = torch.zeros((b, self.opt.n_sharp, w, h)).to(device)
        logit = self.detector(input)
        logit = torch.squeeze(logit,dim=-1)
        logit = torch.squeeze(logit, dim=-1)
        # logit = torch.softmax(self.l1(logit),dim=1)
        # maxval,index = torch.max(logit,dim=1)

        # logit = torch.sigmoid(self.l1(logit))
        # a, idx1 = torch.sort(logit, descending=True, dim=0)
        # idx1_top = idx1[:, 0:self.opt.n_sharp]
        # for i, data in enumerate(zip(input, idx1_top)):
        #     x_n[i, :, :, :] = torch.index_select(data[0], dim=0, index=data[1])

        logit_onehot = F.gumbel_softmax(logit, tau=1, hard=False, dim=-1)
        values, indices = logit_onehot.topk(self.opt.n_range, dim=1, largest=True, sorted=True)

        nonzero = torch.nonzero(logit_onehot)
        nonzero_col = nonzero[:,1]
        x_n = input[torch.arange(0,nonzero.shape[0]),nonzero_col]
        # x_n = input[torch.arange(0, b), index]
        x_n = torch.unsqueeze(x_n,dim=1)
        return x_n



def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)

def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU(0.1, 0.3)
    elif act == 'selu':
        return nn.SELU()
    elif act == 'celu':
        return nn.CELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    from thop import profile
    x = torch.randn((6,3,32,32,32)).cuda()
    y = torch.randn((3,12,6))
    from DConv.atten.submodules import DeformableAttnBlock, DeformableAttnBlock_FUSION
    method = DeformableAttnBlock_FUSION().cuda()
    flops, params = profile(method, inputs=(x,x,))
    print('flops: {}, params: {}'.format(flops, params))

    out = method(x,x)
    print(out.shape)

    # b, c, h, w = x.shape
    # k = 3
    # x_n = torch.zeros((b,k,w,h))
    # detetor = Detector()
    # logit = detetor(x)
    # a, idx1 = torch.sort(logit, descending=True,dim=0)
    # idx1_top = idx1[:,0:k]
    # for i,data in enumerate(zip(x,idx1_top)):
    #     x_n[i,:,:,:] = torch.index_select(data[0],dim=0,index=data[1])



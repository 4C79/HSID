import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
import math
from .model_util import *
from einops import rearrange

class Corrlayer(nn.Module):
    def __init__(self, dim=24):
        super(Corrlayer, self).__init__()
        self.res =  make_layer(ResidualBlockNoBN,2,mid_channels=dim)
        self.down = nn.Conv2d(dim*2,dim,kernel_size=1)

    def forward(self, x, y):
        xy = torch.cat((x,y),dim=1)
        xy = self.down(xy)
        xy = self.res(xy)
        return xy

class CrossAttention(nn.Module):
    def __init__(self, dim=24, num_heads=2, bias=True):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        # self.temperature = nn.Parameter(torch.ones(dim//num_heads,1,))
        self.k = nn.Conv2d(dim, dim, kernel_size=1,padding=0, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, padding=0,bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1,padding=0, bias=bias)
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1,padding=0, bias=bias)
        self.res = make_layer(ResidualBlockNoBN, 2, mid_channels=dim)

    def forward(self, x, y):
        b, c, h, w = x.shape
        k = self.k(y)
        v = self.v(y)
        q = self.q(x)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # out = self.project_out(out)
        # out = out + y
        out = self.res(out)
        out = out+y

        return out

class SpectralMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpectralMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim
    def forward(self,guide_input):
        b,c,h,w, = guide_input.shape
        guide_input = guide_input.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        guide_feature_spec = self.fc1(guide_input)
        guide_feature_spec = guide_feature_spec.view(b, h, w, self.output_dim).permute(0, 3, 1, 2)
        return guide_feature_spec

class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class RouteFuncMLP_2D(nn.Module):
    """
    The routing function for generating the calibration weights.
    """

    def __init__(self, c_in, ratio=1, kernels=[3,3], bn_eps=1e-5, bn_mmt=0.1):
        """
        Args:
            c_in (int): number of input channels.
            ratio (int): reduction ratio for the routing function.
            kernels (list): temporal kernel size of the stacked 1D convolutions
        """
        super(RouteFuncMLP_2D, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.a = nn.Conv2d(
            in_channels=c_in,
            out_channels=int(c_in // ratio),
            kernel_size=[kernels[0], 1],
            padding=[kernels[0] // 2, 0],
        )
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(
            in_channels=int(c_in // ratio),
            out_channels=c_in,
            kernel_size=[kernels[1], 1],
            padding=[kernels[1] // 2, 0],
            bias=False
        )
        self.b.skip_init = True
        self.b.weight.data.zero_()  # to make sure the initial values
        # for the output is 1.

    def forward(self, x):
        x = self.avgpool(x)
        x = self.a(x)
        # # x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x

class RouteFuncMLP(nn.Module):
    """
    The routing function for generating the calibration weights.
    """

    def __init__(self, c_in, ratio=1, kernels=[3,3], bn_eps=1e-5, bn_mmt=0.1):
        """
        Args:
            c_in (int): number of input channels.
            ratio (int): reduction ratio for the routing function.
            kernels (list): temporal kernel size of the stacked 1D convolutions
        """
        super(RouteFuncMLP, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in // ratio),
            kernel_size=[kernels[0], 1, 1],
            padding=[kernels[0] // 2, 0, 0],
        )
        self.bn = nn.BatchNorm3d(int(c_in // ratio), eps=bn_eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(
            in_channels=int(c_in // ratio),
            out_channels=c_in,
            kernel_size=[kernels[1], 1, 1],
            padding=[kernels[1] // 2, 0, 0],
            bias=False
        )
        self.b.skip_init = True
        self.b.weight.data.zero_()  # to make sure the initial values
        # for the output is 1.

    def forward(self, x):
        x = self.avgpool(x)
        x = self.a(x)
        # x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x

class TAdaConv2d_2d(nn.Module):
    """
    Performs temporally adaptive 2D convolution.
    Currently, only application on 5D tensors is supported, which makes TAdaConv2d
        essentially a 3D convolution with temporal kernel size of 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cal_dim="cin"):
        super(TAdaConv2d_2d, self).__init__()
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (list): kernel size of TAdaConv2d. 
            stride (list): stride for the convolution in TAdaConv2d.
            padding (list): padding for the convolution in TAdaConv2d.
            dilation (list): dilation of the convolution in TAdaConv2d.
            groups (int): number of groups for TAdaConv2d. 
            bias (bool): whether to use bias in TAdaConv2d.
        """

        assert cal_dim in ["cin", "cout"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cal_dim = cal_dim

        # base weights (W_b)
        self.weight = nn.Parameter(
            torch.Tensor(1, out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, alpha):
        """
        Args:
            x (tensor): feature to perform convolution on.
            alpha (tensor): calibration weight for the base weights.
                W_t = alpha_t * W_b
        """
        _,  c_out, c_in, kh, kw = self.weight.size()
        b, c_in, h, w = x.size()
        x = x.reshape(1, -1, h, w)


        if self.cal_dim == "cin":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, 1, C, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (alpha.unsqueeze(dim=1)* self.weight).reshape(-1, c_in // self.groups,kh, kw)
            # print(weight.shape)
        elif self.cal_dim == "cout":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, C, 1, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (alpha* self.weight).reshape(-1, c_in // self.groups, kh, kw)

        bias = None
        if self.bias is not None:
            # in the official implementation of TAda2D,
            # there is no bias term in the convs
            # hence the performance with bias is not validated
            bias = self.bias.repeat(b, 1).reshape(-1)
        output = F.conv2d(
            x, weight=weight, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups*b)

        output = output.view(b,  c_out, output.size(-2), output.size(-1))

        return output

    def __repr__(self):
        return f"TAdaConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, " + \
               f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, cal_dim=\"{self.cal_dim}\")"
class TAdaConv2d(nn.Module):
    """
    Performs temporally adaptive 2D convolution.
    Currently, only application on 5D tensors is supported, which makes TAdaConv2d
        essentially a 3D convolution with temporal kernel size of 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cal_dim="cin"):
        super(TAdaConv2d, self).__init__()
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (list): kernel size of TAdaConv2d. 
            stride (list): stride for the convolution in TAdaConv2d.
            padding (list): padding for the convolution in TAdaConv2d.
            dilation (list): dilation of the convolution in TAdaConv2d.
            groups (int): number of groups for TAdaConv2d. 
            bias (bool): whether to use bias in TAdaConv2d.
        """

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert cal_dim in ["cin", "cout"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cal_dim = cal_dim

        # base weights (W_b)
        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups, kernel_size[1], kernel_size[2])
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, alpha):
        """
        Args:
            x (tensor): feature to perform convolution on.
            alpha (tensor): calibration weight for the base weights.
                W_t = alpha_t * W_b
        """
        _, _, c_out, c_in, kh, kw = self.weight.size()
        b, c_in, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(1, -1, h, w)

        if self.cal_dim == "cin":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, 1, C, H(1), W(1)
            # corresponding to calibrating the input channel

            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(2) * self.weight).reshape(-1, c_in // self.groups, kh, kw)
        elif self.cal_dim == "cout":
            # w_alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, C, 1, H(1), W(1)
            # corresponding to calibrating the input channel
            weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(3) * self.weight).reshape(-1, c_in // self.groups, kh, kw)

        bias = None
        if self.bias is not None:
            # in the official implementation of TAda2D,
            # there is no bias term in the convs
            # hence the performance with bias is not validated
            bias = self.bias.repeat(b, t, 1).reshape(-1)
        output = F.conv2d(
            x, weight=weight, bias=bias, stride=self.stride[1:], padding=self.padding[1:],
            dilation=self.dilation[1:], groups=self.groups * b * t)

        output = output.view(b, t, c_out, output.size(-2), output.size(-1)).permute(0, 2, 1, 3, 4)

        return output

    def __repr__(self):
        return f"TAdaConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, " + \
               f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, cal_dim=\"{self.cal_dim}\")"


class TAdaConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TAdaConvBlock, self).__init__()
        self.b = TAdaConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.b_rf = RouteFuncMLP(c_in=in_channels)
    def forward(self,input):
        x = self.b(input, self.b_rf(input))
        return x

class TAdaConvBlock_2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TAdaConvBlock_2D, self).__init__()
        self.b = TAdaConv2d_2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.b_rf = RouteFuncMLP_2D(c_in=in_channels)
    def forward(self,input):
        x = self.b(input, self.b_rf(input))
        return x
class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h

class ConvGRU3D(nn.Module):
    def __init__(self, hidden_dim=24, input_dim=24, num_layers = 1, is_backward=False):
        super(ConvGRU3D, self).__init__()
        self.is_backward = is_backward
        self.conv_gru = ConvGRU(hidden_dim=hidden_dim,input_dim=hidden_dim)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.resblocks = make_layer(ResidualBlockNoBN, 1, mid_channels=hidden_dim)
        # self.conv = nn.Sequential(nn.Conv2d(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=3,padding=1),
        #                           self.lrelu,
        #                           nn.Conv2d(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=3,padding=1))
        if self.is_backward:
            self.fusion = nn.Conv2d(hidden_dim*2, hidden_dim, 3, 1, 1, bias=True)

        # self.convblock = nn.Sequential()
        # self.convblock.append(nn.Conv3d(in_channels=1,out_channels = hidden_dim//3,kernel_size=3,stride=1,padding=1))
        # self.convblock.append(self.lrelu)
        # self.convblock.append(nn.Conv3d(in_channels=hidden_dim//3,out_channels = 1,kernel_size=3,stride=1,padding=1))
        # self.convblock = make_layer(nn.Conv3d,num_layers=num_layers,in_channels=1,out_channels=1,kernel_size = 3,stride =1,padding=1)
    def forward(self, hidden_state, input, hidden_state_f=None):
        # hidden_state = self.conv(hidden_state)
        hidden_state = self.conv_gru(hidden_state,input)
        if self.is_backward and hidden_state_f is not None:
            hidden_state = self.lrelu(self.fusion(torch.cat([hidden_state, hidden_state_f], dim=1)))
        # hidden_state = self.convblock(hidden_state.unsqueeze(dim=1)).squeeze(dim = 1)


        return hidden_state


if __name__ == '__main__':

    input = torch.randn(size=(16,24,24,32,32))
    b = TAdaConv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
    b_rf = RouteFuncMLP(c_in=24)
    x = b(input, b_rf(input))
    print(x.shape)
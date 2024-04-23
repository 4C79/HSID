# from .blocks import *
# from models.modules.DIBD.DRB import UNet_ND_cell
import torch.nn.functional as F

from einops import rearrange

import torch
import torch.nn as nn
import copy

def stride_generator(N, reverse=False):
    strides = [1, 1,1,1,1]
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]

class Fusion(nn.Module):
    def __init__(self, dim=64, num_heads=4, bias=False):
        super(Fusion, self).__init__()
        # self.att1 = SelfAttention(dim=dim, num_heads=num_heads, bias=bias)
        # self.att2 = SelfAttention(dim=dim, num_heads=num_heads, bias=bias)
        self.att1 = CrossAttention_I(dim=dim, num_heads=num_heads, bias=bias)
        self.att2 = CrossAttention_I(dim=dim, num_heads=num_heads, bias=bias)

        # self.att3 = CrossAttention_I(dim=dim, num_heads=num_heads, bias=bias)
        # self.att4 = CrossAttention_I(dim=dim, num_heads=num_heads, bias=bias)


        self.out = nn.Conv2d(dim*2,dim,3,1,1)
    def forward(self, x, y):
        out1 = self.att1(x,y)
        out2 = self.att2(y,x)

        # out1 = self.att3(out1, out2)
        # out2 = self.att4(out2, out1)

        out = torch.cat((x+out1,y+out2),dim=1)

        out = self.out(out)

        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding, output_padding=stride // 2)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class BasicConv3d(nn.Module):
    def __init__(self, in_channels, channels,out_channels, kernel_size=3, stride=1, padding=1, transpose=False, act_norm=False,groups = 1):
        super(BasicConv3d, self).__init__()
        self.act_norm = act_norm
        # self.conv_in = nn.Conv3d(in_channels, channels, kernel_size=1, stride=1, padding=0,groups = groups)
        self.conv_in = nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1, groups=groups)
        if not transpose:
            self.conv = nn.Conv3d(channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups = groups)
        else:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding, output_padding=stride // 2)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv_in(x)
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y

class LKA(nn.Module):
    def __init__(self, C_hid, C_out, act_norm=True):
        super().__init__()
        self.act_norm = act_norm
        dim = C_hid
        # if C_hid == C_out:
        #     self.att = True
        # else:self.att = False
        # self.conv_in = nn.Conv2d(C_in, dim, kernel_size=1, stride=1, padding=0)
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 3, stride=1, padding=3, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, C_out, 1)
        self.norm = nn.GroupNorm(2, C_out)
        self.act = nn.LeakyReLU(0.2, inplace=True)


    def forward(self, x):

        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        if self.act_norm:
            attn = self.act(self.norm(attn))
        # if self.att:
        #     return attn*y
        # else:
        #     return attn
        return attn

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

class SelfAttention_spa(nn.Module):
    def __init__(self, dim=64, num_heads=4, bias=False):
        super(SelfAttention_spa, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        y = x
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn_tmp = (q.transpose(-2, -1) @ k)

        attn = attn_tmp * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v.transpose(-2, -1))

        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class SelfAttention_spec(nn.Module):
    def __init__(self, dim=64, num_heads=4, bias=False):
        super(SelfAttention_spec, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        y = x
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn_tmp = (q @ k.transpose(-2, -1))

        attn = attn_tmp * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class SelfAttention1D(nn.Module):
    def __init__(self, dim=64, num_heads=4, bias=False):
        super(SelfAttention1D, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv1d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv1d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.squeeze(x,-1)
        y = x
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        
        out = torch.unsqueeze(out,-1)
        
        return out

class ECABlock(nn.Module):
    def __init__(self, channel, gamma=2):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv1d(channel, channel // gamma, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // gamma, channel, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1)).unsqueeze(-1)
        y = y * x

        return y
    
class PVTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=16):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        # 实现上这里等价于一个卷积层
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # B, N, C = x.shape
        B,C,H,W = x.shape
        
        x = x.reshape(B,C,H*W) # 16,1,64,64
        
        B,C,N = x.shape # 16,1,4096
        
        x = x.permute(0,2,1) # 16,4096,1
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) # 这里x_.shape = (B, N/R^2, C)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        x = x.permute(0,2,1)
        x = x.reshape(B,C,H,W)

        return x

class SpaFreBlock(nn.Module):
    def __init__(self, C_in, channels, stride,weight_factor=0.):
        super(SpaFreBlock, self).__init__()
            
        self.weight_factor = torch.nn.Parameter(torch.tensor(weight_factor), requires_grad=True)
    
        self.conv_first = nn.Conv2d(C_in, 1, 1, 1, 0)
        # self.att1 = PVTAttention(dim=1,num_heads=1)
        # self.att2 = PVTAttention(dim=1,num_heads=1)
        self.att1 = SelfAttention_spa(dim=1, num_heads=1, bias=False)
        self.att2 = SelfAttention_spa(dim=1, num_heads=1, bias=False)
        self.act_1 = nn.LeakyReLU(0.2, inplace=True)
        self.act_2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # print("x: ", x.shape)
        _, C, _,_ = x.shape   # 16,31,64,64
        
        x_spa = self.conv_first(x) # 16,1,64,64
        
        x_spa_s1 = self.act_1(self.att1(x_spa)) # 16,1,64,64
        x_spa_s2 = self.act_2(self.att2(x_spa)) # 16,1,64,64
        
        # x_spa_r = x_spa.permute(0, 2, 3, 1) 
        
        # msF = torch.fft.rfft(x_spa_r+1e-8, norm='backward') # 16,1,64,32
        
        # msF_amp = torch.abs(msF)
        # msF_pha = torch.angle(msF)
        
        # msF_amp = msF_amp.permute(0, 3, 1, 2)
        # msF_pha = msF_pha.permute(0, 3, 1, 2)
        
        # amp_adj = x_spa_s1 + msF_amp
        # pha_adj = x_spa_s2 + msF_pha
        
        # amp_adj = amp_adj.permute(0, 2, 3, 1)
        # pha_adj = pha_adj.permute(0, 2, 3, 1)
        
        # real = amp_adj * torch.cos(pha_adj)+1e-8
        # imag = amp_adj * torch.sin(pha_adj)+1e-8
        # out = torch.complex(real, imag)+1e-8
        # out = torch.abs(torch.fft.irfft(out ,n=x_spa_r.size(-1), norm='backward'))
        
        # out = out.permute(0, 3, 1, 2)
        
        x_out = torch.tile(x_spa_s1, (1, C, 1, 1))
        
        x_out = self.weight_factor * x_out + x
        
        return x_out

class SpecFreBlock(nn.Module):
    def __init__(self, C_in, channels, stride,weight_factor=0.):
        super(SpecFreBlock, self).__init__()
        
        # self.stride = stride
        self.weight_factor = weight_factor
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.att1 = SelfAttention_spec(dim=channels, num_heads=1, bias=False)
        self.att2 = SelfAttention_spec(dim=channels, num_heads=1, bias=False)
        # self.conv_1 = nn.Conv2d(C_in, C_in, 1, 1, 0)
        # self.conv_2 = nn.Conv2d(C_in, C_in, 1, 1, 0)
        self.act_1 = nn.LeakyReLU(0.2, inplace=True)
        self.act_2 = nn.LeakyReLU(0.2, inplace=True)
        # self.amp_fuse_1 = nn.Sequential(BasicConv2d(channels, channels, 3, 1, 1))
        # self.pha_fuse_1 = nn.Sequential(BasicConv2d(channels, channels, 3, 1, 1))
        # self.down = nn.Conv2d(channels,channels,3,stride,1)
        self.weight_factor = torch.nn.Parameter(torch.tensor(weight_factor), requires_grad=True)

    def forward(self, x):
        # print("x: ", x.shape)
        _, _, H, W = x.shape
        
        # x_g = self.avg_pool(x)
        
        x_out = self.act_1(self.att1(x))
        # x_seq_2 = self.act_2(self.att2(x_g))
        
        # msF = torch.fft.rfft(x_g+1e-8, norm='backward')
        
        # msF_amp = torch.abs(msF)
        # msF_pha = torch.angle(msF)
        
        # amp_adj = x_seq_1 + msF_amp
        # pha_adj = x_seq_2 + msF_pha
        
        # real = amp_adj * torch.cos(pha_adj)+1e-8
        # imag = amp_adj * torch.sin(pha_adj)+1e-8
        # out = torch.complex(real, imag)+1e-8
        # out = torch.abs(torch.fft.irfft(out ,n=x_g.size(-1), norm='backward'))
        
        # x_out = x_seq_1.repeat(1, 1, H, W)
        
        x_out = self.weight_factor * x_out + x
        
        return x_out

class SelfAttention_I(nn.Module):
    def __init__(self, dim=1, num_heads=4, bias=False):
        super(SelfAttention_I, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = LKA(dim * 2, dim * 2)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = LKA(dim, dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class CrossAttention_I(nn.Module):
    def __init__(self, dim=64, num_heads=4, bias=False):
        super(CrossAttention_I, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = LKA(dim * 2, dim * 2)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = LKA(dim, dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class UNet_ND_cell(nn.Module):

    def __init__(self,in_nc=1,out_nc=1,channel =20):
        super(UNet_ND_cell, self).__init__()
        self.main = MainNet(in_nc, out_nc,channel=channel)
        # self.main2 = MainNet(in_nc=2, out_nc=2)
        self.out = nn.Conv3d(out_nc*2, out_nc, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        input = x
        out1 = self.main(input) + x
        cat1 = torch.cat([x, out1], dim=1)
        return self.out(cat1) + x
    
class MainNet(nn.Module):
    """B-DenseUNets"""
    def __init__(self, in_nc=12, out_nc=12,channel = 24):
        super(MainNet, self).__init__()
        lay=2
        feat = channel
        self.inc = nn.Sequential(
            single_conv(in_nc, feat*2),
            single_conv(feat*2, feat*2),
        )
        self.down1 = nn.Conv3d(feat*2, feat*2,kernel_size=(1,2,2),stride=(1,2,2),padding=0)
        self.conv1 = nn.Sequential(
            single_conv(feat*2, feat*4),
            RDB(feat*4, lay, feat),
        )
        self.down2 = nn.Conv3d(feat*4, feat*4,kernel_size=(1,2,2),stride=(1,2,2),padding=0)
        self.conv2 = nn.Sequential(
            single_conv(feat*4, feat*8),
            RDB(feat*8, lay+1, feat),
        )
        self.up1 = up(feat*8)
        self.conv3 = nn.Sequential(
            RDB(feat*4, lay+1, feat),
        )


        self.up2 = up(feat*4)
        self.conv4 = nn.Sequential(
            RDB(feat*2, lay, feat),
        )

        self.outc = outconv(feat*2, out_nc)

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)

        conv1 = self.conv1(down1)


        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)


        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)


        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, (1,2,2), stride=(1,2,2))

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv3d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv3d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class Encoder_spec_fre(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder_spec_fre, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            SpecFreBlock(C_in, C_hid, stride=strides[0]),
            *[SpecFreBlock(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        feat = []
        latent = self.enc[0](x)
        feat.append(latent)

        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
            feat.append(latent)

        return latent

class Encoder_spa_fre(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder_spa_fre, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            SpaFreBlock(C_in, C_hid, stride=strides[0]),
            *[SpaFreBlock(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        feat = []
        latent = self.enc[0](x)
        feat.append(latent)

        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
            feat.append(latent)

        return latent

class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        feat = []
        latent = self.enc[0](x)
        feat.append(latent)

        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
            feat.append(latent)

        return latent, feat


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            ConvSC(C_hid, C_hid, stride=strides[0], transpose=True),
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[1:-1]],
            ConvSC(C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None,enc2 = None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        # Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.dec[-1](hid)
        Y = self.readout(Y)
        return Y


class Mid_Xnet(nn.Module):
    def __init__(self,in_nc=1,out_nc=1,channels = 20):
        super(Mid_Xnet, self).__init__()
        self.net = UNet_ND_cell(in_nc=in_nc,out_nc=out_nc,channel= channels)

    def forward(self, x):
        # x = torch.transpose(x,1,2)
        z = self.net(x)
        # y = torch.transpose(z,1,2)
        return z

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入通道数与输出通道数不一致，需要进行调整
        self.adjust_dimensions = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.adjust_dimensions = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.adjust_dimensions(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual  # 残差连接
        out = self.relu(out)

        return out

class SimVP(nn.Module):
    def __init__(self,
                 hid_S = 12,
                hid_T = 12):
        super(SimVP, self).__init__()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        
        B = 31  # batch spectrum_number to processing
        C = 96  # channel
        T = 16
        N_S = 2
        sr_ratio = 2  # 下采样倍率
        
        # self.norm =  nn.LayerNorm(B) 
        # self.enc = Encoder()
        
        self.rb_amp_spec = ResidualBlock(C,C)
        self.rb_pha_spec = ResidualBlock(C,C)
        self.rb_amp_spa = ResidualBlock(C,C)
        self.rb_pha_spa = ResidualBlock(C,C)
        
        self.fuse_amp = nn.Conv2d(T,C,3,1,1)
        self.fuse_pha = nn.Conv2d(T,C,3,1,1)
        
        self.spec_conv_amp = nn.Conv2d(C,C, kernel_size=sr_ratio, stride=sr_ratio)
        self.spec_conv_pha = nn.Conv2d(C,C, kernel_size=sr_ratio, stride=sr_ratio)
        
        self.enc_spec_amp = Encoder_spec_fre(C, C, N_S)
        self.enc_spec_pha = Encoder_spec_fre(C, C, N_S)

        self.enc_spa_amp = Encoder_spa_fre(C//2, C//2, N_S)
        self.enc_spa_pha = Encoder_spa_fre(C//2, C//2, N_S)
        
        self.spa_trans_amp = nn.ConvTranspose2d(C,C, kernel_size=sr_ratio, stride=sr_ratio)
        self.spa_trans_pha = nn.ConvTranspose2d(C,C, kernel_size=sr_ratio, stride=sr_ratio)

        self.spec_trans_amp = nn.Conv2d(C,T,3,1,1)
        self.spec_trans_pha = nn.Conv2d(C,T,3,1,1)
        
        self.conv1 = nn.Conv2d(C,C//2,3,1,1)
        self.conv2 = nn.Conv2d(C,C//2,3,1,1)
        self.conv3 = nn.Conv2d(C//2,C,3,1,1)
        self.conv4 = nn.Conv2d(C//2,C,3,1,1)
        
        self.down = nn.Conv2d(B,B,3,1,1)
        
        # self.hid = Mid_Xnet(in_nc=B,out_nc=B,channels=T)

    def forward(self, HSI):
        N, B, H, W = HSI.shape # N,B,H,W
        
        # HSI = self.norm(HSI)
        
        x = HSI.reshape(N,H,W,B)
        
        msF = torch.fft.rfft2(x+1e-8, norm='backward') # N,B,H,W
        
        msF_amp = torch.abs(msF) # N,H,W,B/2+1
        msF_pha = torch.angle(msF) # N,H,W,B/2+1
        
        N,H,W,T = msF_amp.shape
        
        msF_amp = msF_amp.reshape(N,T,H,W)
        msF_pha = msF_pha.reshape(N,T,H,W)
        
        amp_res = self.fuse_amp(msF_amp) # N,C,H,W
        pha_res = self.fuse_pha(msF_pha) # N,C,H,W
        
        amp_rb_spec = self.rb_amp_spec(amp_res) # N,C,H,W
        pha_rb_spec = self.rb_pha_spec(pha_res) # N,C,H,W
        
        amp_rb_spec_r = self.spec_conv_amp(amp_rb_spec) # N,C,H/sr_ratio,W/sr_ratio
        pha_rb_spec_r = self.spec_conv_pha(pha_rb_spec) # N,C,H/sr_ratio,W/sr_ratio
        
        embed_amp_spec = self.enc_spec_amp(amp_rb_spec_r) # N,C,H/sr_ratio,W/sr_ratio
        embed_pha_spec = self.enc_spec_pha(pha_rb_spec_r) # N,C,H/sr_ratio,W/sr_ratio
        
        amp_rb_spa = self.rb_amp_spa(embed_amp_spec) # N,C,H/sr_ratio,W/sr_ratio
        pha_rb_spa = self.rb_pha_spa(embed_pha_spec) # N,C,H/sr_ratio,W/sr_ratio
        
        amp_rb_spa = self.conv1(amp_rb_spa)
        pha_rb_spa = self.conv2(pha_rb_spa)
        
        embed_amp_spa = self.enc_spa_amp(amp_rb_spa) # N,C,H/sr_ratio,W/sr_ratio
        embed_pha_spa = self.enc_spa_pha(pha_rb_spa) # N,C,H/sr_ratio,W/sr_ratio
        
        embed_amp_spa = self.conv3(embed_amp_spa)
        embed_pha_spa = self.conv4(embed_pha_spa)
        
        amp_dec_spa = self.spa_trans_amp(embed_amp_spa+amp_rb_spec_r)
        pha_dec_spa = self.spa_trans_pha(embed_pha_spa+pha_rb_spec_r)
        
        amp_dec_spec = self.spec_trans_amp(amp_dec_spa)
        pha_dec_spec = self.spec_trans_pha(pha_dec_spa)
        
        amp_dec_spec = amp_dec_spec.reshape(N,H,W,T)
        pha_dec_spec = pha_dec_spec.reshape(N,H,W,T)
        
        real = amp_dec_spec * torch.cos(pha_dec_spec)+1e-8
        imag = amp_dec_spec * torch.sin(pha_dec_spec)+1e-8
        
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out,  s=(W, B), norm='backward'))
        
        out = out.reshape(N,B,H,W)
        
        # out = torch.unsqueeze(out,dim=2)
        
        # hid = self.hid(out)
        
        # out = out + hid
    
        # out = torch.squeeze(out,dim=2)
        
        out = self.down(out)
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)

        return out + HSI


# class FIDNet_3(nn.Module):
#     def __init__(self,
#                  hid_S,
#                  hid_T,
#                  opt = None):
#         super(FIDNet_3, self).__init__()
#         self.fourierNet = SimVP(hid_S = hid_S,
#                                 hid_T = hid_T)
#     def forward(self,HSI):
#         device = HSI.device
#         # input = torch.unsqueeze(HSI, dim=4)
#         final_denoised = self.fourierNet(HSI)
#         # final_denoised = torch.squeeze(final_denoised, dim=4)

#         return final_denoised

class FIDNet_3(nn.Module):
    def __init__(self,
                 hid_S,
                 hid_T,
                 opt = None):
        super(FIDNet_3, self).__init__()
        self.fourierNet = SimVP(hid_S = hid_S,
                                hid_T = hid_T)

    def forward(self,HSI):
        N,B,H,W = HSI.shape
        device = HSI.device
        denoised_HSI = torch.zeros_like(HSI).to(device)
        if B % 31 == 0:
            num = B // 31
        else:
            num = B // 31 +1

        last = B%31
        for pos in range(num):
            if(pos <num-1 or last == 0):
                input = HSI[:,pos * 31:(pos+1)*31,:,:]
            else:
                input = HSI[:,B-31:B,:,:]

            # input = torch.unsqueeze(input,dim=2)

            final_denoised = self.fourierNet(input)

            # final_denoised = torch.squeeze(final_denoised,dim=2)
            
            if (last == 0 or pos< num -1):
                denoised_HSI[:, pos * 31:(pos+1)*31, :, :] = final_denoised
            else:
                denoised_HSI[:, B-last:B, :, :] = final_denoised[:,31-last:31,:,:]

        return denoised_HSI
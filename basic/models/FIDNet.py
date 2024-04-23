# from .blocks import *
# from models.modules.DIBD.DRB import UNet_ND_cell
import torch.nn.functional as F

from einops import rearrange

import torch
import torch.nn as nn
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

class SelfAttention(nn.Module):
    def __init__(self, dim=64, num_heads=4, bias=False):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
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

class FreBlock(nn.Module):
    def __init__(self, C_in, channels, stride):
        super(FreBlock, self).__init__()
        self.stride = stride
        self.fpre = nn.Conv2d(C_in, channels, 1, 1, 0)
        self.amp_fuse_1 = nn.Sequential(BasicConv2d(channels, channels, 3, 1, 1))
        self.pha_fuse_1 = nn.Sequential(BasicConv2d(channels, channels, 3, 1, 1))

        # self.amp_fuse_1 = nn.Sequential(BasicConv2d(channels, channels, 3, 1, 1),LKA(channels,channels))
        # self.pha_fuse_1 = nn.Sequential(BasicConv2d(channels, channels, 3, 1, 1),LKA(channels,channels))

        # self.amp_fuse_1 = BasicConv2d(channels, channels, 1, 1, 0)
        # self.pha_fuse_1 = BasicConv2d(channels, channels, 1, 1, 0)

        # self.post = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0),nn.Conv2d(channels, channels, 1, 1, 0),nn.Conv2d(channels, channels, 1, 1, 0))
        # self.post = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0))
        # if stride == 2:
        self.down = nn.Conv2d(channels,channels,3,stride,1)
            # self.down = nn.AvgPool2d(2)

    def forward(self, x):
        # print("x: ", x.shape)
        _, _, H, W = x.shape
        msF = torch.fft.rfft2(self.fpre(x)+1e-8, norm='backward')

        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        # print("msf_amp: ", msF_amp.shape)
        amp_res = self.amp_fuse_1(msF_amp)
        pha_res = self.pha_fuse_1(msF_pha)

        # print(amp_fuse.shape, msF_amp.shape)
        amp_fuse = amp_res + msF_amp

        pha_fuse = pha_res + msF_pha

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        # out = self.post(out)
        # out = out + x
        # if self.stride == 2:
        #     out = self.down(out)
        out = self.down(out)
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        # print("out: ", out.shape)
        return out

class FreBlock_plus(nn.Module):
    def __init__(self, C_in, channels, stride):
        super(FreBlock_plus, self).__init__()
        self.stride = stride

        self.fpre = nn.Conv2d(C_in, channels, 1, 1, 0)
        #
        # self.att1 = SelfAttention(dim=channels, num_heads=8, bias=False)
        # self.att2 = SelfAttention(dim=channels, num_heads=8, bias=False)

        self.att1 = SelfAttention_I(dim=channels, num_heads=4, bias=False)
        self.att2 = SelfAttention_I(dim=channels, num_heads=4, bias=False)

        self.amp_fuse_1 = nn.Sequential(BasicConv2d(channels, channels, 3, 1, 1))
        self.pha_fuse_1 = nn.Sequential(BasicConv2d(channels, channels, 3, 1, 1))

        # self.amp_fuse_2 = nn.Sequential(BasicConv2d(channels, channels, 3, 1, 1))
        # self.pha_fuse_2 = nn.Sequential(BasicConv2d(channels, channels, 3, 1, 1))
        # self.amp_fuse_1 = BasicConv2d(channels, channels, 1, 1, 0)
        # self.pha_fuse_1 = BasicConv2d(channels, channels, 1, 1, 0)

        # self.post = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0),nn.Conv2d(channels, channels, 1, 1, 0),nn.Conv2d(channels, channels, 1, 1, 0))
        # self.post = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0))
        # if stride == 2:
        self.down = nn.Conv2d(channels,channels,3,stride,1)
            # self.down = nn.AvgPool2d(2)

    def forward(self, x):
        # print("x: ", x.shape)
        _, _, H, W = x.shape
        msF = torch.fft.rfft2(self.fpre(x)+1e-8, norm='backward')

        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        # print("msf_amp: ", msF_amp.shape)
        amp_res = self.amp_fuse_1(msF_amp)
        pha_res = self.pha_fuse_1(msF_pha)

        amp_res = self.att1(amp_res)
        pha_res = self.att2(pha_res)

        # print(amp_fuse.shape, msF_amp.shape)
        amp_fuse = amp_res + msF_amp

        pha_fuse = pha_res + msF_pha

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        # out = self.post(out)
        # out = out + x
        # if self.stride == 2:
        #     out = self.down(out)
        out = self.down(out)
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        # print("out: ", out.shape)
        return out

class SelfAttention_I(nn.Module):
    def __init__(self, dim=64, num_heads=4, bias=False):
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


class Encoder_fre(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder_fre, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            FreBlock(C_in, C_hid, stride=strides[0]),
            *[FreBlock(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        feat = []
        latent = self.enc[0](x)
        feat.append(latent)

        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
            feat.append(latent)

        return latent, feat

class Encoder_fre_plus(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder_fre_plus, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            FreBlock_plus(C_in, C_hid, stride=strides[0]),
            *[FreBlock_plus(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        feat = []
        latent = self.enc[0](x)
        feat.append(latent)

        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
            feat.append(latent)

        return latent, feat

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
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None,enc2 = None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y


class Mid_Xnet(nn.Module):
    def __init__(self,in_nc=1,out_nc=1,channels = 20):
        super(Mid_Xnet, self).__init__()
        self.net = UNet_ND_cell(in_nc=in_nc,out_nc=out_nc,channel= channels)

    def forward(self, x):
        x = torch.transpose(x,1,2)
        z = self.net(x)
        y = torch.transpose(z,1,2)
        return y


class SimVP(nn.Module):
    def __init__(self,
                 hid_S,
                 hid_T):
        super(SimVP, self).__init__()
        T, C = 31,1
        N_S = 2
        self.up = nn.Conv2d(C,hid_S,3,1,1)
        self.enc = Encoder(hid_S, hid_S, N_S)
        self.enc_fre = Encoder_fre(hid_S, hid_S, N_S)
        self.fusion = Fusion(dim=hid_S)
        self.hid = Mid_Xnet(in_nc=hid_S,out_nc=hid_S,channels=hid_T )
        self.dec = Decoder(hid_S, C, N_S)

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)
        x= self.up(x)
        embed, feat = self.enc(x)
        skip = feat[0]

        embed_fre, feat_fre = self.enc_fre(x)


        embed = self.fusion(embed,embed_fre)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        # z = self.down(z)
        hid = self.hid(z)
        hid = hid.reshape(B * T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y


class FIDNet(nn.Module):
    def __init__(self,
                 hid_S,
                 hid_T,
                 opt = None):
        super(FIDNet, self).__init__()
        self.fourierNet = SimVP(hid_S = hid_S,
                                hid_T = hid_T)
    def forward(self,HSI):
        device = HSI.device
        input = torch.unsqueeze(HSI, dim=2)
        final_denoised = self.fourierNet(input)
        final_denoised = torch.squeeze(final_denoised, dim=2)

        return final_denoised

# class FIDNet(nn.Module):
#     def __init__(self,opt = None):
#         super(FIDNet, self).__init__()
#         self.fourierNet = SimVP()

#     def forward(self,HSI):
#         N,B,H,W = HSI.shape
#         device = HSI.device
#         denoised_HSI = torch.zeros_like(HSI).to(device)
#         if B % 31 == 0:
#             num = B // 31
#         else:
#             num = B // 31 +1

#         last = B%31
#         for pos in range(num):
#             if(pos <num-1 or last == 0):
#                 input = HSI[:,pos * 31:(pos+1)*31,:,:]
#             else:
#                 input = HSI[:,B-31:B,:,:]

#             input = torch.unsqueeze(input,dim=2)

#             final_denoised = self.fourierNet(input)

#             final_denoised = torch.squeeze(final_denoised,dim=2)
#             if (last == 0 or pos< num -1):
#                 denoised_HSI[:, pos * 31:(pos+1)*31, :, :] = final_denoised
#             else:
#                 denoised_HSI[:, B-last:B, :, :] = final_denoised[:,31-last:31,:,:]

#         return denoised_HSI
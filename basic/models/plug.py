import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Four(nn.Module):
    def __init__(self,hid_channel=12,num_heads=4, bias=False):
        super(Four, self).__init__()
        
        self.out = nn.Conv2d(hid_channel*2,hid_channel,3,1,1)
        self.conv_gru = ConvGRU(hidden_dim=hid_channel,input_dim=hid_channel)
        self.att1 = CrossAttention_I(dim=hid_channel, num_heads=num_heads, bias=bias)
        self.att2 = CrossAttention_I(dim=hid_channel, num_heads=num_heads, bias=bias)
        self.amp_fuse_1 = nn.Sequential(BasicConv2d(hid_channel, hid_channel, 3, 1, 1))
        self.pha_fuse_1 = nn.Sequential(BasicConv2d(hid_channel, hid_channel, 3, 1, 1))
        self.amp_fuse_2 = nn.Sequential(BasicConv2d(hid_channel, hid_channel, 3, 1, 1))
        self.pha_fuse_2 = nn.Sequential(BasicConv2d(hid_channel, hid_channel, 3, 1, 1))
        
    def forward(self,band_fea,hid_fea):
        
        _, _, H, W = band_fea.shape
        
        fea_msF = torch.fft.rfft2(band_fea+1e-8, norm='backward') # N,B,H,W
        hid_msF = torch.fft.rfft2(hid_fea+1e-8, norm='backward') # N,B,H,W

        fea_msF_amp = torch.abs(fea_msF) # N,H,W,B/2+1
        fea_msF_pha = torch.angle(fea_msF) # N,H,W,B/2+1
        
        fea_msF_amp = fea_msF_amp + self.amp_fuse_1(fea_msF_amp)
        fea_msF_pha = fea_msF_pha + self.pha_fuse_1(fea_msF_pha)
        
        hid_msF_amp = torch.abs(hid_msF) # N,H,W,B/2+1
        hid_msF_pha = torch.angle(hid_msF) # N,H,W,B/2+1
        
        hid_msF_amp = hid_msF_amp + self.amp_fuse_2(hid_msF_amp)
        hid_msF_pha = hid_msF_pha + self.pha_fuse_2(hid_msF_pha)
        
        # 交叉完后，再次卷积一次乘回来
        # 只在通道层面做交叉注意力是否合理 
        amp = self.att1(hid_msF_amp,fea_msF_amp)
        pha = self.att2(hid_msF_pha,fea_msF_pha)
    
        real = amp * torch.cos(pha)+1e-8
        imag = amp * torch.sin(pha)+1e-8
        out = torch.complex(real, imag)+1e-8          # B,C,H,W  
        out = torch.abs(torch.fft.irfft2(out,  s=(H,W), norm='backward'))   # B,C,H,W  
        
        hidden_state = self.conv_gru(out,hid_fea)
        
        out = torch.cat((band_fea,hidden_state),1)
        
        out = self.out(out)

        return out
    
class Plug(nn.Module):
    def __init__(self):
        super(Plug, self).__init__()
        
        hid_channel = 12
        self.hid_channel = hid_channel
        N_S = 2
        
        # 特征提取器 后期修改
        self.conv1 = nn.Conv2d(1,hid_channel,3,1,1)
        
        self.hid = Mid_Xnet(in_nc=hid_channel,out_nc=hid_channel,channels=hid_channel)
        
        self.four = Four(hid_channel)
        
        self.dec = Decoder(hid_channel, 1, N_S)
        
    def forward(self, HSI):
        
        N, B, H, W = HSI.shape # N,B,H,W
        device = HSI.device
        
        input = HSI

        feature_list = []
        
        # feature_amp = torch.zeros(N, B, self.hid_channel, H, W).to(device)
        # feature_pha = torch.zeros(N, B, self.hid_channel, H, W).to(device)
        hid_fea = torch.zeros(N, self.hid_channel, H, W).to(device)
        
        for band_idx in range(B):
            band_info = input[:, band_idx:band_idx+1, :, :]
            
            # 这里是否应该写到一个模块中，要不然公用一个卷积是否是共享了一套参数？
            band_fea = self.conv1(band_info)
            
            hid_fea = self.four(band_fea,hid_fea)
            
            feature_list.append(hid_fea)
            
        stacked_feature = torch.stack(feature_list, dim=1) # N,B,C,H,W
        
        hid = self.hid(stacked_feature)
        
        N, B, C_, H_, W_ = hid.shape
        
        hid = hid.reshape(N * B, C_, H_, W_)

        out = self.dec(hid)
        
        out = out.reshape(N , B, 1, H_, W_)
        
        out = torch.squeeze(out,dim=2)
        
        out = out + input
        
        return out



class Mid_Xnet(nn.Module):
    def __init__(self,in_nc=1,out_nc=1,channels = 20):
        super(Mid_Xnet, self).__init__()
        self.net = UNet_ND_cell(in_nc=in_nc,out_nc=out_nc,channel= channels)

    def forward(self, x):
        x = torch.transpose(x,1,2)
        z = self.net(x)
        y = torch.transpose(z,1,2)
        return y
    
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
    
def stride_generator(N, reverse=False):
    strides = [1, 1,1,1,1]
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]
    
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
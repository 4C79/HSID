# Code Implementation of the MambaIR Model
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
# from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat


import numpy as np


NEG_INF = -1000000

class SpatialChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=8):
        super(SpatialChannelAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = nn.AdaptiveAvgPool2d(1)(x)  # Efficient spatial information extraction
        spatial_attention = self.conv(avg_pool)  # Spatial branch
        channel_attention = self.conv(x)  # Reuse convolution for channel branch
        return x * spatial_attention * channel_attention



class CAB_1(nn.Module):
    def __init__(self, num_feat, is_light_sr=False, squeeze_factor=8):
        super(CAB, self).__init__()
        if is_light_sr:
            squeeze_factor = 16  # Adjust squeeze factor for light SR tasks
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // 2, 3, 1, 1),
            nn.GELU(),  # Consider GELU for a slight performance boost
            nn.Conv2d(num_feat // 2, num_feat, 3, 1, 1),
            SpatialChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=4): # ori 16
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=10):
        super(CAB, self).__init__()
        if is_light_sr: # a larger compression ratio is used for light-SR
            compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        group_size = (H, W)
        B_, N, C = x.shape
        assert H * W == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SS3D(nn.Module):
    def __init__(
            self,
            d_model,
            zig_path,
            zig_back,
            layer_book = 0,
            zigzag_path = None,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.layer_book = layer_book
        self.zigzag_path = zig_path
        self.zig_back = zig_back
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # self.conv2d = nn.Conv2d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     groups=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=d_conv,
        #     padding=(d_conv - 1) // 2,
        #     **factory_kwargs,
        # )
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=8, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=8, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor,zig_path,zig_back):
        B, C, H, W = x.shape
        L = H * W
        K = 8
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L) # [1, 2, 192, 262144]
        # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # [1, 4, 192, 262144]
        
        x_l = x.view(B, -1, L)
        
        x_l = self.act(self.conv1d(x_l))
                
        zigzag_mask = zig_path

        xs = torch.stack([x_l[:,:,zigzag_mask[0]].contiguous() ,
                            x_l[:,:,zigzag_mask[1]].contiguous() ,
                            x_l[:,:,zigzag_mask[2]].contiguous() ,
                            x_l[:,:,zigzag_mask[3]].contiguous() ,
                            x_l[:,:,zigzag_mask[4]].contiguous() ,
                            x_l[:,:,zigzag_mask[5]].contiguous() ,
                            x_l[:,:,zigzag_mask[6]].contiguous() ,
                            x_l[:,:,zigzag_mask[7]].contiguous() ,
                            ],dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) # [1, 4, 38, 262144] ,[4, 38, 192]
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) # [1, 4, 6, 262144],[1, 4, 16, 262144]
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) # [1, 4, 192, 262144] , [4, 192, 6]
        xs = xs.float().view(B, -1, L) # [1, 768, 262144]
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l) [1, 768, 262144]
        Bs = Bs.float().view(B, K, -1, L) # [1, 4, 16, 262144]
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l) [1, 4, 16, 262144]
        Ds = self.Ds.float().view(-1) # [1, 4, 16, 262144] 
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state) # [768, 16]
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)x [768]
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        
        # 映射回去
        zz_paths_rev = zig_back
        

        out_y = torch.stack([out_y[:,0,:,zz_paths_rev[0]].contiguous() ,
                        out_y[:,1,:,zz_paths_rev[1]].contiguous() ,
                        out_y[:,2,:,zz_paths_rev[2]].contiguous() ,
                        out_y[:,3,:,zz_paths_rev[3]].contiguous() ,
                        out_y[:,0,:,zz_paths_rev[4]].contiguous() ,
                        out_y[:,1,:,zz_paths_rev[5]].contiguous() ,
                        out_y[:,2,:,zz_paths_rev[6]].contiguous() ,
                        out_y[:,3,:,zz_paths_rev[7]].contiguous() ,
                        ],dim=1)
            
        return out_y

    def reverse3d(self,x):
        B, H, W, C = x.shape
        for i in range(H):
            for j in range(W):
                if (i + j) % 2 == 1:
                    x[:,i,j,:] = x[:,i,j,::-1]
        return x
    
    def reverse3d_optimized(self,x):
        """
        对 GPU 上的 [B, H, W, C] 维度矩阵进行反转，其中 (H+W)%2 == 1 的位置被反转

        Args:
            x: 输入张量，维度为 [B, H, W, C]，且在 GPU 上

        Returns:
            y: 输出张量，与输入张量同维度，(H+W)%2 == 1 的位置数据被反转
        """

        # assert x.is_cuda, "Input tensor must be on GPU"

        b, h, w, c = x.shape

        # 计算 H+W 并转换为一维索引
        hw_sum = torch.arange(0, h * w, device=x.device).view(h, w) + torch.arange(0, b, device=x.device).view(b, 1, 1)
        # 奇偶性标记
        parity = hw_sum % 2

        # 对 (H+W)%2 == 1 的位置进行反转
        y = x.clone()
        y[parity == 1] = torch.flip(y[parity == 1], dims=[-1])

        return y

    def forward(self, x: torch.Tensor, zig_path,zig_back,**kwargs):
        B, H, W, C = x.shape
        
        x_t = self.reverse3d_optimized(x)
        
        xz = self.in_proj(x_t)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        
        # x = self.act(self.conv2d(x))
        
        y = self.forward_core(x,zig_path,zig_back)
        assert y.dtype == torch.float32
        y = torch.sum(y, dim=1)
        
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        
        out = self.reverse3d_optimized(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        return out


    
class SS3D_8(nn.Module):
    def __init__(
            self,
            d_model,
            zig_path,
            zig_back,
            layer_book = 0,
            zigzag_path = None,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.layer_book = layer_book
        self.zigzag_path = zig_path
        self.zig_back = zig_back
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # self.conv2d = nn.Conv2d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     groups=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=d_conv,
        #     padding=(d_conv - 1) // 2,
        #     **factory_kwargs,
        # )
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor,zig_path,zig_back):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L) # [1, 2, 192, 262144]
        # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # [1, 4, 192, 262144]
        
        x_l = x.view(B, -1, L)
        
        x_l = self.act(self.conv1d(x_l))
                
        zigzag_mask = zig_path
        
        if self.layer_book % 2 == 0:
            xs = torch.stack([x_l[:,:,zigzag_mask[0]].contiguous() ,
                            x_l[:,:,zigzag_mask[1]].contiguous() ,
                            x_l[:,:,zigzag_mask[2]].contiguous() ,
                            x_l[:,:,zigzag_mask[3]].contiguous() ,
                            ],dim=1)
        else:
            xs = torch.stack([x_l[:,:,zigzag_mask[4]].contiguous() ,
                          x_l[:,:,zigzag_mask[5]].contiguous() ,
                          x_l[:,:,zigzag_mask[6]].contiguous() ,
                          x_l[:,:,zigzag_mask[7]].contiguous() ,
                            ],dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) # [1, 4, 38, 262144] ,[4, 38, 192]
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) # [1, 4, 6, 262144],[1, 4, 16, 262144]
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) # [1, 4, 192, 262144] , [4, 192, 6]
        xs = xs.float().view(B, -1, L) # [1, 768, 262144]
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l) [1, 768, 262144]
        Bs = Bs.float().view(B, K, -1, L) # [1, 4, 16, 262144]
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l) [1, 4, 16, 262144]
        Ds = self.Ds.float().view(-1) # [1, 4, 16, 262144] 
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state) # [768, 16]
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)x [768]
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        
        # 映射回去
        zz_paths_rev = zig_back
        
        if self.layer_book % 2 == 0:
            out_y = torch.stack([out_y[:,0,:,zz_paths_rev[0]].contiguous() ,
                        out_y[:,1,:,zz_paths_rev[1]].contiguous() ,
                        out_y[:,2,:,zz_paths_rev[2]].contiguous() ,
                        out_y[:,3,:,zz_paths_rev[3]].contiguous() ,
                        ],dim=1)
        else:
            out_y = torch.stack([out_y[:,0,:,zz_paths_rev[4]].contiguous() ,
                        out_y[:,1,:,zz_paths_rev[5]].contiguous() ,
                        out_y[:,2,:,zz_paths_rev[6]].contiguous() ,
                        out_y[:,3,:,zz_paths_rev[7]].contiguous() ,
                        ],dim=1)
            
        return out_y

    def reverse3d(self,x):
        B, H, W, C = x.shape
        for i in range(H):
            for j in range(W):
                if (i + j) % 2 == 1:
                    x[:,i,j,:] = x[:,i,j,::-1]
        return x
    
    def reverse3d_optimized(self,x):
        """
        对 GPU 上的 [B, H, W, C] 维度矩阵进行反转，其中 (H+W)%2 == 1 的位置被反转

        Args:
            x: 输入张量，维度为 [B, H, W, C]，且在 GPU 上

        Returns:
            y: 输出张量，与输入张量同维度，(H+W)%2 == 1 的位置数据被反转
        """

        # assert x.is_cuda, "Input tensor must be on GPU"

        b, h, w, c = x.shape

        # 计算 H+W 并转换为一维索引
        hw_sum = torch.arange(0, h * w, device=x.device).view(h, w) + torch.arange(0, b, device=x.device).view(b, 1, 1)
        # 奇偶性标记
        parity = hw_sum % 2

        # 对 (H+W)%2 == 1 的位置进行反转
        y = x.clone()
        y[parity == 1] = torch.flip(y[parity == 1], dims=[-1])

        return y

    def forward(self, x: torch.Tensor, zig_path,zig_back,**kwargs):
        B, H, W, C = x.shape
        
        # x = self.reverse3d_optimized(x)
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        
        # x = self.act(self.conv2d(x))
        
        y = self.forward_core(x,zig_path,zig_back)
        assert y.dtype == torch.float32
        y = torch.sum(y, dim=1)
        
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        
        # out = self.reverse3d_optimized(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L) # [1, 2, 192, 262144]
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # [1, 4, 192, 262144]

        x_dbl = torch.einsum("b k d l, k c d -> b k  c l", xs.view(B, K, -1, L), self.x_proj_weight) # [1, 4, 38, 262144] ,[4, 38, 192]
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) # [1, 4, 6, 262144],[1, 4, 16, 262144]
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) # [1, 4, 192, 262144] , [4, 192, 6]
        xs = xs.float().view(B, -1, L) # [1, 768, 262144]
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l) [1, 768, 262144]
        Bs = Bs.float().view(B, K, -1, L) # [1, 4, 16, 262144]
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l) [1, 4, 16, 262144]
        Ds = self.Ds.float().view(-1) # [1, 4, 16, 262144] 
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state) # [768, 16]
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)x [768]
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            layer_book: int =0,
            hidden_dim: int = 0,
            zig_back:list = None,
            zig_path:list = None,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS3D_8(layer_book=layer_book,d_model=hidden_dim,zig_path=zig_path,zig_back=zig_back,d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim,is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))



    def forward(self, input, x_size,zig_path,zig_back):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x,zig_path,zig_back))
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class BasicLayer(nn.Module):
    """ The Basic MambaIR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 zig_path,
                 zig_back,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,is_light_sr=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.zig_path=zig_path,
        self.zig_back=zig_back,
        self.depth = depth
        self.mlp_ratio=mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock(
                layer_book = i,
                hidden_dim=dim,
                zig_path=zig_path,
                zig_back=zig_back,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,
                input_resolution=input_resolution,is_light_sr=is_light_sr))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size,zig_path,zig_back):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size,zig_path,zig_back)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class MambaDe(nn.Module):
    r""" MambaIR Model
           A PyTorch impl of : `A Simple Baseline for Image Restoration with State Space Model `.

       Args:
           img_size (int | tuple(int)): Input image size. Default 64
           patch_size (int | tuple(int)): Patch size. Default: 1
           in_chans (int): Number of input image channels. Default: 3
           embed_dim (int): Patch embedding dimension. Default: 96
           d_state (int): num of hidden state in the state space model. Default: 16
           depths (tuple(int)): Depth of each RSSG
           drop_rate (float): Dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
           img_range: Image range. 1. or 255.
           upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
           resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
       """
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=1,
                 embed_dim=96,
                 num_feat = 64,
                 depths=(6, 6, 6, 6),
                 drop_rate=0.,
                 d_state = 16,
                 mlp_ratio=2.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super(MambaDe, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio=mlp_ratio

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim


        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = True if self.upsampler=='pixelshuffledirect' else False
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # 6-layer
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                zig_path=None,
                zig_back=None,
                depth=depths[i_layer],
                d_state = d_state,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                is_light_sr = self.is_light_sr
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x,zig_path,zig_back):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x) # N,L,C

        x = self.pos_drop(x)


        for layer in self.layers:
            x = layer(x, x_size,zig_path,zig_back)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x,zig_path,zig_back):
        self.mean = self.mean.type_as(x)
        
        # zig_path = self.zigzag_path(x.shape[2])
        # zig_back = [self.reverse_permut_np(_) for _ in zig_path]
        
        x = (x - self.mean) * self.img_range

        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first,zig_path,zig_back)) + x_first
        x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x
    
    def zigzag_path(self,N):

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

    def reverse_permut_np(self,permutation):
        n = len(permutation)
        reverse = np.array([0] * n)
        for i in range(n):
            reverse[permutation[i]] = i
        return reverse

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


class rnnMamba(nn.Module):
    def __init__(self,
                img_size=64,
                patch_size=1,
                in_chans=1,
                embed_dim=96,
                depths=(3, 3, 3),
                drop_rate=0.,
                d_state = 16,
                mlp_ratio=2.,
                drop_path_rate=0.1,
                resi_connection='3conv'):
        
        super(rnnMamba, self).__init__()
        
        self.mmd = MambaDe(img_size=img_size,
                            patch_size=patch_size,
                            in_chans=in_chans,
                            embed_dim=embed_dim,
                            depths=depths,
                            drop_rate=drop_rate,
                            d_state = d_state,
                            mlp_ratio=mlp_ratio,
                            drop_path_rate=drop_path_rate,
                            resi_connection=resi_connection)

    def forward(self,HSI):
        
        N,B,H,W = HSI.shape
        device = HSI.device
        denoised_HSI = torch.zeros_like(HSI).to(device)
        
        if B % 3 == 0:
            num = B // 3
        else:
            num = B // 3 +1

        last = B%3
        for pos in range(num):
            if(pos <num-1 or last == 0):
                input = HSI[:,pos * 3:(pos+1)*3,:,:]
            else:
                input = HSI[:,B-3:B,:,:]

            final_denoised = self.mmd(input)
            
            if (last == 0 or pos< num -1):
                denoised_HSI[:, pos * 3:(pos+1)*3, :, :] = final_denoised
            else:
                denoised_HSI[:, B-last:B, :, :] = final_denoised[:,3-last:3,:,:]

        return denoised_HSI

class ResidualGroup(nn.Module):
    """Residual State Space Group (RSSG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 zig_path,
                 zig_back,
                 d_state=16,
                 mlp_ratio=4.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection='1conv',
                 is_light_sr = False):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            zig_path=zig_path,
            zig_back=zig_back,
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr = is_light_sr)

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size,zig_path,zig_back):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size,zig_path,zig_back), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops



class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)



class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

# import numpy as np

def zigzag_path(N):

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

if __name__ == "__main__":
        
    def minmax_normalize(array):    
        amin = np.min(array)
        amax = np.max(array)
        return (array - amin) / (amax - amin)
    
    from torchvision.utils import make_grid
    from torch.nn import init

    import matplotlib.pyplot as plt
    from PIL import Image
    
    model = MambaDe(
        img_size=64,
        patch_size=1,
        in_chans=1,
        embed_dim=48,
        depths=(4,4,4),
        drop_rate=0.,
        d_state = 16,
        mlp_ratio=2.,
        drop_path_rate=0.1,
        resi_connection='3conv')
    
    x = torch.ones(1, 1, 256, 256).cuda("cuda:0")
    x.requires_grad = True
    zigzag_p = zigzag_path(256).cuda("cuda:0")
    y = model(x,zigzag_p, [reverse_permut_np(_).cuda("cuda:0") for _ in zigzag_p]).cuda("cuda:0")
    y = y * 255
    mask = torch.zeros_like(y, dtype=torch.float)
    mask[0, 0, 128, 128] = 1
    y = y * mask
    torch.autograd.backward(y, x)
    image = torch.cat([x, y, x.grad], dim=0)
    # image = make_grid(image)
    # image = image / image.max()
    print(image.shape)
    image = image[:,0,:,:]
    result_array = image.permute(1, 2, 0).detach().numpy()
    print(result_array.shape)
    # result_array = np.sum(result_array, axis=-1, keepdims=True)
    result_array = result_array[..., -1]
    print(result_array.shape)
    
    # result_array = 1 - minmax_normalize(result_array)
    
    result_array = (result_array * 255).astype(np.uint8)
    result_array = result_array.squeeze()
    result_array = Image.fromarray(result_array, mode='L')
    plt.imshow(result_array,cmap='viridis')
    plt.savefig("erf_mmd.png")
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


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


class GSAttention(nn.Module):
    """global spectral attention (GSA)

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads
        bias (bool): If True, add a learnable bias to projection
        b : batch_size
        c : channels
    """

    def __init__(self, dim, num_heads, bias):
        super(GSAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # self.mask = nn.Parameter(torch.rand(dim//num_heads,dim//num_heads))

        self.q_mask = nn.Parameter(torch.rand(dim // num_heads, dim // num_heads))
        self.k_mask = nn.Parameter(torch.rand(dim // num_heads, dim // num_heads))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = q.transpose(-1, -2)
        q = torch.matmul(q, self.q_mask)  # 进行矩阵乘法
        q = q.transpose(-1, -2)
        q = torch.nn.functional.normalize(q, dim=-1)
        self.q_mask.data = torch.clamp(self.q_mask.data, 0, 1)

        k = k.transpose(-1, -2)
        k = torch.matmul(k, self.k_mask)
        k = k.transpose(-1, -2)
        k = torch.nn.functional.normalize(k, dim=-1)
        self.k_mask.data = torch.clamp(self.k_mask.data, 0, 1)

        attn = (q @ k.transpose(-2, -1))

        attn = attn * self.temperature

        # attn = attn * self.mask

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

    def flops(self, patchresolution):
        flops = 0
        H, W, C = patchresolution
        flops += H * C * W * C
        flops += C * C * H * W
        return flops

   
class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
                
        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5
                
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
                
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads *3, N).chunk(3, dim=2) # (B, num_heads, head_dim, N)
        
        attn = (k.transpose(-1, -2) @ q) * self.scale
        
        attn = attn.softmax(dim=-2) # (B, h, N, N)
        attn = self.attn_drop(attn)
        
        x = (v @ attn).reshape(B, C, H, W)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b*c, 1, h, w), self.weights, stride=1, padding=self.kernel_size//2)        
        return x.reshape(b, c*9, h*w)

class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size//2)        
        return x

class StokenAttention(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.n_iter = n_iter
        self.stoken_size = stoken_size
                
        self.scale = dim ** - 0.5
        
        self.unfold = Unfold(3)
        self.fold = Fold(3)
        
        self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
       
        
    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size
        
        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            
        _, _, H, W = x.shape
        
        hh, ww = H//h, W//w
        
        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww)) # (B, C, hh, ww)
        
        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh*ww, h*w, C)
        
        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale # (B, hh*ww, h*w, 9)
                
                affinity_matrix = affinity_matrix.softmax(-1) # (B, hh*ww, h*w, 9)
               
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)
               
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
                    
                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
                    
                    stoken_features = stoken_features/(affinity_matrix_sum + 1e-12) # (B, C, hh, ww)
                    
        
        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
       
        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
        
        stoken_features = stoken_features/(affinity_matrix_sum.detach() + 1e-12) # (B, C, hh, ww)
        
        stoken_features = self.stoken_refine(stoken_features)
        
        
        stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9) # (B, hh*ww, C, 9)
       
        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2) # (B, hh*ww, C, h*w)
       
        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
                     
        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]
        
        return pixel_features
    
    
    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        stoken_features = self.stoken_refine(stoken_features)        
        return stoken_features
        
    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)

class SSMA(nn.Module):
    r"""  Transformer Block:Spatial-Spectral Multi-head self-Attention (SSMA)

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, inp_channels,dim, input_resolution, num_heads, window_size=7, shift_size=0, drop_path=0.0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU, bias=False):
        super(SSMA, self).__init__()
        self.dim = dim
        self.inp_channels= inp_channels
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.attn = StokenAttention(dim, stoken_size=stoken_size, 
                                    n_iter=n_iter,                                     
                                    num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                    attn_drop=attn_drop, proj_drop=drop)  

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.num_heads = num_heads

        self.spectral_attn = GSAttention(dim, num_heads, bias)



    def forward(self, x):

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        x = self.superAttn(x)

        # # cyclic shift
        # if self.shift_size > 0:
        #     shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        # else:
        #     shifted_x = x

        # # partition windows
        # x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        # x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # if self.input_resolution == [H, W]:  # non-local speatial attention
        #     attn_windows = self.attn(x_windows, mask=self.attn_mask)
        # else:
        #     attn_windows = self.attn(x_windows, mask=self.calculate_mask([H, W]).to(x.device))

        # # merge windows
        # attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # # reverse cyclic shift
        # if self.shift_size > 0:
        #     x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        # else:
        #     x = shifted_x

        x = x.view(B, H * W, C)

        x = x.transpose(1, 2).view(B, C, H, W)

        x = self.spectral_attn(x)  # global spectral attention

        x = x.flatten(2).transpose(1, 2)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.transpose(1, 2).view(B, C, H, W)
        return x


class SMSBlock(nn.Module):
    """
        residual spatial-spectral block (RSSB).
        Args:
            dim (int, optional): Embedding  dim of features. Defaults to 90.
            window_size (int, optional): window size of non-local spatial attention. Defaults to 8.
            depth (int, optional): numbers of Transformer block at this layer. Defaults to 6.
            num_head (int, optional):Number of attention heads. Defaults to 6.
            mlp_ratio (int, optional):  Ratio of mlp dim. Defaults to 2.
            qkv_bias (bool, optional): Learnable bias to query, key, value. Defaults to True.
            qk_scale (_type_, optional): The qk scale in non-local spatial attention. Defaults to None.
            drop_path (float, optional): drop_rate. Defaults to 0.0.
            bias (bool, optional): Defaults to False.
    """

    def __init__(self,
                 dim=90,
                 inp_channels=31,
                 window_size=8,
                 depth=6,
                 num_head=6,
                 mlp_ratio=2,
                 qkv_bias=True, qk_scale=None,
                 drop_path=0.0,
                 bias=False):
        super(SMSBlock, self).__init__()
        self.smsblock = nn.Sequential(
            *[SSMA(dim=dim,inp_channels=inp_channels, input_resolution=[64, 64], num_heads=num_head, window_size=window_size,
                   shift_size=0 if (i % 2 == 0) else window_size // 2,
                   mlp_ratio=mlp_ratio,
                   drop_path=drop_path[i],
                   qkv_bias=qkv_bias, qk_scale=qk_scale, bias=bias)
              for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        out = self.smsblock(x)
        out = self.conv(out) + x
        return out


class SST(nn.Module):
    """SST
     Spatial-Spectral Transformer for Hyperspectral Image Denoising

        Args:
            inp_channels (int, optional): Input channels of HSI. Defaults to 31.
            dim (int, optional): Embedding dimension. Defaults to 90.
            window_size (int, optional): Window size of non-local spatial attention. Defaults to 8.
            depths (list, optional): Number of Transformer block at different layers of network. Defaults to [ 6,6,6,6,6,6].
            num_heads (list, optional): Number of attention heads in different layers. Defaults to [ 6,6,6,6,6,6].
            mlp_ratio (int, optional): Ratio of mlp dim. Defaults to 2.
            qkv_bias (bool, optional): Learnable bias to query, key, value. Defaults to True.
            qk_scale (_type_, optional): The qk scale in non-local spatial attention. Defaults to None. If it is set to None, the embedding dimension is used to calculate the qk scale.
            bias (bool, optional):  Defaults to False.
            drop_path_rate (float, optional):  Stochastic depth rate of drop rate. Defaults to 0.1.
    """

    def __init__(self,
                 inp_channels=31,
                 dim=90,
                 window_size=8,
                 depths=[6, 6, 6, 6, 6, 6],
                 num_heads=[6, 6, 6, 6, 6, 6],
                 mlp_ratio=2,
                 qkv_bias=True, qk_scale=None,
                 bias=False,
                 drop_path_rate=0.1
                 ):

        super(SST, self).__init__()

        self.conv_first = nn.Conv2d(inp_channels, dim, 3, 1, 1)  # shallow featrure extraction

        self.input3D = nn.Sequential(
            nn.Conv3d(1, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        )

        self.output3D = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.Conv3d(dim, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        )

        self.num_layers = depths
        self.layers = nn.ModuleList()
        print(len(self.num_layers))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i_layer in range(len(self.num_layers)):
            layer = SMSBlock(dim=dim,
                             inp_channels=inp_channels,
                             window_size=window_size,
                             depth=depths[i_layer],
                             num_head=num_heads[i_layer],
                             mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                             bias=bias)
            self.layers.append(layer)

        self.output = nn.Conv2d(int(dim), dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_delasta = nn.Conv2d(dim, inp_channels, 3, 1, 1)  # reconstruction from features

    def forward(self, inp_img):
        # inp_img batch_size,channels,h,w
        f1 = self.conv_first(inp_img)

        # # 3D 卷积
        # B,C,H,W = inp_img.size()
        # f1 = inp_img.unsqueeze(1)
        # f1 = self.input3D(f1)
        # f1 = rearrange(f1, 'B C N H W -> (B N) C H W', N=self.dim)

        x = f1
        for layer in self.layers:
            x = layer(x)

        x = self.output(x + f1)
        x = self.conv_delasta(x) + inp_img
        return x

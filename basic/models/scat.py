import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from timm.models.layers import DropPath,to_2tuple
import cv2

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

class UnFold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b*c, 1, h, w), self.weights, stride=1, padding=self.kernel_size//2)        
        return x.reshape(b, c*(self.kernel_size**2), h*w)


class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
    def forward(self, x):
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size//2)        
        return x

class SASA(nn.Module):
    def __init__(self, dim, n_iter=1,token_size=to_2tuple(4),channel_scale=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.n_iter = n_iter
        self.token_size = token_size
        self.scale = dim ** - 0.5
        self.unfold = UnFold(3)
        self.fold = Fold(3)
        self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.channel_scale = channel_scale
        
    def forward(self, x):
        
        B, N, C = x.shape
        H0 = W0 = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H0, W0)

        if self.channel_scale > 1:
            cc = C//self.channel_scale
            x.contiguous().view(B, C // self.channel_scale, self.channel_scale,  H0, W0)
            # Sum on the third dimension 
            x.sum(dim=2)

        h_spe, w_spae = self.token_size
        _, _, H, W = x.shape
        hh, ww = H//h_spe, W//w_spae
        
        window_feature = F.adaptive_avg_pool2d(x, (hh, ww))
        
        pix_feature = x.reshape(B, C, hh, h_spe, ww, w_spae).permute(0, 2, 4, 3, 5, 1).reshape(B, hh*ww, h_spe*w_spae, C)

        window_feature = self.unfold(window_feature)
        window_feature = window_feature.transpose(1, 2).reshape(B, hh*ww, C, self.channel_scale**2)
        affinity_feature = pix_feature @ window_feature * self.scale
                
        affinity_feature = affinity_feature.softmax(-1)     
        affinity_feature_sum = affinity_feature.sum(2).transpose(1, 2).reshape(B, self.channel_scale**2, hh, ww)  
        affinity_feature_sum = self.fold(affinity_feature_sum)
                    
        window_feature = pix_feature.transpose(-1, -2) @ affinity_feature
        window_feature = self.fold(window_feature.permute(0, 2, 3, 1).reshape(B*C, self.channel_scale**2, hh, ww)).reshape(B, C, hh, ww)            
        window_feature = window_feature/(affinity_feature_sum.detach() + 1e-12)

        if self.channel_scale > 1 and window_feature.shape[1]==cc:
            window_feature.contiguous().view(B,C*self.channel_scale,hh,ww)
        
        window_feature = self.stoken_refine(window_feature)
        window_feature = self.unfold(window_feature) 
        window_feature = window_feature.transpose(1, 2).reshape(B, hh*ww, C, self.channel_scale**2) 
       
        pix_feature = window_feature @ affinity_feature.transpose(-1, -2)
        pix_feature = pix_feature.reshape(B, hh, ww, C, h_spe, w_spae).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)      
        feature_map = pix_feature.reshape(B,H*W,C)
        
        return feature_map
    
    def flops(self,shape):
        flops = 0
        H,W = shape
        th,tw = self.token_size
        flops += th*tw*H*W/self.channel_scale/9
        return flops
    
class CSSA(nn.Module):
    def __init__(self,index,num_heads,
                 dim,dim_out,
                 window_h, w_step, 
                 attn_drop=0., qk_scale=None):
        super().__init__()
        
        self.dim = dim
        self.dim_out = dim_out or dim
        head_dim = dim // num_heads
        self.window_h = window_h
        self.windows_w = w_step
        self.num_heads = num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        if index == 0:
            h_spa, w_spa = self.window_h, self.windows_w
        elif index == 1:
            w_spa, h_spa = self.window_h, self.windows_w
        elif index == 2:
            w_spa = h_spa = self.window_h//2
            
        self.h_spa = h_spa
        self.w_spa = w_spa
        
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)
        
    def spa2window(self,img, h_spa, w_spa):
        
        B, C, H, W = img.shape
        img_reshape = img.view(B, C, H // h_spa, h_spa, W // w_spa, w_spa)
        img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, h_spa* w_spa, C)
        return img_perm

    def window2spa(self,spa_splits_hw, h_spa, w_spa, H, W):

        B = int(spa_splits_hw.shape[0] / (H * W / h_spa / w_spa))
        img = spa_splits_hw.view(B, H // h_spa, W // w_spa, h_spa, w_spa, -1)
        img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return img

    def img2window(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = self.spa2window(x, self.h_spa, self.w_spa)
        x = x.reshape(-1, self.h_spa* self.w_spa, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def AttentionLePE(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)

        h_spa, w_spa = self.h_spa, self.w_spa
        x = x.view(B, C, H // h_spa, h_spa, W // w_spa, w_spa)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, h_spa, w_spa) ### B', C, H', W'

        lepe = func(x) ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, h_spa * w_spa).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.h_spa* self.w_spa).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv,mask=None):

        q,k,v = qkv[0], qkv[1], qkv[2]
        B, _, C = q.shape
        H = W = self.window_h
      
        q = self.img2window(q)
        q = q * self.scale
        k = self.img2window(k)
        v, lepe = self.AttentionLePE(v, self.get_v)
        
        attn = (q @ k.transpose(-2, -1)) 
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.h_spa* self.w_spa, C)
        x = self.window2spa(x, self.h_spa, self.w_spa, H, W).view(B, -1, C)

        return x
    
    def flops(self,shape):
        flops = 0
        H, W = shape
        #q, k, v = (B* H//H_sp * W//W_sp) heads H_sp*W_sp C//heads
        flops += ( (H//self.h_spa) * (W//self.w_spa)) *self.num_heads* (self.h_spa*self.w_spa)*(self.dim//self.num_heads)*(self.h_spa*self.w_spa)
        flops += ( (H//self.h_spa) * (W//self.w_spa)) *self.num_heads* (self.h_spa*self.w_spa)*(self.dim//self.num_heads)*(self.h_spa*self.w_spa)

        return flops

class AttnProcess(nn.Module):
    def __init__(self, dim, num_heads,
                 window_size,token_size, channel_scale, 
                 split_size=1,
                 qkv_bias=0, qk_scale=None,weight_factor=0.1,
                 attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        head_dim = dim // num_heads

        self.num_heads = num_heads
        self.window_size = window_size
        self.token_size = token_size
        self.channel_scale = channel_scale
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.scale = qk_scale or head_dim ** -0.5
        self.weight_factor = weight_factor
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.cssa = nn.ModuleList([CSSA(i,num_heads//3,
                                        dim//3,dim//3,
                                        window_h=self.window_size[0], w_step=split_size,
                                        attn_drop=attn_drop,qk_scale=qk_scale)
                                    for i in range(3)])
        self.sasa = SASA(dim, token_size=token_size,channel_scale=channel_scale,
                            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            attn_drop=attn_drop, proj_drop=attn_drop)  
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        
        x_h = self.cssa[0](qkv[:,:,:,:C//3],mask)
        x_s = self.cssa[2](qkv[:,:,:,C//3:2*C//3],mask)
        x_v = self.cssa[1](qkv[:,:,:,2*C//3:],mask)
        
        cssa_x = torch.cat([x_h,x_s,x_v], dim=2)

        cssa_x = rearrange(cssa_x, 'b n (g d) -> b n ( d g)', g=4)
        
        sasa_x = self.sasa(x)
        
        attn_x = cssa_x + sasa_x * self.weight_factor
        x = self.proj(attn_x)
        x = self.proj_drop(x)
        
        return x
    
    def draw_feature(self,x):
        B, N, C = x.shape
        H0 = W0 = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H0, W0)
        result = x.to('cpu')
        print(result.shape)



    # def extra_repr(self) -> str:
    #     return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, shape):

        flops = 0
        H,W = shape
        # qkv = self.qkv(x)
        flops += self.cssa[0].flops([H,W])/3
        flops += self.cssa[1].flops([H,W])/3
        flops += self.cssa[2].flops([H,W])/3
        flops += self.sasa.flops([H,W])
        return flops
 
class SplitProcess(nn.Module):
    def __init__(self,dim,num_heads, 
                 window_size=7,token_size=8,channel_scale=1,
                 split_size=1, shift_step=0,
                 drop_path=0.0,weight_factor=0.1,
                 mlp_ratio=2.,qkv_bias=True, qk_scale=None, 
                 drop=0., attn_drop=0.,act_layer=nn.GELU):
        super(SplitProcess,self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.token_size = token_size
        self.shift_step = shift_step
        self.weight_factor=weight_factor
        self.mlp_ratio = mlp_ratio

        self.norm1 =  nn.LayerNorm(dim) 
        self.norm2 = nn.LayerNorm(dim) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attnpre = AttnProcess(dim,num_heads=num_heads,
                                   window_size=to_2tuple(self.window_size),token_size=to_2tuple(self.token_size),
                                   channel_scale=channel_scale,split_size=split_size,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,weight_factor=weight_factor,
                                   attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):

        B,C,H,W = x.shape
        count = 1
        
        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        if self.shift_step > 0:
            shifted_window = torch.roll(x, shifts=(-self.shift_step, -self.shift_step), dims=(1, 2))
        else:
            shifted_window = x

        attn_window = self.division(shifted_window, self.window_size)  
        attn_window = attn_window.view(-1, self.window_size * self.window_size, C)  
        
        attn_windows = self.attnpre(attn_window)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_window = self.reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if self.shift_step > 0:
            x = torch.roll(shifted_window, shifts=(self.shift_step, self.shift_step), dims=(1, 2))
        else:
            x = shifted_window
        
        x = x.view(B, H*W,C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).view(B, C, H, W)
        
        return x
    
    def division(self,x, side_size):
        B, H, W, C = x.shape
        x = x.view(B, H // side_size, side_size, W // side_size, side_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, side_size, side_size, C)
        return windows 


    def reverse(self,windows, side_size, H, W):
        B = int(windows.shape[0] / (H * W / side_size / side_size))
        x = windows.view(B, H // side_size, W // side_size, side_size, side_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def flops(self,shape):
        flops = 0
        H,W = shape
        nW = H * W / self.window_size / self.window_size
        flops += nW *self.attnpre.flops([self.window_size,self.window_size])
        return flops

class SCAB(nn.Module):
    def __init__(self,
        dim = 90,num_head=6,depth=6,
        window_size=8,token_size=8,channel_scale=1,
        split_size=1,mlp_ratio=2,
        drop_path=0.0,weight_factor=0.8,
        qkv_bias=True, qk_scale=None):
        
        super(SCAB,self).__init__()
        
        self.scab = nn.Sequential(*[SplitProcess(dim=dim,num_heads=num_head,
                                                     window_size=window_size,token_size=token_size,channel_scale=channel_scale,
                                                     split_size = split_size,shift_step=0 if i%2==0 else window_size//2,
                                                     drop_path = drop_path[i],weight_factor=weight_factor,
                                                     mlp_ratio=mlp_ratio,
                                                     qkv_bias=qkv_bias, qk_scale=qk_scale,)
                                                    for i in range(depth)])
        self.conv_backg = nn.Conv2d(dim, dim, 3, 1, 1)
    
    def forward(self,x):
        
        B,C,H,W = x.shape
        book = self.scab(x)
        out = self.conv_backg(book)+x
    
        return out

    def flops(self,shape):
        flops = 0
        for b in self.scab:
            flops += b.flops(shape)
        return flops
            
class SCAT(nn.Module):
    def __init__(self, 
                inp_channels=31, 
                dim = 90,
                num_heads=[6,6,6],
                depths=[6,6,6],
                window_sizes=[8,8,8],
                token_sizes = [8,8,8],
                channel_scales = [1,1,1],
                split_sizes=[1,1,1],
                mlp_ratio=2,
                drop_path_rate=0.1,
                weight_factor = 0.1,
                qkv_bias=True, 
                qk_scale=None
            ):

        super(SCAT, self).__init__()

        self.layers = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        for i_layer in range(len(depths)):
            layer = SCAB(dim = dim,num_head=num_heads[i_layer],depth=depths[i_layer],
                            window_size=window_sizes[i_layer],token_size=token_sizes[i_layer],channel_scale = channel_scales[i_layer],
                            split_size=split_sizes[i_layer],mlp_ratio=mlp_ratio,
                            drop_path =dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            weight_factor = weight_factor,qkv_bias=qkv_bias, qk_scale=qk_scale)
            self.layers.append(layer)
            
        self.conv_first = nn.Conv2d(inp_channels, dim, 3, 1, 1)
        self.conv_extract = nn.Conv2d(int(dim), dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_delasta = nn.Conv2d(dim,inp_channels, 3, 1, 1)

    def forward(self, inp_img):
        N,B,H,W = inp_img.shape
        device = inp_img.device
        denoised_img = torch.zeros_like(inp_img).to(device)
        if B % 31 == 0:
            num = B // 31
        else:
            num = B // 31 +1

        last = B%31
        for pos in range(num):
            if(pos <num-1 or last == 0):
                input = inp_img[:,pos * 31:(pos+1)*31,:,:]
            else:
                input = inp_img[:,B-31:B,:,:]

            _,_,h_inp,w_inp = input.shape
            hb, wb = 16, 16
            pad_h = (hb - h_inp % hb) % hb
            pad_w = (wb - w_inp % wb) % wb
            input = F.pad(input, (0, pad_h, 0, pad_w), 'reflect')
            f1 = self.conv_first(input)
            print(f1.shape)

            x=f1
            count = 0
            for layer in self.layers:
                x = layer(x)
                count = count + 1
                if pos == 1 and count == 3 :
                    result = x.to('cpu')
                    result = result[0,:,:,:]
                    color_img = np.concatenate([result[9][np.newaxis,:],result[19][np.newaxis,:],result[29][np.newaxis,:]],0)
                    print(color_img.shape)
                    color_img = color_img.transpose((1,2,0))*255
                    cv2.imwrite('/home/jiahua/liuy/hsi_pipeline/draw_pic/hot_test/test.png',color_img)
            
            x = self.conv_extract(x+f1)
            x = self.conv_delasta(x)+input
            x = x[:,:,:h_inp,:w_inp]
            
            if (last == 0 or pos< num -1):
                denoised_img[:, pos * 31:(pos+1)*31, :, :] = x
            else:
                denoised_img[:, B-last:B, :, :] = x[:,31-last:31,:,:]

        return denoised_img
    
    def flops(self,shape):
        flops = 0
        for i, layer in enumerate(self.layers):
            flops += layer.flops(shape)
        return flops
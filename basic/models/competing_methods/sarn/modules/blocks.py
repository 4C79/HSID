import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np

class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """
    def __init__(self, channels, normalize_shape=None, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif norm_type == 'pixel':
            self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type == 'none':
            self.norm = lambda x: x
        else:
            assert 1==0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)


class ReluLayer(nn.Module):
    """Relu Layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu 
            - SELU
            - none: direct pass
    """
    def __init__(self, channels, relu_type='relu'):
        super(ReluLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'selu':
            self.func = nn.SELU(True)
        elif relu_type == 'none':
            self.func = lambda x: x
        else:
            assert 1==0, 'Relu type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale='none', norm_type='none', relu_type='none', use_pad=True):
        super(ConvLayer, self).__init__()
        self.use_pad = use_pad
        
        bias = True if norm_type in ['pixel', 'none'] else False 
        stride = 2 if scale == 'down' else 1

        self.scale_func = lambda x: x
        if scale == 'up':
            self.scale_func = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')

        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2) 
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.relu = ReluLayer(out_channels, relu_type)
        self.norm = NormLayer(out_channels, norm_type=norm_type)

    def forward(self, x):
        out = self.scale_func(x)
        if self.use_pad:
            out = self.reflection_pad(out)
        out = self.conv2d(out)
        out = self.norm(out)
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    """
    Residual block recommended in: http://torch.ch/blog/2016/02/04/resnets.html
    ------------------
    # Args
        - hg_depth: depth of HourGlassBlock. 0: don't use attention map.
        - use_pmask: whether use previous mask as HourGlassBlock input.
    """
    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, att_name='spar'):
        super(ResidualBlock, self).__init__()
        self.c_in      = c_in
        self.c_out     = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth  = hg_depth

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        self.preact_func = nn.Sequential(
                    NormLayer(c_in, norm_type=self.norm_type),
                    ReluLayer(c_in, self.relu_type),
                    )

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']

        self.conv1 = ConvLayer(c_in, c_out, 3, scales[0], **kwargs) 
        self.conv2 = ConvLayer(c_out, c_out, 3, scales[1], norm_type=norm_type, relu_type='none')

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))

        self.att_func = HourGlassBlock(self.hg_depth, c_out, c_attn, **kwargs) 
        
    def forward(self, x):
        identity = self.shortcut_func(x)
        out = self.preact_func(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = identity + self.att_func(out)
        return out
        

class HourGlassBlock(nn.Module):
    """Simplified HourGlass block.
    Reference: https://github.com/1adrianb/face-alignment 
    --------------------------
    """
    def __init__(self, depth, c_in, c_out, 
            c_mid=64,
            norm_type='bn',
            relu_type='prelu',
            ):
        super(HourGlassBlock, self).__init__()
        self.depth     = depth
        self.c_in      = c_in
        self.c_mid     = c_mid
        self.c_out     = c_out
        self.kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if self.depth:
            self._generate_network(self.depth)
            self.out_block = nn.Sequential(
                    ConvLayer(self.c_mid, self.c_out, norm_type='none', relu_type='none'),
                    nn.Sigmoid()
                    )
            
    def _generate_network(self, level):
        if level == self.depth:
            c1, c2 = self.c_in, self.c_mid
        else:
            c1, c2 = self.c_mid, self.c_mid

        self.add_module('b1_' + str(level), ConvLayer(c1, c2, **self.kwargs)) 
        self.add_module('b2_' + str(level), ConvLayer(c1, c2, scale='down', **self.kwargs)) 
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvLayer(self.c_mid, self.c_mid, **self.kwargs)) 

        self.add_module('b3_' + str(level), ConvLayer(self.c_mid, self.c_mid, scale='up', **self.kwargs))

    def _forward(self, level, in_x):
        up1 = self._modules['b1_' + str(level)](in_x)
        low1 = self._modules['b2_' + str(level)](in_x)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules['b2_plus_' + str(level)](low1)

        up2 = self._modules['b3_' + str(level)](low2)
        if up1.shape[2:] != up2.shape[2:]:
            up2 = nn.functional.interpolate(up2, up1.shape[2:])

        return up1 + up2

    def forward(self, x, pmask=None):
        if self.depth == 0: return x
        input_x = x
        x = self._forward(self.depth, x)
        self.att_map = self.out_block(x)
        x = input_x * self.att_map
        return x

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.relu = nn.ReLU(inplace=True)

        self.res_translate = None
        if not inplanes == planes or not stride == 1:
            self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        if self.res_translate is not None:
            residual = self.res_translate(residual)
        out += residual

        return out


from torch import Tensor
import torch
def _no_grad_fill_(tensor, val):
    with torch.no_grad():
        return tensor.fill_(val)


def constant_(tensor: Tensor, val: float) -> Tensor:
    r"""Fills the input Tensor with the value :math:`\text{val}`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        val: the value to fill the tensor with

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.constant_(w, 0.3)
    """
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(constant_, (tensor,), tensor=tensor, val=val)
    return _no_grad_fill_(tensor, val)

def get_same_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class DeformableAttnBlock_FUSION(nn.Module):
    def __init__(self, n_heads=4, n_levels=3, n_points=4, d_model=32):
        super().__init__()
        self.n_levels = n_levels

        self.defor_attn = MSDeformAttn_Fusion(d_model=d_model, n_levels=3, n_heads=n_heads, n_points=n_points)
        self.feed_forward = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.emb_qk = nn.Conv2d(3 * d_model + 4, 3 * d_model, kernel_size=3, padding=1)
        self.emb_v = nn.Conv2d(3 * d_model, 3 * d_model, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.feedforward = nn.Sequential(
            nn.Conv2d(2 * d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.fusion = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def preprocess(self, srcs):
        bs, t, c, h, w = srcs.shape
        masks = [torch.zeros((bs, h, w)).bool().to(srcs.device) for _ in range(t)]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lv1 in range(t):
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs.device)
        return spatial_shapes, valid_ratios

    def forward(self, frame, srcframe):
        b, t, c, h, w = frame.shape
        # bs,t,c,h,w = frame.shape
        # warp_fea01 = warp(frame[:,0],flow_backward[:,0])
        # warp_fea21 = warp(frame[:,2],flow_forward[:,1])

        # qureys = self.act(self.emb_qk(torch.cat([warp_fea01,frame[:,1],warp_fea21,flow_forward[:,1],flow_backward[:,0]],1))).reshape(b,t,c,h,w)
        qureys = self.act(self.emb_qk(
            torch.cat((frame[:, 0], frame[:, 1], frame[:, 2]), 1)).reshape(b, t, c, h, w))

        value = self.act(self.emb_v(frame.reshape(b, t * c, h, w)).reshape(b, t, c, h, w))

        spatial_shapes, valid_ratios = self.preprocess(value)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self.get_reference_points(spatial_shapes[0].reshape(1, 2), valid_ratios, device=value.device)

        output = self.defor_attn(qureys, reference_points, value, spatial_shapes, level_start_index, None)

        output = self.feed_forward(output)
        output = output.reshape(b, c, h, w) + frame[:, 1]

        tseq_encoder_0 = torch.cat([output, srcframe[:, 1]], 1)
        output = output.reshape(b, c, h, w) + self.feedforward(tseq_encoder_0)
        output = self.fusion(output)
        return output


class MSDeformAttn_Fusion(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        # if not _is_power_of_2(_d_per_head):
        #     warnings.warn(
        #         "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
        #         "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        kernel_size = 3
        self.sampling_offsets = nn.Conv2d(n_levels * d_model, n_heads * n_levels * n_points * 2,
                                          kernel_size=kernel_size, padding=kernel_size // 2)
        # self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.attention_weights = nn.Conv2d(n_levels * d_model, n_heads * n_levels * n_points, kernel_size=kernel_size,
                                           padding=kernel_size // 2)
        # self.value_proj = nn.Linear(d_model, d_model)
        # self.output_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2)
        self.output_proj = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2)
        # self._reset_parameters()

    # def _reset_parameters(self):
    #     constant_(self.sampling_offsets.weight.data, 0.)
    #     thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
    #     grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
    #     grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).reshape(self.n_heads, 1, 1, 2).repeat(1,
    #                                                                                                              self.n_levels,
    #                                                                                                              self.n_points,
    #                                                                                                              1)
    #     for i in range(self.n_points):
    #         grid_init[:, :, i, :] *= i + 1
    #     with torch.no_grad():
    #         self.sampling_offsets.bias = nn.Parameter(grid_init.reshape(-1))
    #     # constant_(self.sampling_offsets.bias, 0.)
    #     # constant_(self.sampling_offsets.weight.data, 0.)
    #     constant_(self.attention_weights.weight.data, 0.)
    #     constant_(self.attention_weights.bias.data, 0.)
    #     xavier_uniform_(self.value_proj.weight.data)
    #     constant_(self.value_proj.bias.data, 0.)
    #     xavier_uniform_(self.output_proj.weight.data)
    #     constant_(self.output_proj.bias.data, 0.)

    def flow_guid_offset(self, flow_forward, flow_backward, offset):
        # sampling_offsets = sampling_offsets.reshape(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # sampling_offsets: B,T*H*W,heads,T,K,2

        # flow:b,2,h,w ----> b,3*h*w,2
        # (N, Length_{query}, n_levels, 2)
        # reference_points[:, :, None, :, None, :]
        # _,t,_,_,_,_ = offset.shape
        # offset = offset.reshape()
        # N, h*w, self.n_heads, self.n_levels, self.n_points, 2
        N, HW, heads, n_levels, points, _ = offset.shape
        # N,T,HW,heads,n_levels,points,2
        # offset = offset.reshape(N,,heads,n_levels,points,2)
        # [4, 1, 4096, 8, 3, 12, 2]
        # offset_chunk0,offset_chunk1,offset_chunk2 = torch.chunk(offset, n_levels, dim=3)

        # 4,2,64,64
        # flow_forward01 = flow_forward[:,0]
        flow_forward12 = flow_forward[:, 1]
        # flow_forward02 = flow_forward01 + warp(flow_forward12,flow_forward01)

        flow_backward10 = flow_backward[:, 0]
        flow_zeros = torch.zeros_like(flow_forward12)
        flow_stack = torch.stack([flow_backward10, flow_zeros, flow_forward12], dim=2)
        # 3,2,3,64,64
        # N,HW,n_levels,2
        offset = offset + flow_stack.reshape(N, 2, n_levels, HW).permute(0, 3, 2, 1)[:, :, None, :, None, :]
        # flow_backward21 = flow_backward[:,1]
        # flow_backward20 = flow_backward21 + warp(flow_backward10,flow_backward21)

        # b,c,h,w = flow_backward10.shape
        # 4,h*w,2
        # flow_forward01 = flow_forward01.permute(0, 2, 3, 1).reshape(b,-1,c)
        # flow_forward12 = flow_forward12.permute(0, 2, 3, 1).reshape(b,-1,c)
        # flow_forward02 = flow_forward02.permute(0, 2, 3, 1).reshape(b,-1,c)

        # flow_backward10 = flow_backward10.permute(0, 2, 3, 1).reshape(b,-1,c)
        # flow_backward21 = flow_backward21.permute(0, 2, 3, 1).reshape(b,-1,c)
        # flow_backward20 = flow_backward20.permute(0, 2, 3, 1).reshape(b,-1,c)

        # flow_zeros = torch.zeros_like(flow_forward01)
        # 4,4096,3,2
        # offset_chunk0 = offset_chunk0 + torch.stack([flow_zeros,flow_forward01,flow_forward02],dim=2)[:,None,:,None,:,None,:]
        # offset_chunk1 = offset_chunk1 + torch.stack([flow_backward10,flow_zeros,flow_forward12],dim=2)[:,None,:,None,:,None,:]
        # offset_chunk2 = offset_chunk2 + torch.stack([flow_backward20,flow_backward21,flow_zeros],dim=2)[:,None,:,None,:,None,:]

        # offset = torch.cat([offset_chunk0,offset_chunk1,offset_chunk2],dim=1).reshape( N,THW,heads,n_levels,points,2)

        return offset

    def _reset_offset(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).reshape(self.n_heads, 1, 1, 2).repeat(1,
                                                                                                                 self.n_levels,
                                                                                                                 self.n_points,
                                                                                                                 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.reshape(-1))

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index,
                input_padding_mask):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """

        # assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        bs, t, c, h, w = query.shape
        value = self.value_proj(input_flatten.reshape(bs * t, c, h, w)).reshape(bs, t, c, h, w)
        sampling_offsets = self.sampling_offsets(query.reshape(bs, t * c, h, w)).reshape(bs, -1, h, w)

        attention_weights = self.attention_weights(query.reshape(bs, t * c, h, w)).reshape(bs, -1, h, w)

        # query = query.flatten(3).transpose(2, 3).contiguous().reshape(bs,-1,c)
        query = query.flatten(3).transpose(2, 3).contiguous().reshape(bs, -1, c)

        value = value.flatten(3).transpose(2, 3).contiguous().reshape(bs, -1, c)
        sampling_offsets = sampling_offsets.flatten(2).transpose(1, 2).contiguous().reshape(bs, -1,
                                                                                            self.n_heads * self.n_levels * self.n_points * 2)

        attention_weights = attention_weights.flatten(2).transpose(1, 2).contiguous().reshape(bs, -1,
                                                                                              self.n_heads * self.n_levels * self.n_points)
        N, Len_q, _ = query.shape
        N, Len_in, _ = value.shape
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.reshape(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = sampling_offsets.reshape(N, h * w, self.n_heads, self.n_levels, self.n_points, 2)
        # sampling_offsets = self.flow_guid_offset(flow_forward,flow_backward,sampling_offsets)
        attention_weights = attention_weights.reshape(N, h * w, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).reshape(N, h * w, self.n_heads, self.n_levels,
                                                                     self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # print(reference_points[:, :, None, :, None, :].shape)
            # print(offset_normalizer[None, None, None, :, None, :].shape)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights,
            self.im2col_step)
        output = output.reshape(bs, h, w, c).permute(0, 3, 1, 2)
        output = self.output_proj(output)
        return output
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import MultiScaleDeformableAttention as MSDA

class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
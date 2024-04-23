import torch
import torch.nn as nn
import torch.nn.functional as F
from .drconv import *
from .model_util import *
from einops import rearrange
from .base_block import *

cor = True
dr = True
atten = True
back = True

class Encoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=14):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        # 1st GRU layer, bi-directional
        # self.corr_f = CrossAttention(dim=hidden_dim)
        self.corr_f = Corrlayer(dim=hidden_dim)

        # self.corr_b = CrossAttention(dim=hidden_dim)
        self.corr_b = Corrlayer(dim=hidden_dim)

        self.drconv_ss_f = DRConv2d_ss_k(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, region_num=8,
                               guide_input_channel=True, padding=1)
        self.drconv_ss_b = DRConv2d_ss_k(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, region_num=8,
                                       guide_input_channel=True, padding=1)
        self.encoder_layer1_f = ConvGRU3D(hidden_dim=hidden_dim, input_dim=input_dim)
        self.encoder_layer1_b = ConvGRU3D(hidden_dim=hidden_dim, input_dim=input_dim, is_backward=True)
        self.fution = nn.Conv2d(hidden_dim*2, hidden_dim, 3, 1, 1, bias=True)
        self.fution_ss = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, 1, 1, bias=True)

        self.conv_inp_1 = nn.Conv2d(1, self.hidden_dim, 3, 1, 1, bias=True)
        self.conv_inp_2 = nn.Conv2d(self.hidden_dim, hidden_dim, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def in2fea(self, x):
        out = self.lrelu(self.conv_inp_1(x))
        out = self.lrelu(self.conv_inp_2(out))
        out = out + x
        return out

    def corr_atten_f(self,x,y):
        return self.corr_f(x,y)

    def corr_atten_b(self,x,y):
        return self.corr_b(x,y)

    def forward(self, x):
        if len(x.shape) == 4:
            x = torch.unsqueeze(x, dim=2)
        n, b, c, h, w = x.shape

        # 1st ConvGRU3D layer, bi-directional
        # forward

        hidden_states_f = []
        hidden_state = x.new_zeros(n, self.hidden_dim, h, w)
        in_fea = []
        in_fea_ori = []
        for i in range(0, b):
            in_fea_i = self.in2fea(x[:,i])
            in_fea_ori.append(in_fea_i)
            if dr:
                in_fea_i = self.drconv_ss_f(in_fea_i, hidden_state)
            in_fea.append(in_fea_i)
            if cor:
                hidden_state = self.corr_f(in_fea_ori[i],hidden_state)
            hidden_state = self.encoder_layer1_f(hidden_state,in_fea_i)

            hidden_states_f.append(hidden_state)
        # backward
        if back:
            print("?????")
            hidden_states_b = []
            hidden_state = x.new_zeros(n, self.hidden_dim, h, w)
            for i in range(b-1, -1, -1):
                hidden_state_f = hidden_states_f[i]
                # if dr:
                #     in_fea[i] = self.drconv_ss_b(in_fea[i], hidden_state)
                # if cor:
                #     hidden_state = self.corr_b(in_fea[i],hidden_state)
                # if cor:
                #     hidden_state = self.corr_b(in_fea[i], hidden_state)

                hidden_state = self.encoder_layer1_b(hidden_state, in_fea[i], hidden_state_f)
                hidden_states_b.append(hidden_state)
            hidden_states_b = hidden_states_b[::-1]

            inputs = hidden_states_b
        else: 
            print("out")
            inputs = hidden_states_f
        # inputs = hidden_states_f
        return inputs

class Decoder(nn.Module):
    def __init__(self, hidden_dim=14):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.range = 1
        # ConvGRU3D layers, uni-direction
        self.drconv_ss = DRConv2d_ss_k(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, region_num=8,
                                         guide_input_channel=True, padding=1)
        # self.corr = CrossAttention(dim=hidden_dim)
        self.corr = Corrlayer(dim=hidden_dim)
        self.convgru_layers = ConvGRU3D(hidden_dim=hidden_dim,input_dim=hidden_dim)
            # self.recon_layers.append(make_layer(ResidualBlockNoBN, num_blocks, mid_channels=hidden_dim))
        self.fution_layers = nn.Conv2d(hidden_dim*2, hidden_dim, 3, 1, 1, bias=True)
        self.attention = Attention_3d(dec_hid_dim = 2*self.range+1)
        # self.attention = CrossAttention(dim=hidden_dim)

        # Convert hidden state to output

        self.conv_out_1 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True)
        self.conv_out_2 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True)
        self.conv_out_3 = nn.Conv2d(hidden_dim, 1, 3, 1, 1, bias=True)

        # Convert t-th output to (t+1)-th input


        self.conv_inp_1 = nn.Conv2d(1, self.hidden_dim, 3, 1, 1, bias=True)
        self.conv_inp_2 = nn.Conv2d(self.hidden_dim, hidden_dim, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv_stage2_1 = nn.Conv2d(1, self.hidden_dim, 3, 1, 1, bias=True)
        self.conv_stage2_2 = nn.Conv2d(self.hidden_dim, 1, 3, 1, 1, bias=True)
        self.conv_stage2_3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1, bias=True)

    def hidden2out(self, x_fea, base):
        out = self.lrelu(self.conv_out_1(x_fea))
        out = self.lrelu(self.conv_out_2(out))
        out = self.conv_out_3(out)
        return out + base

    def out2inp(self, x):
        out = self.lrelu(self.conv_inp_1(x))
        out = self.lrelu(self.conv_inp_2(out))
        # x = x.repeat(1, self.hidden_dim, 1, 1)
        out = out + x
        return out
    def forward(self, x, encoder_out):
        '''
        encoder_out: the output of the top layer [n,t,c,h,w]
        encoder_hidden_state: the hidden state of every layers lx[n,c,h,w]
        flows: flows_forward, flows_backward
        '''
        if len(x.shape) == 4:
            x = torch.unsqueeze(x, dim=2)
        n, b, c, h, w = x.shape
        base = x

        # GRU layers, uni-directional
        # initial hidden state
        hidden_state = encoder_out[-1]
        # backward
        outputs = []
        input = encoder_out[-1]
        for i in range(b - 1, -1, -1):
            encoder_out_m = torch.unsqueeze(encoder_out[i], dim=1)
            if i == b-1:
                encoder_out_l = get_back(encoder_out,i,self.range)
                adjacent_encoder_out = torch.cat([encoder_out_l, encoder_out_m], dim=1)
            elif i == 0:
                encoder_out_r = get_forth(encoder_out,i,self.range)
                adjacent_encoder_out = torch.cat([encoder_out_m,encoder_out_r], dim=1)
            else:
                encoder_out_l = get_back(encoder_out, i, self.range)
                encoder_out_r = get_forth(encoder_out, i, self.range)
                adjacent_encoder_out = torch.cat([encoder_out_l, encoder_out_m, encoder_out_r],dim=1)

            if atten:
                context = self.attention(hidden_state,adjacent_encoder_out)

            else: context = torch.sum(adjacent_encoder_out, dim=1)/3

            # if cor:
            #     input_res = input
            #     input = self.corr(context,input)+input_res
            # if dr:
            #     input = self.drconv_ss(input, hidden_state)

            input = self.lrelu(self.fution_layers(torch.cat((input, context), dim=1)))

            if cor:
                hidden_state = self.corr(input,hidden_state)
            hidden_state = self.convgru_layers(hidden_state, input)
            # if dr:
            #     hidden_state = self.drconv(hidden_state)

            output = self.hidden2out(hidden_state, base[:,i])

            input = self.out2inp(output)
            # input = self.drconv(input)
            # print(input)
            outputs.append(output)
        outputs = outputs[::-1]
        outputs = torch.stack(outputs, dim=1).squeeze(dim=2)
        # if len(outputs.shape) ==3:
        #     outputs = torch.unsqueeze(outputs, dim=0)
        # outputs = outputs.requires_grad
        # outputs = self.drconv(outputs, outputs)
        # if dr:
        #     outputs_2 = []
        #     r = 24
        #     for i in range(b):
        #         fea_stage2_i = self.conv_stage2_1(outputs[:,i:i+1])
        #         adjacent = get_adjacent(outputs,i,r)
        #         adjacent = self.conv_stage2_3(adjacent)
        #         out = self.drconv(fea_stage2_i,adjacent)
        #         out = out + fea_stage2_i
        #         out = self.conv_stage2_2(out) + base[:,i]
        #         outputs_2.append(out)
        #     outputs_2 = torch.stack(outputs_2, dim=1).squeeze(dim=2)
        #     return outputs,outputs_2
        return outputs

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):

        super().__init__()
        self.attn = nn.Conv3d(1 + dec_hid_dim, dec_hid_dim, 3, 1, 1, bias=False)
        self.v = nn.Conv3d(dec_hid_dim, dec_hid_dim, 3, 1, 1, bias=False)
    def forward(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=1)))
        attention = self.v(energy)
        attention = F.softmax(attention, dim=1)
        context = torch.sum(attention*encoder_outputs, dim=1)

        return context


class Attention_ss(nn.Module):
    def __init__(self, dec_hid_dim):
        super().__init__()

        self.conv1 = nn.Conv3d(dec_hid_dim, dec_hid_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(dec_hid_dim, dec_hid_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(dec_hid_dim, dec_hid_dim, kernel_size=3, padding=1)
        # self.attn =TAdaConvBlock(dec_hid_dim, dec_hid_dim)
        # self.v = TAdaConvBlock(dec_hid_dim, dec_hid_dim)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(dim=2)
        hidden = encoder_outputs+hidden
        # energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=1)))
        attention = self.attn(hidden)
        attention = torch.tanh(attention)
        # encoder_outputs_res = encoder_outputs_res + attention
        # attention = self.v(energy)
        # attention = self.attn(hidden)
        attention = self.v(attention)
        attention = F.softmax(attention, dim=2)
        context = torch.sum(attention*encoder_outputs, dim=2)

        return context

class Attention_3d(nn.Module):
    def __init__(self, dec_hid_dim):
        super().__init__()

        self.conv1 = nn.Conv3d(dec_hid_dim+1, dec_hid_dim, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv3d(dec_hid_dim, dec_hid_dim, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv3d(dec_hid_dim, dec_hid_dim, kernel_size=3, padding=1)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(dim=1)
        hidden = torch.cat((hidden,encoder_outputs),dim=1)
        map = self.conv1(hidden)
        # map = torch.tanh(self.conv2(hidden))
        map = torch.tanh(map)
        # energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=1)))
        # hidden = self.conv3(hidden)
        # hidden = F.softmax(self.conv3(map), dim=1)
        hidden = F.softmax(map, dim=1)
        context = torch.sum(hidden * encoder_outputs, dim=1)

        return context
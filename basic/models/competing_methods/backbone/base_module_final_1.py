
from .drconv import *
from .base_block import *

# cor = True
# dr = True
# atten = True
# local_range = 1
# region_num = 8
# hidden_dim = 14
back = True
norm  = 0

class Encoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=14,cor=1,dr=1,atten=1,
                 drSpa=1,drSpaConv=1,drSpec=1,
                 local_range = 1,region_num = 8,cuda=0,device=1):
        super(Encoder, self).__init__()
        self.cor = cor
        self.dr = dr
        self.hidden_dim = hidden_dim
        self.drSpa=drSpa
        self.drSpaConv=drSpaConv
        self.drSpec=drSpec
        cuda_ = cuda
        self.device = 'cpu' if cuda_ else 'cuda:' + str(device)[0] 
        if norm ==1:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)

        if self.cor==1:
            # self.corr = Corrlayer(dim=hidden_dim)
            self.corr = CrossAttention(dim=hidden_dim)

        self.atten = atten
        
        # print("opt: drcoratten:",self.dr,self.cor,self.atten)
        if self.dr==1:
            self.drconv = SSRA(hidden_dim = hidden_dim, drSpa= self.drSpa,drSpaConv= self.drSpaConv,drSpec=self.drSpec,region_num = region_num)

        self.encoder_layer = ConvGRU3D(hidden_dim=hidden_dim, input_dim=input_dim)

        if back==True:
            self.encoder_layer1_b = ConvGRU3D(hidden_dim=hidden_dim, input_dim=input_dim, is_backward=True)

        self.conv_inp_1 = nn.Conv2d(1, self.hidden_dim, 3, 1, 1, bias=True)
        self.conv_inp_2 = nn.Conv2d(self.hidden_dim, hidden_dim, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def in2fea(self, x):
        out = self.lrelu(self.conv_inp_1(x))
        out = self.lrelu(self.conv_inp_2(out))
        out = out + x
        return out

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
        # for i in range(b - 1, -1, -1):
            in_fea_i = self.in2fea(x[:,i])
            if norm == 1:
                n,c,h,w = in_fea_i.shape
                in_fea_i = in_fea_i.flatten(2).transpose(1, 2)
                hidden_state = hidden_state.flatten(2).transpose(1, 2)
                in_fea_i =  self.norm1(in_fea_i)
                hidden_state = self.norm2(hidden_state)
                in_fea_i = in_fea_i.view(n, c, h, w)
                hidden_state = hidden_state.view(n, c, h, w)
            in_fea_ori.append(in_fea_i)
            # if self.dr==1:
            #     in_fea_i = self.drconv(in_fea_i, hidden_state)
            in_fea.append(in_fea_i)
            if self.cor==1:
                hidden_state = self.corr(in_fea_ori[i],hidden_state)
                # hidden_state = self.corr(in_fea_ori[b-i-1],hidden_state)
            hidden_state = self.encoder_layer(hidden_state,in_fea_i)

            hidden_states_f.append(hidden_state)
        # backward
        if back:
            # print("?????")
            hidden_states_b = []
            hidden_state = x.new_zeros(n, self.hidden_dim, h, w).to(self.device)
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
            # print("out")
            inputs = hidden_states_f
        if self.dr==1:
            inputs = self.drconv(inputs)
        # print(inputs)
        # inputs = hidden_states_f
        return inputs


class SSRA(nn.Module):
    def __init__(self, hidden_dim=14,
                 drSpa=1,drSpaConv=1,drSpec=1,region_num = 8):
        super(SSRA, self).__init__()
    
        self.drconv = DRConv2d_ss_k(in_channels=hidden_dim, out_channels=hidden_dim, 
                                    drSpa=drSpa,drSpaConv=drSpaConv,drSpec=drSpec,
                                    kernel_size=3, region_num=region_num,
                            guide_input_channel=True, padding=1)

    def forward(self, hidden_states_b):      
        hidden_states_enhance = []
    
        b = len(hidden_states_b)
        for i in range(0, b): 
            hidden_state =  hidden_states_b[i]      
            hidden_state = self.drconv(hidden_state, hidden_state)
            hidden_states_enhance.append(hidden_state)  

        return hidden_states_enhance


class Decoder(nn.Module):
    def __init__(self, hidden_dim=14,cor=1,dr=1,atten=1,
                 local_range = 1,region_num = 8,direct=1,fair=1,ar=1):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.range = local_range
        self.cor = cor
        self.dr = dr
        self.atten = atten
        self.direct= direct
        self.fair = fair
        self.ar = ar
        if self.cor==1:
            # self.corr = Corrlayer(dim=hidden_dim)
            self.corr = CrossAttention(dim=hidden_dim)
        self.convgru_layers = ConvGRU3D(hidden_dim=hidden_dim,input_dim=hidden_dim)
        if back==True:
            self.decoder_layer1_b = ConvGRU3D(hidden_dim=hidden_dim, input_dim=hidden_dim, is_backward=True)

        if self.atten==1:
            self.attention = Attention_3d(dec_hid_dim = 2*self.range+1)

        self.conv_out_1 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True)
        self.conv_out_2 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True)
        self.conv_out_3 = nn.Conv2d(hidden_dim, 1, 3, 1, 1, bias=True)


        self.conv_inp_1 = nn.Conv2d(1, self.hidden_dim, 3, 1, 1, bias=True)
        self.conv_inp_2 = nn.Conv2d(self.hidden_dim, hidden_dim, 3, 1, 1, bias=True)

        if self.ar == 1:            
            self.fution_layers = nn.Conv2d(hidden_dim*2, hidden_dim, 3, 1, 1, bias=True)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

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
        if len(x.shape) == 4:
            x = torch.unsqueeze(x, dim=2)
        n, b, c, h, w = x.shape
        base = x


        hidden_state = encoder_out[-1]
        hidden_states_f = []
        outputs = []
        input = encoder_out[-1]
        input_list = []
        for i in range(b - 1, -1, -1):
        # for i in range(0 , b):
            encoder_out_m = torch.unsqueeze(encoder_out[i], dim=1)
            if self.range == 0:
                adjacent_encoder_out = encoder_out_m
            else:
                if i == b - 1:
                    encoder_out_l = get_back(encoder_out, i, self.range)
                    adjacent_encoder_out = torch.cat([encoder_out_l, encoder_out_m], dim=1)
                elif i == 0:
                    encoder_out_r = get_forth(encoder_out, i, self.range)
                    adjacent_encoder_out = torch.cat([encoder_out_m, encoder_out_r], dim=1)
                else:
                    encoder_out_l = get_back(encoder_out, i, self.range)
                    encoder_out_r = get_forth(encoder_out, i, self.range)
                    adjacent_encoder_out = torch.cat([encoder_out_l, encoder_out_m, encoder_out_r], dim=1)

            if self.atten==1:
                context = self.attention(hidden_state,adjacent_encoder_out)

            else: 
                if self.direct==1:
                    context = input
                elif self.fair==1:
                    context = encoder_out[-1]
                else:
                    context = torch.sum(adjacent_encoder_out, dim=1)/3
            if self.ar == 1:
                input = self.lrelu(self.fution_layers(torch.cat((input, context), dim=1)))
            else:
                input = context
            input_list.append(input)
            if self.cor==1:
                hidden_state = self.corr(input,hidden_state)

            hidden_state = self.convgru_layers(hidden_state, input)
            hidden_states_f.append(hidden_state)
            if not back:
                output = self.hidden2out(hidden_state, base[:,i])

                input = self.out2inp(output)

                outputs.append(output)

        if back:    
            input_list = input_list[::-1]
            hidden_states_f = hidden_states_f[::-1]
            hidden_state = x.new_zeros(n, self.hidden_dim, h, w)
            for i in range(0 , b):
                hidden_state_f = hidden_states_f[i]
                hidden_state = self.decoder_layer1_b(hidden_state,input_list[i],hidden_state_f)
                output = self.hidden2out(hidden_state, base[:,i])
                input = self.out2inp(output)
                outputs.append(output)
        if not back:
            outputs = outputs[::-1]
        outputs = torch.stack(outputs, dim=1).squeeze(dim=2)

        return outputs

class Attention_3d(nn.Module):
    def __init__(self, dec_hid_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(dec_hid_dim+1, dec_hid_dim, kernel_size=3, padding=1)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(dim=1)
        hidden = torch.cat((hidden,encoder_outputs),dim=1)
        map = self.conv1(hidden)
        map = torch.tanh(map)
        hidden = F.softmax(map, dim=1)
        context = torch.sum(hidden * encoder_outputs, dim=1)

        return context
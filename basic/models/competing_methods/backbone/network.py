import torch
import torch.nn as nn
from .base_module_final_2 import *
from thop import profile

# cor = True
# dr = True
# atten = True

class S2SHSID(nn.Module):
    def __init__(self, opt):
        super(S2SHSID, self).__init__()
        self.encoder = Encoder(input_dim=1,hidden_dim=opt['hidden_dim'],cor=opt['cor'],dr=opt['dr'],
                               drSpa=opt['drSpa'],drSpaConv=opt['drSpaConv'],drSpec=opt['drSpec'],
                               atten=opt['atten'],local_range = opt['local_range'],region_num =opt['region_num'],cuda=opt['cpu'],device=opt['gpu_ids'])
        self.decoder = Decoder(hidden_dim=opt['hidden_dim'],cor=opt['cor'],dr=opt['dr'],atten=opt['atten'],
                               local_range = opt['local_range'],region_num = opt['region_num'],direct=opt['direct'],fair=opt['fair'],ar=opt['ar'])

    def forward(self, x):
        '''
        :param x: [n,b,c,h,w]
        :return: out: [n,b,c,h,w]
        '''
        # encoder
        encoder_out = self.encoder(x)
        # decoder
        out = self.decoder(x, encoder_out)
        return out


if __name__ == '__main__':
    model = S2SHSID(dim = 16)
    x = torch.zeros(size=(1,31,512,512))
    out = model(x)
    flops, params = profile(model, inputs=(x,))
    flops = flops / (1e+9)
    params = params/(1e+6)
    print("flops", flops*2)
    print("params", params)
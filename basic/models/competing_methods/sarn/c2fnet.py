from models.blocks import *
import torch
from torch import nn
import numpy as np
from models.modules.coarseNet import *
from models.modules.refineNet import *
from queue import Queue


class C2FNet(nn.Module):
    """Deep residual network with spatial attention for face SR.
    # Arguments:
        - n_ch: base convolution channels
        - down_steps: how many times to downsample in the encoder
        - res_depth: depth of residual layers in the main body
        - up_res_depth: depth of residual layers in each upsample block

    """

    def __init__(
            self,
            feat_num=4,
            opt=None
    ):
        super(C2FNet, self).__init__()
        self.feat_num = feat_num
        self.opt = opt
        self.coarseNet = COARSENet().to(self.opt.device)
        self.refineNet = REFINENet()

    def forward(self, input_img):
        self.img_DN = torch.zeros_like(input_img)
        # FIFO
        F_res = []
        R_res = []
        C = []
        self.coarse = torch.zeros_like(input_img)
        self.R = torch.zeros_like(input_img)
        self.F = torch.zeros_like(input_img)
        device = input_img.device
        for pos in range(input_img.shape[1]):
            if pos == 0:
                input = torch.concat((input_img[:, pos:pos + 1, :, :], input_img[:, pos:pos + 2, :, :]), dim=1).unsqueeze(dim=2)
                n, c, b, h, w = input.shape
                self.F_t, self.C_t, self.R_t = self.coarseNet(input, pos)
                self.C_t_1 = self.R_t_1 = torch.zeros(size=(n, b, 1, h, w)).to(device)
                self.F_res_t = self.C_t - self.F_t
                self.R_res_t = self.C_t - self.R_t

            elif pos == input_img.shape[1]-1:
                input = torch.concat((input_img[:, pos - 1 : pos + 1, :, :], input_img[:, pos:pos + 1, :, :]), dim=1).unsqueeze(dim=2)
                n, c, b, h, w = input.shape
                self.F_t_a1 = self.C_t_a1 = torch.zeros(size=(n, b, 1, h, w)).to(device)
                self.F_t, self.C_t, self.R_t = self.coarseNet(input, pos)
                self.F_res_t = self.C_t - self.F_t
                self.R_res_t = self.C_t - self.R_t
            else:
                input = input_img[:, pos - 1:pos + 2, :, :].unsqueeze(dim=2)
                self.F_t, self.C_t, self.R_t = self.coarseNet(input, pos)
                self.F_res_t = self.C_t - self.F_t
                self.R_res_t = self.C_t - self.R_t


            if pos == 0:
                self.R_res_t_1 = self.C_t_1 - self.R_t_1
                R_res.append(self.R_res_t_1.clone())

            F_res.append(self.F_res_t.clone())
            R_res.append(self.R_res_t.clone())
            C.append(self.C_t.clone())
            self.coarse[:, pos:pos + 1, :, :] = self.C_t.squeeze(dim=2)
            self.R[:, pos:pos + 1, :, :] = self.R_t.squeeze(dim=2)
            self.F[:, pos:pos + 1, :, :] = self.F_t.squeeze(dim=2)

            if pos == input_img.shape[1] - 1:
                self.F_res_t_a1 = self.C_t_a1 - self.F_t_a1
                F_res.append(self.F_res_t.clone())

            if pos !=0:
                DN = self.refineNet(F_res[pos], R_res[pos - 2], C[pos - 1])
                out_img =  C[pos - 1].squeeze(2) + DN
                self.img_DN[:, pos - 1:pos, :, :] = out_img

            if pos == input_img.shape[1] - 1:
                DN = self.refineNet(F_res[pos+1], R_res[pos - 1], C[pos])
                out_img = DN + C[pos].squeeze(2)
                self.img_DN[:, pos:pos + 1, :, :] = out_img

        return self.img_DN,self.coarse,self.F, self.R


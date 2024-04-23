from models.blocks import *
import torch
from torch import nn
import numpy as np
from models.modules.coarseNet import *
from models.modules.refineNet_gray import *
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
            opt=None
    ):
        super(C2FNet, self).__init__()
        self.opt = opt
        self.feat_num = self.opt.feat_num
        self.k = self.opt.k
        self.coarseNet = COARSENet(opt = self.opt).to(self.opt.device)
        self.refineNet = REFINENet(self.opt)

        self.ad_co = self.opt.ad_co
        self.stage =self.opt.stage

    # def get_input(self, input_img,pos):
    #
    #     if pos == 0:
    #         input = torch.concat((input_img[:, pos:pos + 1, :, :], input_img[:, pos:pos + 2, :, :]),
    #                              dim=1).unsqueeze(dim=2)
    #
    #     elif pos == input_img.shape[1] - 1:
    #         input = torch.concat((input_img[:, pos - 1: pos + 1, :, :], input_img[:, pos:pos + 1, :, :]),
    #                              dim=1).unsqueeze(dim=2)
    #     else:
    #         input = input_img[:, pos - 1:pos + 2, :, :].unsqueeze(dim=2)
    #
    #     return input
    def get_input(self, input_img,pos):

        if pos == 0:
            input = torch.concat((input_img[:, pos:pos + 1, :, :], input_img[:, pos:pos + 2, :, :]),
                                 dim=1)

        elif pos == input_img.shape[1] - 1:
            input = torch.concat((input_img[:, pos - 1: pos + 1, :, :], input_img[:, pos:pos + 1, :, :]),
                                 dim=1)
        else:
            input = input_img[:, pos - 1:pos + 2, :, :]

        return input

    def get_back(self, input_img, pos):
        c = input_img.shape[1]
        if pos <= self.k:
            back = input_img[:, 0: pos, :, :]
        elif pos > self.k and pos < c - self.k:
            back = input_img[:, pos - self.k: pos, :, :]
        else:
            back = input_img[:, c - 2 * self.k - 1: pos, :, :]
        return  back

    def get_back_D(self, input, pos):
        c = input.shape[1]
        if pos <= self.k:
            back = input[:, 0: pos, :, :]
        elif pos > self.k and pos < c - self.k:
            back = input[:, pos - self.k: pos, :, :] - input[:, pos: pos + 1, :,:]
        else:
            back = input[:, c - 2 * self.k - 1: pos, :, :] -input[:,pos: pos+1, :,:]
        return  back

    def get_forth(self, input_img, pos):
        c = input_img.shape[1]
        if pos <= self.k:
            forth = input_img[:, pos + 1: 2 * self.k + 1, :, :]
        elif pos > self.k and pos < c - self.k:
            forth = input_img[:, pos + 1: pos + self.k + 1, :, :]
        else:
            forth = input_img[:, pos + 1: c:, :, :]

        return forth

    def get_forth_D(self, input, pos):
        c = input.shape[1]

        D = input[:, pos: pos+1, :, :]
        if pos <= self.k:
            forth = input[:, pos + 1: 2 * self.k + 1, :, :] -D
        elif pos > self.k and pos < c - self.k:
            forth = input[:, pos + 1: pos + self.k + 1, :, :]-D
        else:
            forth = input[:, pos + 1: c:, :, :]-D


        return forth



    def c2f_2stage(self,input_img):
        n, c, h, w = input_img.shape
        device = input_img.device
        self.img_DN = torch.zeros_like(input_img)
        # FIFO

        self.coarse = torch.zeros_like(input_img)
        self.R = torch.zeros_like(input_img)
        self.F = torch.zeros_like(input_img)
        if self.ad_co:
            # ------------------stage-1--------------------------
            if self.stage == 1:
                for pos in range(c):
                    # ------------------coarse--------------------------
                    input = self.get_input(input_img, pos)
                    self.F_t, self.C_t, self.R_t = self.coarseNet(input, pos)
                    # self.F_t, self.C_t, self.R_t = self.F_t.squeeze(dim=2), self.C_t.squeeze(dim=2), self.R_t.squeeze(dim=2)
                    self.F_t, self.C_t, self.R_t = self.F_t, self.C_t, self.R_t
                    self.coarse[:, pos:pos + 1, :, :] = self.C_t.clone()
                    self.R[:, pos:pos + 1, :, :] = self.R_t.clone()
                    self.F[:, pos:pos + 1, :, :] = self.F_t.clone()

                # print(self.C_total)
                return self.coarse, self.F,  self.R
            # ------------------end_stage-1--------------------------
            else:
                # ------------------stage-2--------------------------
                for pos in range(c):
                    # ------------------coarse--------------------------
                    input = self.get_input(input_img,pos)
                    self.F_t, self.C_t, self.R_t = self.coarseNet(input, pos)

                    # ------------------data_prepare_refine--------------------------

                    # self.F_t, self.C_t, self.R_t = self.F_t.squeeze(dim=2), self.C_t.squeeze(dim=2), self.R_t.squeeze(dim=2)
                    # self.F_t, self.C_t, self.R_t = self.F_t, self.C_t, self.R_t
                    # sum_F[:,pos:pos+1,:,:] = self.F_t.clone()
                    # sum_R[:,pos:pos+1,:,:] = self.R_t.clone()
                    # sum_F_T.append(sum_F.clone())
                    # sum_R_T.append(sum_R.clone())

                    self.coarse[:, pos:pos + 1, :, :] = self.C_t.clone()
                    self.R[:, pos:pos + 1, :, :] = self.R_t.clone()
                    self.F[:, pos:pos + 1, :, :] = self.F_t.clone()


                sum_R_flip = torch.flip(self.R,dims=[0])

                sum_F = torch.cumsum(self.F,dim=0)

                sum_R = torch.cumsum(sum_R_flip,dim=0)
                sum_R = torch.flip(sum_R,dims=[0])

                # ------------------begining_refine--------------------------
                for pos in range(c):
                    S_F,S_R = self.get_forth(self.coarse,pos),self.get_back(self.coarse,pos)
                    sum_R_D = self.get_back_D(sum_R,pos)
                    sum_F_D = self.get_forth_D(sum_F,pos)

                    R_res_k = S_R + sum_R_D
                    F_res_k = S_F - sum_F_D

                    current_band = self.coarse[:, pos:pos + 1, :, :]

                    current_concat = torch.concat((R_res_k,current_band,F_res_k),dim=1)

                    avg_current = torch.mean(torch.sum(current_concat,dim=1,keepdim=True),dim=1,keepdim=True)

                    DN = self.refineNet(avg_current)
                    out_img = current_band + DN
                    self.img_DN[:, pos:pos + 1, :, :] = out_img

                return self.img_DN, self.coarse, self.F, self.R
            # ------------------end-stage-2--------------------------
        else:
            for pos in range(c):
                input = input_img[:, pos:pos + 1, :, :]
                DN = self.refineNet(input)
                out_img = input + DN
                self.img_DN[:, pos:pos + 1, :, :] = out_img
            return self.img_DN

    def forward(self, input_img):
        if self.ad_co:
            if self.stage ==1:
                coarse, F , R = self.c2f_2stage(input_img)
                return coarse,F, R
            else:
                img_DN, coarse, F, R = self.c2f_2stage(input_img)
                return img_DN, coarse, F, R
        else:
            img_DN = self.c2f_2stage(input_img)
            return img_DN





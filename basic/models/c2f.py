from models.blocks import *
import torch
from torch import nn
import numpy as np
from models.modules.coarseNet_cor import *
from models.modules.refineNet import *
from models.modules.detector import *
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
        # # if opt.ad_co != 0:
        # self.coarseNet = COARSENet(opt = self.opt)
        # self.refineNet = REFINENet(opt = self.opt)
        if self.opt.refine == "SST" or self.opt.refine == "SERT" or self.opt.refine == "DPHSIR":
            self.refineNet = denoiser(self.opt.refine,k=0)
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

    def get_back(self, input_img, pos,num):
        c = input_img.shape[1]
        if pos <= num:
            back = input_img[:, 0: pos, :, :]
        elif pos > num and pos < c - num:
            back = input_img[:, pos - num: pos, :, :]
        else:
            back = input_img[:, c - 2 * num - 1: pos, :, :]
        return  back

    def get_back_D(self, input, pos,num):
        c = input.shape[1]
        if pos <= num:
            back = input[:, 0: pos, :, :]
        elif pos > num and pos < c - num:
            back = input[:, pos - num: pos, :, :] - input[:, pos: pos + 1, :,:]
        else:
            back = input[:, c - 2 * num - 1: pos, :, :] -input[:,pos: pos+1, :,:]
        return  back

    def get_forth(self, input_img, pos,num):
        c = input_img.shape[1]
        if pos <= num:
            forth = input_img[:, pos + 1: 2 * num + 1, :, :]
        elif pos > num and pos < c - num:
            forth = input_img[:, pos + 1: pos + num + 1, :, :]
        else:
            forth = input_img[:, pos + 1: c:, :, :]

        return forth

    def get_forth_D(self, input, pos,num):
        c = input.shape[1]

        D = input[:, pos: pos+1, :, :]
        if pos <= num:
            forth = input[:, pos + 1: 2 * num + 1, :, :] -D
        elif pos > num and pos < c - num:
            forth = input[:, pos + 1: pos + num + 1, :, :]-D
        else:
            forth = input[:, pos + 1: c:, :, :]-D


        return forth



    def c2f_2stage(self,input_img,img_GT_F=None,img_GT_R=None):
        n, c, h, w = input_img.shape
        device = input_img.device
        self.img_DN = torch.zeros_like(input_img)
        # FIFO
        if self.opt.ad_co == 1:
            self.coarse = torch.zeros_like(input_img)
            self.R = torch.zeros_like(input_img)
            self.F = torch.zeros_like(input_img)
        if self.opt.refine == "Partial_Dnet":
            # noise_map = torch.zeros_like(input_img).to(device)
            # for pos in range(c):
            #     from .modules.Partial_Dnet.Partial_Dnet import predict_noise_map
            #     estimator = predict_noise_map().to(device)
            #     input = input_img[:, pos:pos + 1, :, :]
            #     noise_map_C = estimator(input)
            #     noise_map[:,pos:pos+1,:,:] = noise_map_C

            from .modules.Partial_Dnet.Partial_Dnet import predict_noise_map
            c = input_img.shape[1]
            path ="/home/jiahua/HSI-Group/HSI-MM/models/modules/Partial_Dnet/predict.pth"
            estimator = predict_noise_map().to(device)
            estimator.load_state_dict(torch.load(path))
            estimator.eval()
            noise_map_predict = torch.ones_like(input_img).to(device)
            noise_map = torch.zeros_like(input_img).to(device)

            for i in range(c):
                out = estimator(input_img[:,i:i+1,:,:])
                noise_map[:,i:i+1,:,:] = torch.mean(out)*noise_map_predict[:,i:i+1,:,:]


        if self.ad_co == 1:
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

                    if self.opt.coarse==1:
                        if self.opt.align == 1:
                            self.F_t, self.C_t, self.R_t = self.coarseNet(input, pos)
                            self.coarse[:, pos:pos + 1, :, :] = self.C_t.clone()
                            self.R[:, pos:pos + 1, :, :] = self.R_t.clone()
                            self.F[:, pos:pos + 1, :, :] = self.F_t.clone()
                        else:
                            self.C_t = self.coarseNet(input, pos)
                            self.coarse[:, pos:pos + 1, :, :] = self.C_t.clone()

                    else:
                        self.F_t,self.R_t = self.coarseNet(input, pos)
                        self.R[:, pos:pos + 1, :, :] = self.R_t.clone()
                        self.F[:, pos:pos + 1, :, :] = self.F_t.clone()
                    # ------------------data_prepare_refine--------------------------

                if self.opt.align == 1:

                        sum_R_flip = torch.flip(self.R,dims=[1])

                        sum_F = torch.cumsum(self.F,dim=1)

                        sum_R = torch.cumsum(sum_R_flip,dim=1)
                        sum_R = torch.flip(sum_R,dims=[1])

                # ------------------begining_refine--------------------------
                for pos in range(c):

                    if self.opt.coarse==1:
                        S_F, S_R = self.get_forth(self.coarse, pos, num=self.k), self.get_back(self.coarse, pos, num=self.k)
                        current_band = self.coarse[:, pos:pos + 1, :, :]
                    else:
                        S_F, S_R = self.get_forth(input_img, pos, num=self.k), self.get_back(input_img, pos,
                                                                                               num=self.k)
                        current_band = input_img[:, pos:pos + 1, :, :]
                    if self.opt.align ==1:

                        sum_R_D = self.get_back_D(sum_R,pos,num=self.k)
                        sum_F_D = self.get_forth_D(sum_F,pos,num=self.k)

                        R_res_k = S_R + sum_R_D
                        F_res_k = S_F - sum_F_D
                    else:
                        R_res_k = S_R
                        F_res_k = S_F

                    if self.opt.refine == "Partial_Dnet":
                        noise_map_C = noise_map[:, pos:pos + 1, :, :]
                        noise_map_adj_F, noise_map_adj_R = self.get_forth(noise_map, pos, num=self.k), self.get_back(noise_map, pos, num=self.k)
                        noise_map_adj = torch.concat((noise_map_adj_F,noise_map_adj_R),dim=1)
                        DN = self.refineNet(R_res_k, F_res_k, current_band,noise_map_C,noise_map_adj)
                    else:
                        DN = self.refineNet(R_res_k,F_res_k, current_band)
                    out_img = DN
                    self.img_DN[:, pos:pos + 1, :, :] = out_img
                if self.opt.coarse==1:
                    if self.opt.align == 1:
                        return self.img_DN, self.coarse, self.F, self.R
                    else:
                        return self.img_DN, self.coarse

                else: return self.img_DN, self.F, self.R
            # ------------------end-stage-2--------------------------
        else:
            if self.opt.refine == "SST" or self.opt.refine == "SERT":

                self.refineNet = self.refineNet.to(device)
                self.img_DN = self.refineNet(input_img)
                return self.img_DN
            if self.opt.refine == "DPHSIR" or self.opt.refine == "QRNN3D":
                self.refineNet = self.refineNet.to(device)
                self.img_DN = self.refineNet(torch.unsqueeze(input_img,dim=1))
                self.img_DN = torch.squeeze(self.img_DN,dim=1)
                return self.img_DN
            for pos in range(c):
                # print(img_GT_F)
                S_F, S_R = self.get_forth(input_img, pos, num=self.k), self.get_back(input_img, pos, num=self.k)
                if self.opt.align == 1:
                    # S_F_res,S_R_res = self.get_forth(img_GT_F,pos,num=self.k),self.get_back(img_GT_R,pos,num=self.k)
                    sum_R_flip = torch.flip(img_GT_R, dims=[1])

                    sum_F = torch.cumsum(img_GT_F, dim=1)

                    sum_R = torch.cumsum(sum_R_flip, dim=1)
                    sum_R = torch.flip(sum_R, dims=[1])

                    sum_R_D = self.get_back_D(sum_R, pos, num=self.k)
                    sum_F_D = self.get_forth_D(sum_F, pos, num=self.k)

                    R_res_k = S_R +  sum_R_D
                    F_res_k = S_F - sum_F_D

                    S_F, S_R = F_res_k, R_res_k

                input = input_img[:, pos:pos + 1, :, :]

                if self.opt.refine == "Partial_Dnet":

                    noise_map_C = noise_map[:, pos:pos + 1, :, :]
                    noise_map_adj_F, noise_map_adj_R = self.get_forth(noise_map, pos, num=self.k), self.get_back(
                        noise_map, pos, num=self.k)
                    noise_map_adj = torch.concat((noise_map_adj_F, noise_map_adj_R), dim=1)
                    DN = self.refineNet(S_R, S_F, input, noise_map_C, noise_map_adj)
                else:
                    DN = self.refineNet(S_R,S_F, input)
                out_img =  DN
                self.img_DN[:, pos:pos + 1, :, :] = out_img
            return self.img_DN

    def forward(self, input_img,img_GT_F=None,img_GT_R=None):
        if self.ad_co:
            if self.stage ==1:
                coarse, F , R = self.c2f_2stage(input_img)
                return coarse,F, R
            else:
                if self.opt.coarse==1:
                    if self.opt.align ==1:
                        img_DN, coarse, F, R = self.c2f_2stage(input_img)
                        return  img_DN, coarse, F, R
                    else:
                        img_DN, coarse = self.c2f_2stage(input_img)
                        return img_DN, coarse

                else:
                    img_DN, F, R = self.c2f_2stage(input_img)
                    return img_DN, F, R
        else:
            img_DN = self.c2f_2stage(input_img,img_GT_F,img_GT_R)
            return img_DN


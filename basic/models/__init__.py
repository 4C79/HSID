from .SST_total_mask import SST_total_mask
from .SST_sptoken import SST_sptoken
from .SST_mask import SST_mask
from .sert import SERT
from .competing_methods import *
from .FIDNet import FIDNet
from .FIDNet_1 import FIDNet_1
from .FIDNet_2 import FIDNet_2
from .FIDNet_3 import FIDNet_3
from .SCAT_1 import SCAT_1
from .scat import SCAT
from .plug import Plug
from .man import MAN,ASC
from .SM_CNN import SMCNN
from .HSID import HSID

from .mamba import MambaIRUNet
from .mmbd import MambaDNet
from .MambaIR import MambaIR
from .SST_mask import SST_mask
from .mambad import MambaDe,rnnMamba
from .videomamba import VideoMamba

""" Models
"""


def rnnmmd():
    net = rnnMamba(
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=32,
        depths=(3, 3, 3),
        drop_rate=0.,
        d_state = 16,
        mlp_ratio=2.,
        drop_path_rate=0.1,
        resi_connection='3conv')
    net.use_2dconv = True
    net.bandwise = False
    return net 

def vmamba():
    net = VideoMamba( 
            img_size=512, 
            patch_size=16, 
            depth=24, 
            embed_dim=64, 
            channels=1, 
            drop_path_rate=0.,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True,
            pool_type="cls+avg",
            # video
            kernel_size=1, 
            num_frames=31, 
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
            # clip,
            clip_decoder_embed_dim=768,
            clip_output_dim=512,
            clip_return_layer=1,
            clip_student_return_interval=1,
            add_pool_norm=True,
    )    
    net.use_2dconv = True
    net.bandwise = False
    return net 

def mmd_s():
    net = MambaDe(
        img_size=64,
        patch_size=1,
        in_chans=31,
        embed_dim=96,
        depths=(3,3),
        drop_rate=0.,
        d_state = 16,
        mlp_ratio=2.,
        drop_path_rate=0.1,
        resi_connection='3conv')
    net.use_2dconv = True
    net.bandwise = False
    return net 

def mmd():
    net = MambaDe(
        img_size=64,
        patch_size=1,
        in_chans=31,
        embed_dim=92,
        depths=(4,4,4),
        drop_rate=0.,
        d_state = 16,
        mlp_ratio=2.,
        drop_path_rate=0.1,
        resi_connection='3conv')
    net.use_2dconv = True
    net.bandwise = False
    return net 

def mmd_wdc():
    net = MambaDe(
        img_size=64,
        patch_size=1,
        in_chans=191,
        embed_dim=191,
        depths=(4,4,4),
        drop_rate=0.,
        d_state = 16,
        mlp_ratio=2.,
        drop_path_rate=0.1,
        resi_connection='3conv')
    net.use_2dconv = True
    net.bandwise = False
    return net 

def mmbir_s():
    net = MambaIR(
        img_size=64,
        patch_size=1,
        in_chans=31,
        embed_dim=96,
        depths=(3, 3, 3),
        drop_rate=0.,
        d_state = 16,
        mlp_ratio=2.,
        drop_path_rate=0.1,
        resi_connection='3conv')
    net.use_2dconv = True
    net.bandwise = False
    return net 

def mmbir():
    net = MambaIR(
        img_size=64,
        patch_size=1,
        in_chans=31,
        embed_dim=96,
        depths=(6, 6, 6, 6),
        drop_rate=0.,
        d_state = 16,
        mlp_ratio=2.,
        drop_path_rate=0.1)
    net.use_2dconv = True
    net.bandwise = False
    return net  

def mmbd():
    net = MambaDNet(
        inp_channels=31,
        out_channels=31,
        dim=32,
        num_blocks=[2, 3, 3, 4],
        num_refinement_blocks=4,
        mlp_ratio=2.,
        bias=False,
        dual_pixel_task=False
    )
    net.use_2dconv = True
    net.bandwise = False
    return net  

def mamba():
    net = MambaIRUNet(
        inp_channels=31,
        out_channels=31,
        dim=32,
        num_blocks=[2, 3, 3, 4],
        num_refinement_blocks=4,
        mlp_ratio=2.,
        bias=False,
        dual_pixel_task=False
    )
    net.use_2dconv = True
    net.bandwise = False
    return net  

def mamba_basic():
    net = MambaIRUNet(
        inp_channels=31,
        out_channels=31,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        mlp_ratio=2.,
        bias=False,
        dual_pixel_task=False
    )
    net.use_2dconv = True
    net.bandwise = False
    return net  


def hsid():
    net = HSID()
    net.use_2dconv = True
    net.bandwise = False
    return net  

def spar():
    net = SPARNet()
    net.use_2dconv = True
    net.bandwise = False
    return net

def smcnn():
    net = SMCNN(20, 20, 60,24)
    net.use_2dconv = True
    net.bandwise = False
    return net

def man_s():
    net = MAN(1, 12, 5, [1, 3], Fusion=ASC)
    net.use_2dconv = False
    net.bandwise = False
    return net


def man_m():
    net = MAN(1, 16, 5, [1, 3], Fusion=ASC)
    net.use_2dconv = False
    net.bandwise = False
    return net


def man_l():
    net = MAN(1, 20, 5, [1, 3], Fusion=ASC)
    net.use_2dconv = False
    net.bandwise = False
    return net

def plug():
    net = Plug()
    net.use_2dconv = True
    net.bandwise = False
    return net

def hsdt():
    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net

def hsdt_s():
    net = HSDT(1, 8, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net

def s2s(opt):
    net = S2SHSID(opt)
    net.use_2dconv = True
    net.bandwise = False
    return net

def scat():
    net = SCAT(inp_channels=31,dim = 96,
                num_heads=[ 6,6,6],        
                depths=[ 6,6,6], 
                window_sizes=[32,32,32] ,
                token_sizes=[8,8,8] ,     
                channel_scales=[3,3,3],
                split_sizes=[1,2,4],
                mlp_ratio=2,
                weight_factor=0.8 )

    net.use_2dconv = True     
    net.bandwise = False          
    return net


def scat_1():
    net = SCAT_1(inp_channels=31,dim = 96,
                num_heads=[ 6,6,6],        
                depths=[ 6,6,6], 
                window_sizes=[32,32,32] ,
                token_sizes=[8,8,8] ,     
                channel_scales=[3,3,3],
                split_sizes=[1,2,4],
                mlp_ratio=2,
                weight_factor=0.8)

    net.use_2dconv = True     
    net.bandwise = False          
    return net

def fidnet_1():
    net = FIDNet_1() 
    net.use_2dconv = True
    net.bandwise = False
    return net

def fidnet():
    net = FIDNet(hid_S = 32,
                 hid_T = 8) 
    net.use_2dconv = True
    net.bandwise = False
    return net

def fidnet_64_8():
    net = FIDNet(hid_S = 64,
                 hid_T = 8) 
    net.use_2dconv = True
    net.bandwise = False
    return net

def fidnet_64_16():
    net = FIDNet(hid_S = 64,
                 hid_T = 16) 
    net.use_2dconv = True
    net.bandwise = False
    return net

def fidnet_c():
    net = FIDNet_1(hid_S = 16,
                 hid_T = 16) 
    net.use_2dconv = True
    net.bandwise = False
    return net

def fidnet_d():
    net = FIDNet_2(hid_S = 16,
                 hid_T = 16) 
    net.use_2dconv = True
    net.bandwise = False
    return net

def fidnet_t():
    net = FIDNet_3(hid_S = 16,
                 hid_T = 16) 
    net.use_2dconv = True
    net.bandwise = False
    return net

def sst_mask():
    net = SST_mask(inp_channels=31,dim =120, # 6.6 160->180 6.14 128-160 7.7 160-90
        window_size=8,
        depths=[6,6,6,6,6,6],     # 7.7 12-6
        num_heads=[6,6,6,6,6,6],mlp_ratio=2)   #7.7 8-6
    net.use_2dconv = True
    net.bandwise = False
    return net

def sst_total_mask():
    net = SST_total_mask(inp_channels=31,dim =90, # 6.6 160->180 6.14 128-160 7.7 160-90
        window_size=8,
        depths=[6,6,6,6,6,6],     # 7.7 12-6
        num_heads=[6,6,6,6,6,6],mlp_ratio=2)   #7.7 8-6
    net.use_2dconv = True
    net.bandwise = False
    return net

def sst_sptoken():
    net = SST_sptoken(inp_channels=31,dim =90, # 6.6 160->180 6.14 128-160 7.7 160-90
        window_size=8,
        depths=[6,6,6,6,6,6],     # 7.7 12-6
        num_heads=[6,6,6,6,6,6],mlp_ratio=2)   #7.7 8-6
    net.use_2dconv = True
    net.bandwise = False
    return net

def sert_base():
    net = SERT(inp_channels=31,dim = 96,         window_sizes=[16,32,32] ,        depths=[ 6,6,6],         num_heads=[ 6,6,6],split_sizes=[1,2,4],mlp_ratio=2,weight_factor=0.1,memory_blocks=128,down_rank=8)     #16,32,32

    net.use_2dconv = True     
    net.bandwise = False          
    return net

def sert_tiny():
    net = SERT(inp_channels=31,dim = 96,         window_sizes=[16,32] ,        depths=[ 4,4],         num_heads=[ 6,6],split_sizes=[2,4],mlp_ratio=2,weight_factor=0.1,memory_blocks=128,down_rank=8)     #16,32,32
    net.use_2dconv = True     
    net.bandwise = False          
    return net

def sert_small():
    net = SERT(inp_channels=31,dim = 96,         window_sizes=[16,32,32] ,        depths=[ 4,4,4],         num_heads=[ 6,6,6],split_sizes=[1,2,4],mlp_ratio=2,weight_factor=0.1,memory_blocks=128,down_rank=8)     #16,32,32
    net.use_2dconv = True     
    net.bandwise = False          
    return net

def sert_urban():
    net = SERT(inp_channels=210,dim = 96*2,         window_sizes=[8,16,16] ,        depths=[ 6,6,6],         num_heads=[ 6,6,6],split_sizes=[2,4,4],mlp_ratio=2,down_rank=8,memory_blocks=128)  
    net.use_2dconv = True     
    net.bandwise = False          
    return net


def sert_real():
    net = SERT(inp_channels=34,dim = 96,         window_sizes=[16,32,32] ,        depths=[6,6,6],down_rank=8,         num_heads=[ 6,6,6],split_sizes=[1,2,4],mlp_ratio=2,memory_blocks=64)

    net.use_2dconv = True     
    net.bandwise = False          
    return net

def qrnn3d():
    net = QRNNREDC3D(1, 16, 5, [1, 3], has_ad=True)
    net.use_2dconv = False
    net.bandwise = False
    return net

def grn_net():
    net = U_Net_GR(in_ch=31,out_ch=31)
    net.use_2dconv = True
    net.bandwise = False
    return net


def grn_net_real():
    net = U_Net_GR(in_ch=34,out_ch=34)
    net.use_2dconv = True
    net.bandwise = False
    return net

def grn_net_urban():
    net = U_Net_GR(in_ch=210,out_ch=210)
    net.use_2dconv = True
    net.bandwise = False
    return net


def t3sc():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('models/competing_methods/T3SC/layers/t3sc.yaml')
    net = MultilayerModel(**cfg.params)
    net.use_2dconv = True
    net.bandwise = False
    return net

def t3sc_real():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('models/competing_methods/T3SC/layers/t3sc_real.yaml')
    net = MultilayerModel(**cfg.params)
    net.use_2dconv = True
    net.bandwise = False
    return net

def t3sc_urban():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('models/competing_methods/T3SC/layers/t3sc_urban.yaml')
    net = MultilayerModel(**cfg.params)
    net.use_2dconv = True
    net.bandwise = False
    return net

def macnet():
    net = MACNet(in_channels=1,channels=16,num_half_layer=5)
    net.use_2dconv = True
    net.bandwise = False          
    return net 

def sst():
    net = SST(inp_channels=31,dim = 90,
        window_size=8,
        depths=[ 6,6,6,6,6,6],
        num_heads=[ 6,6,6,6,6,6],mlp_ratio=2)
    net.use_2dconv = True
    net.bandwise = False
    return net

def sst_real():
    net = SST(inp_channels=34,depths=[6,6,6])
    net.use_2dconv = True
    net.bandwise = False
    return net

def sst_urban():
    net = SST(inp_channels=210,dim = 210,
        window_size=8,
        depths=[ 6,6,6,6,6,6],
        num_heads=[ 6,6,6,6,6,6],mlp_ratio=2)
    net.use_2dconv = True
    net.bandwise = False
    return net
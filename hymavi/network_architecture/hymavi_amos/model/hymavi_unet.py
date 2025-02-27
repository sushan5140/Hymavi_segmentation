import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("hymavi/network_architecture/hymavi_amos")
from pytorch_3dunet.pytorch3dunet.unet3d.config import load_config
from pytorch_3dunet.pytorch3dunet.unet3d.model import get_model
from pytorch_3dunet.pytorch3dunet.unet3d.buildingblocks import Encoder, Decoder
from model.hymavi import HymaviBlock, HymaSwinBlock, HymaFullStartBlock, HymaFullEndBlock, HymaSwinDecBlock
from config import config as cf
from pytorch_3dunet.pytorch3dunet.unet3d.buildingblocks import ResNetBlock

from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
from model.in_out_conv import StemConv, ExpandingConv





class HymaviFullModel(nn.Module):
    def __init__(self, num_classes=cf.num_classes, num_layers =2):
        super(HymaviFullModel, self).__init__()
        
        self.stem_conv = StemConv(in_channels=1, out_channels=cf.dim_after_stem)
        self.first_resblk = Encoder(in_channels=cf.dim_after_stem, out_channels=cf.dim_after_stem, basic_module=ResNetBlock)
        self.end_resblk = Decoder(in_channels=cf.f_maps_dim[0], out_channels=cf.dim_after_stem, basic_module=ResNetBlock)

        self.full_hyma_start = HymaFullStartBlock(num_layers=2)
        self.full_hyma_end = HymaFullEndBlock(num_layers=2)

        config, path_ = load_config()
        unet = get_model(config['model'])

        self.unet_encoder = unet.encoders
        self.unet_decoder = unet.decoders

        
        self.hmv_blk = HymaviBlock(num_layers=num_layers)

        self.final_conv = nn.Conv3d(cf.dim_after_stem, num_classes, 1)
        if cf.is_do_ds:
            self.out_conv_inter1 = nn.Conv3d(cf.f_maps_dim[0], num_classes,1)
            self.out_conv_inter0 = nn.Conv3d(cf.f_maps_dim[1], num_classes, 1)

        self.up_conv = ExpandingConv(in_channels=cf.dim_after_stem)

        hymaswin_enc = nn.ModuleList()
        for i in range(len(cf.swin_input_resolutions)):
            hymaswin_enc.append(HymaSwinBlock(ith_layer=i, num_layers=num_layers, reduce_rate=cf.reduce_rate_before_hymaviswin[i]))
        self.hymswin_enc = hymaswin_enc

        hymaswin_dec = nn.ModuleList()
        for i in range(len(cf.swin_input_resolutions)):
            hymaswin_dec.append(HymaSwinDecBlock(ith_layer=len(cf.swin_input_resolutions)-1-i, num_layers=num_layers, reduce_rate=cf.reduce_rate_before_hymaviswin[len(cf.swin_input_resolutions)-1-i]))
        self.hymswin_dec = hymaswin_dec

        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x, is_do_ds=False, is_train=False):
        # x_ori = x

        x = self.stem_conv(x)
        x = self.first_resblk(x)
        first_ = x
        x = self.max_pool(x)
        
        
        encoders_features = []

        i_sw = 0
        for idx, encoder in enumerate(self.unet_encoder):
            x = encoder(x)

            #full hyma
            if idx ==0:
                x = self.full_hyma_start(x)

            # reverse the encoder outputs to be aligned with the decoder
            if idx >= ((len(cf.f_maps_dim)-1) - len(cf.swin_input_resolutions)) and idx<(len(cf.f_maps_dim)-1):
                x = self.hymswin_enc[i_sw](x)
                i_sw +=1
            
            encoders_features.insert(0, x)

            if idx < len(cf.f_maps_dim)-1:
                # x = self.pools[idx](x)
                x = self.max_pool(x)

        x = self.hmv_blk(x)

        encoders_features = encoders_features[1:]

        out_ds = []
        i_sw = 0
        for idx, (decoder, encoder_features) in enumerate(zip(self.unet_decoder, encoders_features)):
            x = decoder(encoder_features, x) 
            if idx<=len(cf.swin_input_resolutions)-1:
                x = self.hymswin_dec[i_sw](x, encoder_features)
                if is_do_ds and (i_sw==1):
                    out_ds.append(x)

                i_sw +=1
            else:
                x = self.full_hyma_end(x, encoder_features)
                if is_do_ds:
                    out_ds.append(x)


        x = self.end_resblk(first_, x)
        x = x + first_

        x_inter = self.up_conv(x)

        x = self.final_conv(x_inter)
        
        if is_do_ds:
            out_ds_final = []

            out_ds_final.append(self.out_conv_inter0(out_ds[0]))
            out_ds_final.append(self.out_conv_inter1(out_ds[1]))
            out_ds_final.append(x)

            x = out_ds_final[::-1]

        if is_train:
            return x, x_inter
        else:
            return x







import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("hymavi/network_architecture/hymavi_amos")
from mam_att.mam_att import SimHymba
from mam_att.mam_att_local import SimHymbaSwin, SimHymbaSwinDec, HymbaRMSNorm
from config import config as cf
from einops import rearrange

def window_partition(x, window_size):

    B, S, H, W, C = x.shape
    
    x = x.view(B, S // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size, S, H, W):
   
    B = int(windows.shape[0] / (S * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, S // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    return x


class IntraviewSwinBlock(nn.Module):
    def __init__(self, ith_layer = None, layer_ord_in_block=0, stride_2d = 1, reduce_rate=None):
        super(IntraviewSwinBlock, self).__init__()
        '''ith_layer: order of the simhymbaswin layer (start from 0 to "total number of simhymbaswin layers")
        '''
        
        self.axial_hb = SimHymbaSwin(embed_dim_att=cf.embed_dim_att_swin[ith_layer], num_attention_heads=cf.num_attention_heads_swin[ith_layer], \
                mamba_states=cf.mamba_states_swin[ith_layer], seq_len= cf.seq_len_swin[ith_layer][0],hidden_size=cf.hidden_sz_swin[ith_layer][0], \
                    value_dim=cf.value_dim_att_swin[ith_layer], window_size=cf.window_sizes[ith_layer], shift_size=cf.swin_shift_sz[ith_layer], ith_layer=ith_layer)
        self.sagittal_hb = SimHymbaSwin(embed_dim_att=cf.embed_dim_att_swin[ith_layer], num_attention_heads=cf.num_attention_heads_swin[ith_layer], \
                mamba_states=cf.mamba_states_swin[ith_layer], seq_len= cf.seq_len_swin[ith_layer][1], hidden_size=cf.hidden_sz_swin[ith_layer][1], \
                    value_dim=cf.value_dim_att_swin[ith_layer], window_size=cf.window_sizes[ith_layer], shift_size=cf.swin_shift_sz[ith_layer], ith_layer=ith_layer)
        self.coronal_hb = SimHymbaSwin(embed_dim_att=cf.embed_dim_att_swin[ith_layer], num_attention_heads=cf.num_attention_heads_swin[ith_layer], \
                mamba_states=cf.mamba_states_swin[ith_layer], seq_len= cf.seq_len_swin[ith_layer][2], hidden_size=cf.hidden_sz_swin[ith_layer][2], \
                    value_dim=cf.value_dim_att_swin[ith_layer], window_size=cf.window_sizes[ith_layer], shift_size=cf.swin_shift_sz[ith_layer], ith_layer=ith_layer)
        self.down_dims = nn.Conv3d(cf.C, cf.C//cf.reduce_rate_before_hymavi ,kernel_size=1)

        self.up_dims = nn.Conv3d(cf.C//cf.reduce_rate_before_hymavi, cf.C, kernel_size=1)

        # dim=cf.f_maps_dim[ith_layer]
        # self.input_resolution=cf.swin_input_resolutions[ith_layer]
        # num_heads=cf.swin_nheads[ith_layer]
        self.window_size=cf.window_sizes[ith_layer]
        self.shift_size=[0,0,0] if layer_ord_in_block==0 else [self.window_size[0]//2, self.window_size[1]//2, self.window_size[2]//2]
        # self.ith_layer = ith_layer

        # dims = cf.f_maps_dim[ith_layer+1]//cf.reduce_rate_before_hymaviswin[ith_layer]
        # self.block_2d = nn.Sequential(
        #     nn.Conv2d(dims, dims, kernel_size = 3, padding=1, stride=stride_2d),
        #     nn.GroupNorm(1, num_channels=dims),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5)

        # )
        # self.dimm = dims
        # self.inter = InterviewIntermediateBlock(in_dims=cf.window_sizes[ith_layer][0]*cf.window_sizes[ith_layer][1] * cf.window_sizes[ith_layer][2], factor=16)
        # print(">>>>>>>>>>>>>>?????????????????? ", reduce_rate)
        self.drop = nn.Dropout(0.4)
        dim = cf.swin_f_map_dims[ith_layer]// reduce_rate
        self.norm_out = nn.LayerNorm(dim)
        self.ln_out = nn.Linear(dim, dim)
        # self.dim = dim
        
    def forward(self, x):
        #x: B, C, D, H, W
        B, C, D, H, W = x.shape
        x = x.permute(0,2,3,4,1)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_g = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]

        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g)) 

        _, Sp, Hp, Wp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1],-self.shift_size[2]), dims=(1, 2,3))
        else:
            shifted_x = x

        # partition windows
        x = window_partition(shifted_x, self.window_size) #(B*n_windows, d, h, w, C)
        b, d, h, w, c = x.shape
        residual = x.contiguous().view(b, -1, c) #(B, DHW, C)

        x = x.permute(0, 4, 1, 2, 3) #(b, c, d ,h, w)

        x_axl = x.permute(0,2,1,3,4).contiguous().view(b, d, -1) #(b,d,chw) = (B,L,D)
        x_sgt = x.permute(0,4,1,2,3).contiguous().view(b, w, -1) #(b, w, cdh)
        x_crn = x.permute(0,3,1,2,4).contiguous().view(b, h, -1) #(b, h, cdw)

        x_axl = self.axial_hb(x_axl) 
        x_axl = x_axl.view(b, d, c, h, w).permute(0,1,3,4,2).contiguous().view(b, -1, c) #(b, dhw, c)

        x_sgt = self.sagittal_hb(x_sgt) 
        x_sgt = x_sgt.view(b, w, c, d, h).permute(0,3,4,1,2).contiguous().view(b, -1, c) #(b, dhw, c)

        x_crn = self.coronal_hb(x_crn)
        x_crn = x_crn.view(b, h, c, d, w).permute(0,3,1,4,2).contiguous().view(b, -1, c) #(b, dhw, c)


        x = (x_axl + x_sgt + x_crn)/3 

        x =x+ self.drop(residual)
        x = x + self.drop(self.ln_out(self.norm_out(x)))

        # merge windows
        x = x.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(x, self.window_size, Sp, Hp, Wp)  # B D H W C
        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        x = x.permute(0,4,1,2,3) #(B, C, D, H, W)
        return x
    
class IntraviewSwinDecBlock(nn.Module):
    def __init__(self, ith_layer = None, layer_ord_in_block=0, stride_2d = 1, reduce_rate = None):
        super(IntraviewSwinDecBlock, self).__init__()
        '''ith_layer: order of the simhymbaswin layer (start from 0 to "total number of simhymbaswin layers")
        '''
        
        self.axial_hb = SimHymbaSwinDec(embed_dim_att=cf.embed_dim_att_swin[ith_layer], num_attention_heads=cf.num_attention_heads_swin[ith_layer], \
                mamba_states=cf.mamba_states_swin[ith_layer], seq_len= cf.seq_len_swin[ith_layer][0],hidden_size=cf.hidden_sz_swin[ith_layer][0], \
                    value_dim=cf.value_dim_att_swin[ith_layer], window_size=cf.window_sizes[ith_layer], shift_size=cf.swin_shift_sz[ith_layer], ith_layer=ith_layer)
        self.sagittal_hb = SimHymbaSwinDec(embed_dim_att=cf.embed_dim_att_swin[ith_layer], num_attention_heads=cf.num_attention_heads_swin[ith_layer], \
                mamba_states=cf.mamba_states_swin[ith_layer], seq_len= cf.seq_len_swin[ith_layer][1], hidden_size=cf.hidden_sz_swin[ith_layer][1], \
                    value_dim=cf.value_dim_att_swin[ith_layer], window_size=cf.window_sizes[ith_layer], shift_size=cf.swin_shift_sz[ith_layer], ith_layer=ith_layer)
        self.coronal_hb = SimHymbaSwinDec(embed_dim_att=cf.embed_dim_att_swin[ith_layer], num_attention_heads=cf.num_attention_heads_swin[ith_layer], \
                mamba_states=cf.mamba_states_swin[ith_layer], seq_len= cf.seq_len_swin[ith_layer][2], hidden_size=cf.hidden_sz_swin[ith_layer][2], \
                    value_dim=cf.value_dim_att_swin[ith_layer], window_size=cf.window_sizes[ith_layer], shift_size=cf.swin_shift_sz[ith_layer], ith_layer=ith_layer)
        # self.down_dims = nn.Conv3d(cf.C, cf.C//cf.reduce_rate_before_hymavi ,kernel_size=1)

        # self.up_dims = nn.Conv3d(cf.C//cf.reduce_rate_before_hymavi, cf.C, kernel_size=1)

        # dim=cf.f_maps_dim[ith_layer]
        # self.input_resolution=cf.swin_input_resolutions[ith_layer]
        # num_heads=cf.swin_nheads[ith_layer]
        self.window_size=cf.window_sizes[ith_layer]
        self.shift_size=[0,0,0] if layer_ord_in_block==0 else [self.window_size[0]//2, self.window_size[1]//2, self.window_size[2]//2]
        # self.ith_layer = ith_layer

        # dims = cf.f_maps_dim[ith_layer+1]//cf.reduce_rate_before_hymaviswin[ith_layer]
        # self.block_2d = nn.Sequential(
        #     nn.Conv2d(dims, dims, kernel_size = 3, padding=1, stride=stride_2d),
        #     nn.GroupNorm(1, num_channels=dims),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5)

        # )
        # self.dimm = dims
        # self.inter = InterviewIntermediateBlock(in_dims=cf.window_sizes[ith_layer][0]*cf.window_sizes[ith_layer][1] * cf.window_sizes[ith_layer][2], factor=16)
        
        self.drop = nn.Dropout(0.4)
        dim = cf.swin_f_map_dims[ith_layer]// reduce_rate
        dim_in = dim//16
        if dim_in<=0:
            dim_in = dim
        self.norm_out = nn.LayerNorm(dim)
        self.ln_out = nn.Sequential(
            nn.Linear(dim, dim_in),
            nn.Linear(dim_in, dim) 
        )
        # self.ith = ith_layer
        # self.dim = dim
        
    def forward(self, x, x_skip):
        #x: B, C, D, H, W
        B, C, D, H, W = x.shape

        x = x.permute(0,2,3,4,1)
        x_skip = x_skip.permute(0,2,3,4,1)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_g = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]

        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g)) 
        x_skip = F.pad(x_skip, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))

        _, Sp, Hp, Wp, _ = x.shape

        # partition windows
        x = window_partition(x, self.window_size) #(B*n_windows, d, h, w, C)
        b, d, h, w, c = x.shape
        # print("::::::::::::::::::::::::!!!!!! ", x.shape)
        residual = x.contiguous().view(b, -1, c) #(B, DHW, C)

        x = x.permute(0, 4, 1, 2, 3) #(b, c, d ,h, w)

        x_skip = window_partition(x_skip, self.window_size)
        x_skip = x_skip.permute(0, 4, 1, 2, 3)

        x_axl = x.permute(0,2,1,3,4).contiguous().view(b, d, -1) #(b,d,chw) = (B,L,D)
        x_sgt = x.permute(0,4,1,2,3).contiguous().view(b, w, -1) #(b, w, cdh)
        x_crn = x.permute(0,3,1,2,4).contiguous().view(b, h, -1) #(b, h, cdw)

        x_axl_skip = x_skip.permute(0,2,1,3,4).contiguous().view(b, d, -1) #(b,d,chw) = (B,L,D)
        x_sgt_skip = x_skip.permute(0,4,1,2,3).contiguous().view(b, w, -1) #(b, w, cdh)
        x_crn_skip = x_skip.permute(0,3,1,2,4).contiguous().view(b, h, -1) #(b, h, cdw)

        x_axl = self.axial_hb(x_axl, x_axl_skip) 
        x_axl = x_axl.view(b, d, c, h, w).permute(0,1,3,4,2).contiguous().view(b, -1, c) #(b, dhw, c)

        x_sgt = self.sagittal_hb(x_sgt, x_sgt_skip) 
        x_sgt = x_sgt.view(b, w, c, d, h).permute(0,3,4,1,2).contiguous().view(b, -1, c) #(b, dhw, c)

        x_crn = self.coronal_hb(x_crn, x_crn_skip)
        x_crn = x_crn.view(b, h, c, d, w).permute(0,3,1,4,2).contiguous().view(b, -1, c) #(b, dhw, c)


        x = (x_axl + x_sgt + x_crn)/3 

        #residual
        x =x+ self.drop(residual)
        x = x + self.drop(self.ln_out(self.norm_out(x)))


        # merge windows
        x = x.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(x, self.window_size, Sp, Hp, Wp)  # B D H W C
        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        x = x.permute(0,4,1,2,3) #(B, C, D, H, W)
        return x

    
class IntraviewBlock(nn.Module):
    def __init__(self, stride_2d = 1):
        super(IntraviewBlock, self).__init__()
        
        self.axial_hb = SimHymba(embed_dim_att=cf.embed_dim_att, num_attention_heads=cf.num_attention_heads, \
                  sequence_len=cf.D, mamba_states=cf.mamba_states, hidden_size=cf.v1_dim//(stride_2d**2), value_dim=cf.value_dim_att)
        self.sagittal_hb = SimHymba(embed_dim_att=cf.embed_dim_att, num_attention_heads=cf.num_attention_heads, \
                  sequence_len=cf.W, mamba_states=cf.mamba_states, hidden_size=cf.v2_dim//(stride_2d**2), value_dim=cf.value_dim_att)
        self.coronal_hb = SimHymba(embed_dim_att=cf.embed_dim_att, num_attention_heads=cf.num_attention_heads, \
                 sequence_len=cf.H, mamba_states=cf.mamba_states, hidden_size=cf.v3_dim//(stride_2d**2), value_dim=cf.value_dim_att)
        self.down_dims = nn.Conv3d(cf.C, cf.C//cf.reduce_rate_before_hymavi ,kernel_size=1)

        self.up_dims = nn.Conv3d(cf.C//cf.reduce_rate_before_hymavi, cf.C, kernel_size=1)

        dims = cf.f_maps_dim[-1]//cf.reduce_rate_before_hymavi
        self.block_2d = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size = 3, padding=1, stride=stride_2d),
            nn.GroupNorm(1, num_channels=dims),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)

        )
        # self.pool_1d = nn.AvgPool1d(kernel_size=2, stride=2)

        self.drop = nn.Dropout(0.4)
        dim = cf.C//cf.reduce_rate_before_hymavi
        self.norm_out = nn.LayerNorm(dim)
        self.ln_out = nn.Sequential(
            nn.Linear(dim, dim//16),
            nn.Linear(dim//16, dim) 
        )
        # self.dim = dim
    def forward(self, x):
        #x: B, C, D, H, W
        B, C, D, H, W = x.shape

        residual = x.permute(0,2,3,4,1).contiguous().view(B, -1, C) #(B, DHW, C) #No

        x_axl = x.permute(0,2,1,3,4).contiguous().view(-1, C, H, W) 
        x_sgt = x.permute(0,4,1,2,3).contiguous().view(-1, C, D, H) 
        x_crn = x.permute(0,3,1,2,4).contiguous().view(-1, C, D, W) 

        x_axl = self.block_2d(x_axl)
        x_sgt = self.block_2d(x_sgt)
        x_crn = self.block_2d(x_crn)


        _, _, D, W = x_crn.shape
        _, _, H, _ = x_axl.shape

        x_axl = x_axl.contiguous().view(B, -1, C, H, W).view(B, -1, C*H*W)
        x_sgt = x_sgt.contiguous().view(B, -1, C, D, H).view(B, -1, C*D*H)
        x_crn = x_crn.contiguous().view(B, -1, C, D, W).view(B, -1, C*D*W) 

        x_axl = self.axial_hb(x_axl)
        x_axl = x_axl.view(B, D, C, H, W).permute(0,1,3,4,2).contiguous().view(B, -1, C) #(B, DHW, C)

        x_sgt = self.sagittal_hb(x_sgt)
        x_sgt = x_sgt.view(B, W, C, D, H).permute(0,3,4,1,2).contiguous().view(B, -1, C) #(B, DHW, C)

        x_crn = self.coronal_hb(x_crn)
        x_crn = x_crn.view(B, H, C, D, W).permute(0,3,1,4,2).contiguous().view(B, -1, C) #(B, DHW, C)

        #residual
        x_axl = x_axl + self.drop(residual)
        x_axl = (x_axl + self.drop(self.ln_out(self.norm_out(x_axl)))).transpose(-1,-2).view(B, C, D, H, W)

        x_sgt = x_sgt + self.drop(residual)
        x_sgt = (x_sgt + self.drop(self.ln_out(self.norm_out(x_sgt)))).transpose(-1,-2).view(B, C, D, H, W)

        x_crn = x_crn + self.drop(residual)
        x_crn = (x_crn + self.drop(self.ln_out(self.norm_out(x_crn)))).transpose(-1,-2).view(B, C, D, H, W)

        return x_axl, x_sgt, x_crn
    
class IntraviewFirstMam(nn.Module):
    def __init__(self, stride_2d = 6, factor=102400):
        super(IntraviewFirstMam, self).__init__()

        
        # dim_first = [cf.first_mam_res[1]*cf.first_mam_res[2]*cf.dim_after_stem1//cf.reduce_rate_first_mam//(stride_2d**2), \
        #              cf.first_mam_res[1]*cf.first_mam_res[0]*cf.dim_after_stem1//cf.reduce_rate_first_mam//(stride_2d**2), \
        #                 cf.first_mam_res[0]*cf.first_mam_res[2]*cf.dim_after_stem1//cf.reduce_rate_first_mam//(stride_2d**2)]
        a,b,c = cf.first_mam_res[0]//stride_2d, cf.first_mam_res[1]//stride_2d, cf.first_mam_res[2]//stride_2d
        dim_first = [b*c*cf.dim_after_stem1//cf.reduce_rate_first_mam, \
                     a*b*cf.dim_after_stem1//cf.reduce_rate_first_mam, \
                        a*c*cf.dim_after_stem1//cf.reduce_rate_first_mam]
        
        self.axial_hb = SimHymba(sequence_len=cf.first_mam_res[0], mamba_states=8, hidden_size=dim_first[0], is_att=False, factor=factor)
        self.sagittal_hb = SimHymba(sequence_len=cf.first_mam_res[2], mamba_states=8, hidden_size=dim_first[1], is_att=False, factor=factor)
        self.coronal_hb = SimHymba(sequence_len=cf.first_mam_res[1], mamba_states=8, hidden_size=dim_first[2], is_att=False, factor=factor)
        
        dims = cf.dim_after_stem1//cf.reduce_rate_first_mam
        self.block_2d = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size = stride_2d, padding=0, stride=stride_2d),
            nn.GroupNorm(1, num_channels=dims),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)

        )     

        self.pool_1d = nn.AvgPool1d(kernel_size=stride_2d, stride=stride_2d)

        dim = dims
        self.drop = nn.Dropout(0.4)
        self.norm_out = nn.LayerNorm(dim)
        self.ln_out = nn.Sequential(
            nn.Linear(dim, dim//16),
            nn.Linear(dim//16, dim) 
        )
        self.dim = dim
        self.stride = stride_2d
        
        
    def forward(self, x):
        #x: B, C, D, H, W
        B, C, D, H, W = x.shape
        x_axl = x.permute(0,2,1,3,4).contiguous().view(-1, C, H, W) 
        x_sgt = x.permute(0,4,1,2,3).contiguous().view(-1, C, D, H) 
        x_crn = x.permute(0,3,1,2,4).contiguous().view(-1, C, D, W) 

        x_axl = self.block_2d(x_axl)
        x_sgt = self.block_2d(x_sgt)
        x_crn = self.block_2d(x_crn)


        _, _, D, W = x_crn.shape
        _,_,H,W = x_axl.shape

        x_axl = x_axl.contiguous().view(B, -1, C, H, W).view(B, -1, C*H*W) #(B,D_old,CHW) = (B,L,D)
        x_sgt = x_sgt.contiguous().view(B, -1, C, D, H).view(B, -1, C*D*H) #(B, W_old, CDH)
        x_crn = x_crn.contiguous().view(B, -1, C, D, W).view(B, -1, C*D*W) #(B, H_old, CDW)

        residual_axl = x_axl
        residual_sgt = x_sgt
        residual_crn = x_crn

        x_axl = self.axial_hb(x_axl) #(2,4,8192)
        x_axl = x_axl + self.drop(residual_axl)
        x_axl = rearrange(x_axl, 'b d_old (c h w) -> b (d_old h w) c', b=B, c=C, h=H, w=W)
        x_axl = x_axl + self.drop(self.ln_out(self.norm_out(x_axl)))
        x_axl = rearrange(x_axl, 'b (d_old h w) c -> b (c h w) d_old', b=B, c=C, h=H, w=W)
        x_axl = self.pool_1d(x_axl) #reduce from old D to new D
        x_axl = rearrange(x_axl, 'b (c h w) d -> b c d h w', b=B, c=C, h=H, w=W)
        
        x_sgt = self.sagittal_hb(x_sgt) #(2,16,2048)
        x_sgt = x_sgt + self.drop(residual_sgt)
        x_sgt = rearrange(x_sgt, 'b w_old (c d h) -> b (w_old d h) c', b=B, c=C, h=H, d=D)
        x_sgt = x_sgt + self.drop(self.ln_out(self.norm_out(x_sgt)))
        x_sgt = rearrange(x_sgt, 'b (w_old d h) c -> b (c d h) w_old', b=B, c=C, h=H, d=D)
        x_sgt = self.pool_1d(x_sgt) #reduce from old W to new W
        x_sgt = rearrange(x_sgt, 'b (c d h) w -> b c d h w', b=B, c=C, h=H, d=D)

        x_crn = self.coronal_hb(x_crn) #(2,16,2048)
        x_crn = x_crn + self.drop(residual_crn)
        x_crn = rearrange(x_crn, 'b h_old (c d w) -> b (h_old d w) c', b=B, c=C, d=D, w=W)
        x_crn = x_crn + self.drop(self.ln_out(self.norm_out(x_crn)))
        x_crn = rearrange(x_crn, 'b (h_old d w) c -> b (c d w) h_old', b=B, c=C, d=D, w=W)
        x_crn = self.pool_1d(x_crn) #reduce from old H to new H
        x_crn = rearrange(x_crn, 'b (c d w) h -> b c d h w', b=B, c=C, d=D, w=W)

        return x_axl, x_sgt, x_crn

class IntraviewEndMam(nn.Module):
    def __init__(self, stride_2d = 6, factor=102400):
        super(IntraviewEndMam, self).__init__()
        '''
        each block has multiple layers, the first layer is for cross-att, and the remainings are for self-att
        Therefore, is_start_layer marks if the layer is the first layer of the block -> cross-att
        '''
        # dim_end = [cf.end_mam_res[1]*cf.end_mam_res[2]*cf.dim_after_stem1//cf.reduce_rate_first_mam//(stride_2d**2), \
        #              cf.end_mam_res[1]*cf.end_mam_res[0]*cf.dim_after_stem1//cf.reduce_rate_first_mam//(stride_2d**2), \
        #                 cf.end_mam_res[0]*cf.end_mam_res[2]*cf.dim_after_stem1//cf.reduce_rate_first_mam//(stride_2d**2)]
        a,b,c = cf.first_mam_res[0]//stride_2d, cf.first_mam_res[1]//stride_2d, cf.first_mam_res[2]//stride_2d
        dim_end = [b*c*cf.dim_after_stem1//cf.reduce_rate_end_mam, \
                     a*b*cf.dim_after_stem1//cf.reduce_rate_end_mam, \
                        a*c*cf.dim_after_stem1//cf.reduce_rate_end_mam]
        
        self.axial_hb = SimHymba(sequence_len=cf.end_mam_res[0], mamba_states=8, hidden_size=dim_end[0], is_att=False, factor=factor)
        self.sagittal_hb = SimHymba(sequence_len=cf.end_mam_res[2], mamba_states=8, hidden_size=dim_end[1], is_att=False, factor=factor)
        self.coronal_hb = SimHymba(sequence_len=cf.end_mam_res[1], mamba_states=8, hidden_size=dim_end[2], is_att=False, factor=factor)
        
        dims = cf.dim_after_stem1//cf.reduce_rate_end_mam
        self.block_2d = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size = stride_2d, padding=0, stride=stride_2d),
            nn.GroupNorm(1, num_channels=dims),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)

        )
        self.pool_1d = nn.AvgPool1d(kernel_size=stride_2d, stride=stride_2d)
        
        dim = dims
        self.drop = nn.Dropout(0.4)
        self.norm_out = nn.LayerNorm(dim)
        self.ln_out = nn.Sequential(
            nn.Linear(dim, dim//16),
            nn.Linear(dim//16, dim) 
        )
        self.dim = dim
        self.stride = stride_2d
    def forward(self, x, x_skip=None):
        #x: B, C, D, H, W
        ''' x_skip of end_mam is None'''
        B, C, D, H, W = x.shape
        
        if x_skip==None:
            x_axl = x.permute(0,2,1,3,4).contiguous().view(-1, C, H, W) 
            x_sgt = x.permute(0,4,1,2,3).contiguous().view(-1, C, D, H) 
            x_crn = x.permute(0,3,1,2,4).contiguous().view(-1, C, D, W)

            x_axl = self.block_2d(x_axl)
            x_sgt = self.block_2d(x_sgt)
            x_crn = self.block_2d(x_crn)

            _, _, D, W = x_crn.shape
            # _, _, H, _ = x_axl.shape
            _,_,H,W = x_axl.shape

            x_axl = x_axl.contiguous().view(B, -1, C, H, W).view(B, -1, C*H*W) #(B,D_old,CHW) = (B,L,D)
            x_sgt = x_sgt.contiguous().view(B, -1, C, D, H).view(B, -1, C*D*H) #(B, W_old, CDH)
            x_crn = x_crn.contiguous().view(B, -1, C, D, W).view(B, -1, C*D*W) #(B, H_old, CDW)

            residual_axl = x_axl
            residual_sgt = x_sgt
            residual_crn = x_crn

            x_axl = self.axial_hb(x_axl) #(2,4,8192)
            x_axl = x_axl + self.drop(residual_axl)
            x_axl = rearrange(x_axl, 'b d_old (c h w) -> b (d_old h w) c', b=B, c=C, h=H, w=W)
            x_axl = x_axl + self.drop(self.ln_out(self.norm_out(x_axl)))
            x_axl = rearrange(x_axl, 'b (d_old h w) c -> b (c h w) d_old', b=B, c=C, h=H, w=W)
            x_axl = self.pool_1d(x_axl) #reduce from old D to new D
            x_axl = rearrange(x_axl, 'b (c h w) d -> b c d h w', b=B, c=C, h=H, w=W)


            x_sgt = self.sagittal_hb(x_sgt) #(2,16,2048)
            x_sgt = x_sgt + self.drop(residual_sgt)
            x_sgt = rearrange(x_sgt, 'b w_old (c d h) -> b (w_old d h) c', b=B, c=C, h=H, d=D)
            x_sgt = x_sgt + self.drop(self.ln_out(self.norm_out(x_sgt)))
            x_sgt = rearrange(x_sgt, 'b (w_old d h) c -> b (c d h) w_old', b=B, c=C, h=H, d=D)
            x_sgt = self.pool_1d(x_sgt) #reduce from old W to new W
            x_sgt = rearrange(x_sgt, 'b (c d h) w -> b c d h w', b=B, c=C, h=H, d=D)

            x_crn = self.coronal_hb(x_crn) #(2,16,2048)
            x_crn = x_crn + self.drop(residual_crn)
            x_crn = rearrange(x_crn, 'b h_old (c d w) -> b (h_old d w) c', b=B, c=C, d=D, w=W)
            x_crn = x_crn + self.drop(self.ln_out(self.norm_out(x_crn)))
            x_crn = rearrange(x_crn, 'b (h_old d w) c -> b (c d w) h_old', b=B, c=C, d=D, w=W)
            x_crn = self.pool_1d(x_crn) #reduce from old H to new H
            x_crn = rearrange(x_crn, 'b (c d w) h -> b c d h w', b=B, c=C, d=D, w=W)

        return x_axl, x_sgt, x_crn
    
    

class IntraviewFullStartBlock(nn.Module):
    def __init__(self, stride_2d = 2):
        super(IntraviewFullStartBlock, self).__init__()
        
        self.axial_hb = SimHymba(embed_dim_att=cf.embed_dim_att_start, num_attention_heads=cf.num_attention_heads_start, \
                  sequence_len=cf.start_res[0], mamba_states=cf.mamba_states_start, hidden_size=cf.v1_dim_start//(stride_2d**2), value_dim=cf.value_dim_att_start)
        self.sagittal_hb = SimHymba(embed_dim_att=cf.embed_dim_att_start, num_attention_heads=cf.num_attention_heads_start, \
                  sequence_len=cf.start_res[2], mamba_states=cf.mamba_states_start, hidden_size=cf.v2_dim_start//(stride_2d**2), value_dim=cf.value_dim_att_start)
        self.coronal_hb = SimHymba(embed_dim_att=cf.embed_dim_att_start, num_attention_heads=cf.num_attention_heads_start, \
                 sequence_len=cf.start_res[1], mamba_states=cf.mamba_states_start, hidden_size=cf.v3_dim_start//(stride_2d**2), value_dim=cf.value_dim_att_start)
        
        dims = cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_start
        self.block_2d = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size = 3, padding=1, stride=stride_2d),
            nn.GroupNorm(1, num_channels=dims),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)

        )
        self.pool_1d = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool2d = nn.ConvTranspose2d(
            in_channels=dims,       # Input channels
            out_channels=dims,      # Output channels (same as input)
            kernel_size=3,       # Kernel size
            stride=stride_2d,            # Stride to upsample by a factor of 2
            padding=1,            # Padding to control output size
            output_padding=1
        )


        dim = dims
        self.drop = nn.Dropout(0.4)
        self.norm_out = nn.LayerNorm(dim)
        self.ln_out = nn.Sequential(
            nn.Linear(dim, dim//16),
            nn.Linear(dim//16, dim) 
        )
        self.dim = dim
        
        
    def forward(self, x, is_final=False):
        #x: B, C, D, H, W
        B, C, D, H, W = x.shape
        B1, C1, D1, H1, W1 = x.shape

        x_axl = x.permute(0,2,1,3,4).contiguous().view(-1, C, H, W) 
        x_sgt = x.permute(0,4,1,2,3).contiguous().view(-1, C, D, H) 
        x_crn = x.permute(0,3,1,2,4).contiguous().view(-1, C, D, W) 

        x_axl = self.block_2d(x_axl)
        x_sgt = self.block_2d(x_sgt)
        x_crn = self.block_2d(x_crn)


        _, _, D, W = x_crn.shape
        _, _, H, _ = x_axl.shape

        x_axl = x_axl.contiguous().view(B, -1, C, H, W).view(B, -1, C*H*W) #(B,D_old,CHW) = (B,L,D)
        x_sgt = x_sgt.contiguous().view(B, -1, C, D, H).view(B, -1, C*D*H) #(B, W_old, CDH)
        x_crn = x_crn.contiguous().view(B, -1, C, D, W).view(B, -1, C*D*W) #(B, H_old, CDW)

        residual_axl = x_axl
        residual_sgt = x_sgt
        residual_crn = x_crn

        x_axl = self.axial_hb(x_axl) #(2,4,8192)
        x_axl = x_axl + self.drop(residual_axl)
        x_axl = rearrange(x_axl, 'b d_old (c h w) -> b (d_old h w) c', b=B, c=C, h=H, w=W)
        x_axl = x_axl + self.drop(self.ln_out(self.norm_out(x_axl)))
        if not is_final:
            x_axl = rearrange(x_axl, 'b (d_old h w) c -> (b d_old) c h w', b=B, c=C, h=H, w=W)
            x_axl = self.pool2d(x_axl)
            x_axl = rearrange(x_axl, '(b d) c h w -> b c d h w', b=B1, c=C1, h=H1, w=W1)
        else:
            x_axl = rearrange(x_axl, 'b (d_old h w) c -> b (c h w) d_old', b=B, c=C, h=H, w=W)
            x_axl = self.pool_1d(x_axl)
            x_axl = rearrange(x_axl, 'b (c h w) d -> b (d h w) c', b=B, c=C, h=H, w=W)

        x_sgt = self.sagittal_hb(x_sgt) #(2,16,2048)
        x_sgt = x_sgt + self.drop(residual_sgt)
        x_sgt = rearrange(x_sgt, 'b w_old (c d h) -> b (w_old d h) c', b=B, c=C, h=H, d=D)
        x_sgt = x_sgt + self.drop(self.ln_out(self.norm_out(x_sgt)))
        if not is_final:
            x_sgt = rearrange(x_sgt, 'b (w_old d h) c -> (b w_old) c d h', b=B, c=C, h=H, d=D)
            x_sgt = self.pool2d(x_sgt)
            x_sgt = rearrange(x_sgt, '(b w) c d h -> b c d h w', b=B1, c=C1, h=H1, d=D1)
        else:
            x_sgt = rearrange(x_sgt, 'b (w_old d h) c -> b (c d h) w_old', b=B, c=C, h=H, d=D)
            x_sgt = self.pool_1d(x_sgt)
            x_sgt = rearrange(x_sgt, 'b (c d h) w -> b (d h w) c', b=B, c=C, h=H, d=D)

        x_crn = self.coronal_hb(x_crn) #(2,16,2048)
        x_crn = x_crn + self.drop(residual_crn)
        x_crn = rearrange(x_crn, 'b h_old (c d w) -> b (h_old d w) c', b=B, c=C, d=D, w=W)
        x_crn = x_crn + self.drop(self.ln_out(self.norm_out(x_crn)))
        if not is_final:
            x_crn = rearrange(x_crn, 'b (h_old d w) c -> (b h_old) c d w', b=B, c=C, d=D, w=W)
            x_crn = self.pool2d(x_crn)
            x_crn = rearrange(x_crn, '(b h) c d w -> b c d h w', b=B1, c=C1, d=D1, w=W1)
        else:
            x_crn = rearrange(x_crn, 'b (h_old d w) c -> b (c d w) h_old', b=B, c=C, d=D, w=W)
            x_crn = self.pool_1d(x_crn)
            x_crn = rearrange(x_crn, 'b (c d w) h -> b (d h w) c',b=B, c=C, d=D, w=W)

        return x_axl, x_sgt, x_crn
    


class IntraviewFullEndBlock(nn.Module):
    def __init__(self, stride_2d = 2, is_start_layer=False):
        super(IntraviewFullEndBlock, self).__init__()
        '''
        each block has multiple layers, the first layer is for cross-att, and the remainings are for self-att
        Therefore, is_start_layer marks if the layer is the first layer of the block -> cross-att
        '''
        
        self.axial_hb = SimHymba(embed_dim_att=cf.embed_dim_att_end, num_attention_heads=cf.num_attention_heads_end, \
                  sequence_len=cf.end_res[0], mamba_states=cf.mamba_states_end, hidden_size=cf.v1_dim_end//(stride_2d**2), is_start_layer=is_start_layer, value_dim=cf.value_dim_att_end)
        self.sagittal_hb = SimHymba(embed_dim_att=cf.embed_dim_att_end, num_attention_heads=cf.num_attention_heads_end, \
                  sequence_len=cf.end_res[2], mamba_states=cf.mamba_states_end, hidden_size=cf.v2_dim_end//(stride_2d**2), is_start_layer=is_start_layer, value_dim=cf.value_dim_att_end)
        self.coronal_hb = SimHymba(embed_dim_att=cf.embed_dim_att_end, num_attention_heads=cf.num_attention_heads_end, \
                 sequence_len=cf.end_res[1], mamba_states=cf.mamba_states_end, hidden_size=cf.v3_dim_end//(stride_2d**2), is_start_layer=is_start_layer, value_dim=cf.value_dim_att_end)
        
        dims = cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_end #f_maps_dims[0] is correct
        self.block_2d = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size = 3, padding=1, stride=stride_2d),
            nn.GroupNorm(1, num_channels=dims),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)

        )
        self.pool_1d = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool2d = nn.ConvTranspose2d(
            in_channels=dims,       # Input channels
            out_channels=dims,      # Output channels (same as input)
            kernel_size=3,       # Kernel size
            stride=stride_2d,            # Stride to upsample by a factor of 2
            padding=1,            # Padding to control output size
            output_padding=1
        )
        
        dim = dims
        self.drop = nn.Dropout(0.4)
        self.norm_out = nn.LayerNorm(dim)
        self.ln_out = nn.Sequential(
            nn.Linear(dim, dim//16),
            nn.Linear(dim//16, dim) 
        )
        self.dim = dim

    def forward(self, x, x_skip=None, is_final=False):
        #x: B, C, D, H, W
        B, C, D, H, W = x.shape
        B1, C1, D1, H1, W1 = x.shape
        if x_skip==None:

            x_axl = x.permute(0,2,1,3,4).contiguous().view(-1, C, H, W) 
            x_sgt = x.permute(0,4,1,2,3).contiguous().view(-1, C, D, H) 
            x_crn = x.permute(0,3,1,2,4).contiguous().view(-1, C, D, W)

            x_axl = self.block_2d(x_axl)
            x_sgt = self.block_2d(x_sgt)
            x_crn = self.block_2d(x_crn)

            _, _, D, W = x_crn.shape
            _, _, H, _ = x_axl.shape

            x_axl = x_axl.contiguous().view(B, -1, C, H, W).view(B, -1, C*H*W) #(B,D_old,CHW) = (B,L,D)
            x_sgt = x_sgt.contiguous().view(B, -1, C, D, H).view(B, -1, C*D*H) #(B, W_old, CDH)
            x_crn = x_crn.contiguous().view(B, -1, C, D, W).view(B, -1, C*D*W) #(B, H_old, CDW)

            residual_axl = x_axl
            residual_sgt = x_sgt
            residual_crn = x_crn

            x_axl = self.axial_hb(x_axl) #(2,4,8192)
            x_axl = x_axl + self.drop(residual_axl)
            x_axl = rearrange(x_axl, 'b d_old (c h w) -> b (d_old h w) c', b=B, c=C, h=H, w=W)
            x_axl = x_axl + self.drop(self.ln_out(self.norm_out(x_axl)))
            if not is_final:
                x_axl = rearrange(x_axl, 'b (d_old h w) c -> (b d_old) c h w', b=B, c=C, h=H, w=W)
                x_axl = self.pool2d(x_axl)
                x_axl = rearrange(x_axl, '(b d) c h w -> b c d h w', b=B1, c=C1, h=H1, w=W1)
            else:
                x_axl = rearrange(x_axl, 'b (d_old h w) c -> b (c h w) d_old', b=B, c=C, h=H, w=W)
                x_axl = self.pool_1d(x_axl)
                x_axl = rearrange(x_axl, 'b (c h w) d -> b (d h w) c', b=B, c=C, h=H, w=W)

            x_sgt = self.sagittal_hb(x_sgt) #(2,16,2048)
            x_sgt = x_sgt + self.drop(residual_sgt)
            x_sgt = rearrange(x_sgt, 'b w_old (c d h) -> b (w_old d h) c', b=B, c=C, h=H, d=D)
            x_sgt = x_sgt + self.drop(self.ln_out(self.norm_out(x_sgt)))
            if not is_final:
                x_sgt = rearrange(x_sgt, 'b (w_old d h) c -> (b w_old) c d h', b=B, c=C, h=H, d=D)
                x_sgt = self.pool2d(x_sgt)
                x_sgt = rearrange(x_sgt, '(b w) c d h -> b c d h w', b=B1, c=C1, h=H1, d=D1)
            else:
                x_sgt = rearrange(x_sgt, 'b (w_old d h) c -> b (c d h) w_old', b=B, c=C, h=H, d=D)
                x_sgt = self.pool_1d(x_sgt)
                x_sgt = rearrange(x_sgt, 'b (c d h) w -> b (d h w) c', b=B, c=C, h=H, d=D)


            x_crn = self.coronal_hb(x_crn) #(2,16,2048)
            x_crn = x_crn + self.drop(residual_crn)
            x_crn = rearrange(x_crn, 'b h_old (c d w) -> b (h_old d w) c', b=B, c=C, d=D, w=W)
            x_crn = x_crn + self.drop(self.ln_out(self.norm_out(x_crn)))
            if not is_final:
                x_crn = rearrange(x_crn, 'b (h_old d w) c -> (b h_old) c d w', b=B, c=C, d=D, w=W)
                x_crn = self.pool2d(x_crn)
                x_crn = rearrange(x_crn, '(b h) c d w -> b c d h w', b=B1, c=C1, d=D1, w=W1)
            else:
                x_crn = rearrange(x_crn, 'b (h_old d w) c -> b (c d w) h_old', b=B, c=C, d=D, w=W)
                x_crn = self.pool_1d(x_crn)
                x_crn = rearrange(x_crn, 'b (c d w) h -> b (d h w) c', b=B, c=C, d=D, w=W)


        else:

            x_axl = x.permute(0,2,1,3,4).contiguous().view(-1, C, H, W) 
            x_sgt = x.permute(0,4,1,2,3).contiguous().view(-1, C, D, H) 
            x_crn = x.permute(0,3,1,2,4).contiguous().view(-1, C, D, W)

            x_axl_skip = x_skip.permute(0,2,1,3,4).contiguous().view(-1, C, H, W) 
            x_sgt_skip = x_skip.permute(0,4,1,2,3).contiguous().view(-1, C, D, H) 
            x_crn_skip = x_skip.permute(0,3,1,2,4).contiguous().view(-1, C, D, W)

            x_axl = self.block_2d(x_axl)
            x_sgt = self.block_2d(x_sgt)
            x_crn = self.block_2d(x_crn)

            x_axl_skip = self.block_2d(x_axl_skip)
            x_sgt_skip = self.block_2d(x_sgt_skip)
            x_crn_skip = self.block_2d(x_crn_skip)

            _, _, D, W = x_crn.shape
            _, _, H, _ = x_axl.shape

            x_axl = x_axl.contiguous().view(B, -1, C, H, W).view(B, -1, C*H*W) #(B,D_old,CHW) = (B,L,D)
            x_sgt = x_sgt.contiguous().view(B, -1, C, D, H).view(B, -1, C*D*H) #(B, W_old, CDH)
            x_crn = x_crn.contiguous().view(B, -1, C, D, W).view(B, -1, C*D*W) #(B, H_old, CDW)

            x_axl_skip = x_axl_skip.contiguous().view(B, -1, C, H, W).view(B, -1, C*H*W) #(B,D_old,CHW) = (B,L,D)
            x_sgt_skip = x_sgt_skip.contiguous().view(B, -1, C, D, H).view(B, -1, C*D*H) #(B, W_old, CDH)
            x_crn_skip = x_crn_skip.contiguous().view(B, -1, C, D, W).view(B, -1, C*D*W) #(B, H_old, CDW)
            
            residual_axl = x_axl+x_axl_skip
            residual_sgt = x_sgt+x_sgt_skip
            residual_crn = x_crn+x_crn_skip

            x_axl = self.axial_hb(x_axl, x_axl_skip) #(2,4,8192)
            x_axl = x_axl + self.drop(residual_axl)
            x_axl = rearrange(x_axl, 'b d_old (c h w) -> b (d_old h w) c', b=B, c=C, h=H, w=W)
            x_axl = x_axl + self.drop(self.ln_out(self.norm_out(x_axl)))
            if not is_final:
                x_axl = rearrange(x_axl, 'b (d_old h w) c -> (b d_old) c h w', b=B, c=C, h=H, w=W)
                x_axl = self.pool2d(x_axl)
                x_axl = rearrange(x_axl, '(b d) c h w -> b c d h w', b=B1, c=C1, h=H1, w=W1)
            else:
                x_axl = rearrange(x_axl, 'b (d_old h w) c -> b (c h w) d_old', b=B, c=C, h=H, w=W)
                x_axl = self.pool_1d(x_axl)
                x_axl = rearrange(x_axl, 'b (c h w) d -> b (d h w) c', b=B, c=C, h=H, w=W)

            x_sgt = self.sagittal_hb(x_sgt, x_sgt_skip) #(2,16,2048)
            x_sgt = x_sgt + self.drop(residual_sgt)
            x_sgt = rearrange(x_sgt, 'b w_old (c d h) -> b (w_old d h) c', b=B, c=C, h=H, d=D)
            x_sgt = x_sgt + self.drop(self.ln_out(self.norm_out(x_sgt)))
            if not is_final:
                x_sgt = rearrange(x_sgt, 'b (w_old d h) c -> (b w_old) c d h', b=B, c=C, h=H, d=D)
                x_sgt = self.pool2d(x_sgt)
                x_sgt = rearrange(x_sgt, '(b w) c d h -> b c d h w', b=B1, c=C1, h=H1, d=D1)
            else:
                x_sgt = rearrange(x_sgt, 'b (w_old d h) c -> b (c d h) w_old', b=B, c=C, h=H, d=D)
                x_sgt = self.pool_1d(x_sgt)
                x_sgt = rearrange(x_sgt, 'b (c d h) w -> b (d h w) c', b=B, c=C, h=H, d=D)

            x_crn = self.coronal_hb(x_crn, x_crn_skip) #(2,16,2048)
            x_crn = x_crn + self.drop(residual_crn)
            x_crn = rearrange(x_crn, 'b h_old (c d w) -> b (h_old d w) c', b=B, c=C, d=D, w=W)
            x_crn = x_crn + self.drop(self.ln_out(self.norm_out(x_crn)))
            if not is_final:
                x_crn = rearrange(x_crn, 'b (h_old d w) c -> (b h_old) c d w', b=B, c=C, d=D, w=W)
                x_crn = self.pool2d(x_crn)
                x_crn = rearrange(x_crn, '(b h) c d w -> b c d h w', b=B1, c=C1, d=D1, w=W1)
            else:
                x_crn = rearrange(x_crn, 'b (h_old d w) c -> b (c d w) h_old', b=B, c=C, d=D, w=W)
                x_crn = self.pool_1d(x_crn)
                x_crn = rearrange(x_crn, 'b (c d w) h -> b (d h w) c', b=B, c=C, d=D, w=W)


        return x_axl, x_sgt, x_crn
    

class InterviewBlock(nn.Module):
    def __init__(self, factor = 16):
        super(InterviewBlock, self).__init__()

        self.ln1 = nn.Sequential(
            nn.Linear(cf.H*cf.W*cf.D, cf.H*cf.W*cf.D//factor),
            nn.Linear(cf.H*cf.W*cf.D//factor, cf.H*cf.W*cf.D)
        )
        self.ln2 = nn.Sequential(
            nn.Linear(cf.H*cf.W*cf.D, cf.H*cf.W*cf.D//factor),
            nn.Linear(cf.H*cf.W*cf.D//factor, cf.H*cf.W*cf.D)
        )

        
    def forward(self, x_axl, x_sgt, x_crn):
        f_sc = F.softmax(self.ln1(torch.matmul(x_sgt, x_axl.transpose(-1, -2))), dim=-1)
        f_cs = F.softmax(self.ln2(torch.matmul(x_crn, x_axl.transpose(-1, -2))), dim=-1)

        f_a_sc = torch.matmul(f_sc, x_sgt)
        f_a_cs = torch.matmul(f_cs, x_crn)

        return x_axl + f_a_sc + f_a_cs + x_sgt + x_crn

class InterviewIntermediateBlock(nn.Module):
    def __init__(self, in_dims, factor = 16):
        super(InterviewIntermediateBlock, self).__init__()

        self.ln1 = nn.Sequential(
            nn.Linear(in_dims, int(in_dims//factor)),
            nn.Linear(int(in_dims//factor), in_dims)
        )
        self.ln2 = nn.Sequential(
            nn.Linear(in_dims, int(in_dims//factor)),
            nn.Linear(int(in_dims//factor), in_dims)
        )
        self.indims = in_dims
        self.factor = factor
        
    def forward(self, x_axl, x_sgt, x_crn):

        f_sc = F.softmax(self.ln1(torch.matmul(x_sgt, x_axl.transpose(-1, -2))), dim=-1)
        f_cs = F.softmax(self.ln2(torch.matmul(x_crn, x_axl.transpose(-1, -2))), dim=-1)

        f_a_sc = torch.matmul(f_sc, x_sgt)
        f_a_cs = torch.matmul(f_cs, x_crn)

        return x_axl + f_a_sc + f_a_cs + x_sgt + x_crn


class HymaSwinBlock(nn.Module): #no inter-view as Hymavi
    def __init__(self, ith_layer, num_layers= 2, reduce_rate = 4):
        super(HymaSwinBlock, self).__init__()
        
        self.down_dims = nn.Conv3d(cf.swin_f_map_dims[ith_layer], cf.swin_f_map_dims[ith_layer]// reduce_rate,kernel_size=1)
        self.up_dims = nn.Conv3d(cf.swin_f_map_dims[ith_layer]//reduce_rate, cf.swin_f_map_dims[ith_layer], kernel_size=1)

        self.intra = nn.ModuleList([
            # nn.Sequential(
            #     *[
                    IntraviewSwinBlock(ith_layer=ith_layer, layer_ord_in_block=j, reduce_rate=reduce_rate)
                    for j in range(num_layers)
            #     ]
            # )
        ])

        # self.intra = IntraviewSwinBlock(ith_layer=ith_layer, layer_ord_in_block=layer_ord_in_block)
        # self.inter = InterviewIntermediateBlock(in_dims=cf.window_sizes[ith_layer][0]*cf.window_sizes[ith_layer][1] * cf.window_sizes[ith_layer][2], factor=16)
        self.ith = ith_layer
        self.rdr = reduce_rate
        self.num_layers = num_layers
        
    def forward(self, x):
        #x: B, C, D, H, W
        B, C, D, H, W = x.shape
        residual = x

        x = self.down_dims(x)
        for i, layer in enumerate(self.intra):
            x = layer(x)

        x = self.up_dims(x)

        x = x + residual

        return x

class HymaSwinDecBlock(nn.Module): #no inter-view as Hymavi
    def __init__(self, ith_layer, num_layers= 2, reduce_rate = 4):
        super(HymaSwinDecBlock, self).__init__()
        
        self.down_dims = nn.Conv3d(cf.swin_f_map_dims[ith_layer], cf.swin_f_map_dims[ith_layer]// reduce_rate,kernel_size=1)
        self.up_dims = nn.Conv3d(cf.swin_f_map_dims[ith_layer]//reduce_rate, cf.swin_f_map_dims[ith_layer], kernel_size=1)

        self.intra = nn.ModuleList([
            # nn.Sequential(
            #     *[
                    IntraviewSwinDecBlock(ith_layer=ith_layer, layer_ord_in_block=j, reduce_rate=reduce_rate) if j==0 else IntraviewSwinBlock(ith_layer=ith_layer, layer_ord_in_block=j, reduce_rate=reduce_rate)
                    #IntraviewSwinBlock(ith_layer=ith_layer, layer_ord_in_block=j, reduce_rate=reduce_rate)
                    for j in range(num_layers)
            #     ]
            # )
        ])

        # self.intra = IntraviewSwinBlock(ith_layer=ith_layer, layer_ord_in_block=layer_ord_in_block)
        # self.inter = InterviewIntermediateBlock(in_dims=cf.window_sizes[ith_layer][0]*cf.window_sizes[ith_layer][1] * cf.window_sizes[ith_layer][2], factor=16)
        self.ith = ith_layer
        self.rdr = reduce_rate
        self.num_layers = num_layers

        self.drop = nn.Dropout(0.4)
        dim = cf.swin_f_map_dims[ith_layer]
        self.dim = dim
        # self.norm_out = HymbaRMSNorm(dim)
        self.norm_out = nn.LayerNorm([dim] + cf.swin_input_dec_resolutions[ith_layer]) 
        self.ln_out = nn.Sequential(
            nn.Linear(dim, dim//16),
            nn.Linear(dim//16, dim) 
        )
        
        
    def forward(self, x, x_skip):
        #x: B, C, D, H, W
        B, C, D, H, W = x.shape
        residual = x + x_skip

        x = self.down_dims(x)
        x_skip = self.down_dims(x_skip)

        for i, layer in enumerate(self.intra):
            if i == 0:
                x = layer(x, x_skip)
            else:
                x = layer(x)

        x = self.up_dims(x)
        x = x + residual
        return x
    
class HymaFullStartBlock(nn.Module): #no inter-view as Hymavi
    def __init__(self, reduce_rate = 4, stride_2d = 2, num_layers=2):
        super(HymaFullStartBlock, self).__init__()
        
        self.down_dims = nn.Conv3d(cf.f_maps_dim[0], cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_start ,kernel_size=1)
        self.up_dims = nn.Conv3d(cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_start, cf.f_maps_dim[0], kernel_size=1)

        # self.intra = IntraviewFullStartBlock(stride_2d=stride_2d)
        self.intra = nn.ModuleList([
                    IntraviewFullStartBlock(stride_2d=stride_2d)
                    for j in range(num_layers)
                  
        ])

        # self.up_spatial_insideIntra = nn.ModuleList([
        #     nn.ConvTranspose3d(in_channels=cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_start, \
        #                                      out_channels=cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_start, \
        #                                         kernel_size=2, stride=2)
        #     for _ in range(num_layers-1)
        # ])
        # self.inter = InterviewBlock()
        # self.ith = ith_layer
        self.rdr = reduce_rate
        self.inter = InterviewIntermediateBlock(in_dims=cf.start_res[0]*cf.start_res[1]*cf.start_res[2]//(stride_2d**3), \
                                                factor=512*8*2)
        self.stride_2d = stride_2d
        self.up_spatial = nn.ConvTranspose3d(in_channels=cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_start, \
                                             out_channels=cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_start, \
                                                kernel_size=2, stride=2)

        self.num_layers = num_layers
    def forward(self, x):
        #x: B, C, D, H, W
        B, C, D, H, W = x.shape
        residual = x
        x = self.down_dims(x)
        for i, layer in enumerate(self.intra):

            is_final = False
            if i==self.num_layers-1:
                is_final=True
            x_axl, x_sgt, x_crn = layer(x, is_final =is_final)
            if not is_final:
                x = (x_axl + x_sgt + x_crn)/3

        x = self.inter(x_axl, x_sgt, x_crn)

        D, H, W = D//self.stride_2d, H//self.stride_2d, W//self.stride_2d

        x = x.contiguous().view(B, D, H, W, -1)
        x = x.permute(0, 4, 1, 2, 3) #(B, C, D, H, W)
        x = self.up_spatial(x)
        x = self.up_dims(x)

        x =x+ residual

        return x

class HymaFullEndBlock(nn.Module): #no inter-view as Hymavi
    def __init__(self, reduce_rate = 4, stride_2d=2, num_layers = 2):
        super(HymaFullEndBlock, self).__init__()
        
        self.down_dims = nn.Conv3d(cf.f_maps_dim[0], cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_end ,kernel_size=1)
        self.up_dims = nn.Conv3d(cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_end, cf.f_maps_dim[0], kernel_size=1)

        # self.intra = IntraviewFullEndBlock(stride_2d=stride_2d)
        self.intra = nn.ModuleList([
                    IntraviewFullEndBlock(stride_2d=stride_2d, is_start_layer=True) if j==0 else IntraviewFullEndBlock(stride_2d=stride_2d, is_start_layer=False) 
                    for j in range(num_layers)
                  
        ])

        # self.up_spatial_insideIntra = nn.ModuleList([
        #     nn.ConvTranspose3d(in_channels=cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_start, \
        #                                      out_channels=cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_start, \
        #                                         kernel_size=2, stride=2)
        #     for _ in range(num_layers-1)
        # ])

        self.inter = InterviewIntermediateBlock(in_dims=cf.end_res[0]*cf.end_res[1]*cf.end_res[2]//(stride_2d**3), \
                                                factor=512*8*2)
        self.stride_2d = stride_2d
        self.up_spatial = nn.ConvTranspose3d(in_channels=cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_end, \
                                             out_channels=cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_end, \
                                                kernel_size=2, stride=2)
        self.rdr = reduce_rate
        self.num_layers = num_layers
        
    def forward(self, x, x_skip):
        #x: B, C, D, H, W
        B, C, D, H, W = x.shape
        residual = x + x_skip
        x = self.down_dims(x) #(B,32,4,16,16)
        x_skip = self.down_dims(x_skip)

        for i, layer in enumerate(self.intra):
            is_final = False
            if i==self.num_layers-1:
                is_final=True

            if i==0:
                x_axl, x_sgt, x_crn = layer(x, x_skip, is_final =is_final) 
            else:
                x_axl, x_sgt, x_crn = layer(x, is_final = is_final)

            if not is_final:
                x = (x_axl + x_sgt + x_crn)/3

        x = self.inter(x_axl, x_sgt, x_crn)
        D, H, W = D//self.stride_2d, H//self.stride_2d, W//self.stride_2d

        x = x.contiguous().view(B, D, H, W, -1)
        x = x.permute(0, 4, 1, 2, 3) #(B, C, D, H, W)
        x = self.up_spatial(x)

        x = self.up_dims(x)

        x =x+ residual

        return x



class FirstMamBlock(nn.Module): #no inter-view as Hymavi
    def __init__(self, reduce_rate = 4, stride_2d = 8, num_layers=2):
        super(FirstMamBlock, self).__init__()

        # self.down_dims = nn.Conv3d(cf.f_maps_dim[0], cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_start ,kernel_size=1)
        # self.up_dims = nn.Conv3d(cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_start, cf.f_maps_dim[0], kernel_size=1)
        
        self.intra = nn.ModuleList([
                    IntraviewFirstMam(stride_2d=stride_2d)
                    for j in range(num_layers)
                  
        ])

        padd = (cf.first_mam_res[0]%stride_2d, cf.first_mam_res[1]%stride_2d, cf.first_mam_res[2]%stride_2d)
        self.up_spatial_insideIntra = nn.ModuleList([
            nn.ConvTranspose3d(in_channels=cf.dim_after_stem1, \
                                             out_channels=cf.dim_after_stem1, \
                                                kernel_size=stride_2d, stride=stride_2d, padding=0, output_padding=padd)
            for _ in range(num_layers)
        ])
        # self.inter = InterviewBlock()
        # self.ith = ith_layer
        self.rdr = reduce_rate
        self.stride_2d = stride_2d

        self.num_layers = num_layers
    def forward(self, x):
        #x: B, C, D, H, W
        B, C, D, H, W = x.shape
        residual = x

        for i, layer in enumerate(self.intra):
            x_axl, x_sgt, x_crn = layer(x)
            x = (x_axl + x_sgt + x_crn)/3
            x = self.up_spatial_insideIntra[i](x)

        x =x+ residual

        return x

class EndMamBlock(nn.Module): #no inter-view as Hymavi
    def __init__(self, reduce_rate = 4, stride_2d = 8, num_layers=2):
        super(EndMamBlock, self).__init__()

        # self.down_dims = nn.Conv3d(cf.f_maps_dim[0], cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_start ,kernel_size=1)
        # self.up_dims = nn.Conv3d(cf.f_maps_dim[0]//cf.reduce_rate_before_hymavi_start, cf.f_maps_dim[0], kernel_size=1)
        
        self.intra = nn.ModuleList([
                    IntraviewEndMam(stride_2d=stride_2d)
                    for j in range(num_layers)
                  
        ])

        padd = (cf.first_mam_res[0]%stride_2d, cf.first_mam_res[1]%stride_2d, cf.first_mam_res[2]%stride_2d)
        self.up_spatial_insideIntra = nn.ModuleList([
            nn.ConvTranspose3d(in_channels=cf.dim_after_stem1, \
                                             out_channels=cf.dim_after_stem1, \
                                                kernel_size=stride_2d, stride=stride_2d, padding=0, output_padding=padd)
            for _ in range(num_layers)
        ])
        # self.inter = InterviewBlock()
        # self.ith = ith_layer
        self.rdr = reduce_rate
        self.stride_2d = stride_2d

        self.num_layers = num_layers
    def forward(self, x, x_skip):
        #x: B, C, D, H, W
        B, C, D, H, W = x.shape
        x = x+x_skip
        residual = x

        for i, layer in enumerate(self.intra):
            x_axl, x_sgt, x_crn = layer(x)
            x = (x_axl + x_sgt + x_crn)/3

            x = self.up_spatial_insideIntra[i](x)

        x =x+ residual

        return x


class HymaviBlock(nn.Module):
    def __init__(self, num_layers = 2, stride_2d = 1):
        super(HymaviBlock, self).__init__()
        
        self.down_dims = nn.Conv3d(cf.C, cf.C//cf.reduce_rate_before_hymavi ,kernel_size=1)
        self.up_dims = nn.Conv3d(cf.C//cf.reduce_rate_before_hymavi, cf.C, kernel_size=1)

        # self.intra = IntraviewBlock()
        self.intra = nn.ModuleList([
                    IntraviewBlock(stride_2d=stride_2d)
                    for j in range(num_layers)
                  
        ])
        self.inter = InterviewBlock()
        
    def forward(self, x):
        #x: B, C, D, H, W
        B, C, D, H, W = x.shape
        residual = x
        x = self.down_dims(x)

        for i, layer in enumerate(self.intra):
            x_axl, x_sgt, x_crn = layer(x)
            x = (x_axl + x_sgt + x_crn)/3

        x_axl = x_axl.permute(0,2,3,4,1).contiguous().view(x.size(0), -1, x.size(1))
        x_sgt = x_sgt.permute(0,2,3,4,1).contiguous().view(x.size(0), -1, x.size(1))
        x_crn = x_crn.permute(0,2,3,4,1).contiguous().view(x.size(0), -1, x.size(1))

        x = self.inter(x_axl, x_sgt, x_crn)


        x = x.view(B, D, H, W, -1).permute(0, 4, 1, 2, 3)

        x = self.up_dims(x)

        x =x+ residual

        return x

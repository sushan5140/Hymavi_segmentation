import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import random
from config import config as cf

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, prediction, ground_truth):
        """
        Computes the Dice Loss.
        
        Args:
            prediction (torch.Tensor): Predicted tensor of shape (B, C, D, H, W).
            ground_truth (torch.Tensor): Ground truth tensor of shape (B, D, H, W).

        Returns:
            torch.Tensor: Dice loss for each class, averaged across the batch.
        """
        # Convert prediction to probability using softmax
        prediction = F.softmax(prediction, dim=1)  # Shape: (B, C, D, H, W)
        if len(ground_truth.shape)==5:
            ground_truth = ground_truth.squeeze(1)

        # Convert ground truth to one-hot encoding
        B, C, D, H, W = prediction.shape
        ground_truth_one_hot = F.one_hot(ground_truth, num_classes=C)  # Shape: (B, D, H, W, C)

        try:
            ground_truth_one_hot = ground_truth_one_hot.permute(0, 4, 1, 2, 3).float()  # Shape: (B, C, D, H, W)
        except:
            print(">>>>>>>>><<<<<<<<< ", ground_truth.shape)

        # Compute Dice score
        intersection = torch.sum(prediction * ground_truth_one_hot, dim=(2, 3, 4))
        union = torch.sum(prediction, dim=(2, 3, 4)) + torch.sum(ground_truth_one_hot, dim=(2, 3, 4))

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score  # Dice Loss is 1 - Dice Coefficient

        return dice_loss.mean()  # Mean Dice loss over batch


# class CustomLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.accumulated_rep = torch.zeros(cf.num_classes, cf.f_maps_dim[0]).to(device)
#         self.num_accumulated = torch.zeros(cf.num_classes)

#     def get_f_representative(self, x):
#         #n_pixels, dims
#         return x.mean(dim=0)

#     def get_segmentation_boundary_3d(self, segmentation_mask):
#         """
#         Extracts the boundary of a 3D segmentation mask.

#         Parameters:
#         - segmentation_mask (torch.Tensor): A 4D tensor of shape (B, C, D, H, W)
        
#         Returns:
#         - boundary_mask (torch.Tensor): Tensor of the same shape as `segmentation_mask` containing boundaries (1s) and 0 elsewhere.
#         """
#         # Ensure input is 5D for batch operations
#         assert segmentation_mask.ndim == 5, "Expected input shape (B, C, D, H, W)"

#         # Create kernel with the correct channel size
#         B, C, D, H, W = segmentation_mask.shape
#         kernel = torch.ones((C, 1, 3, 3, 3), device=segmentation_mask.device)

#         # Pad the input to ensure boundary is preserved
#         padded_mask = F.pad(segmentation_mask.float(), (1, 1, 1, 1, 1, 1), mode='constant', value=0).clone()

#         # Perform erosion by convolving the segmentation mask per channel
#         eroded_mask = F.conv3d(padded_mask, kernel, stride=1, groups=C) == kernel[0].sum()

#         # Extract boundary by XOR between the mask and eroded version
#         boundary_mask = segmentation_mask & (~eroded_mask)


#         return boundary_mask.squeeze(1)


#     def set_boundary_zero(self, tensor, offset):
#         """
#         Set the boundary of a 4D tensor (B, D, H, W) to zero with an offset.
#         Args:
#             tensor: PyTorch tensor of shape (B, D, H, W)
#             offset: Number of pixels/voxels from the edge to set as zero
#         """
#         B, D, H, W = tensor.shape

#         # Set depth boundaries
#         tensor[:, :offset, :, :] = 0
#         tensor[:, -offset:, :, :] = 0

#         # Set height boundaries
#         tensor[:, :, :offset, :] = 0
#         tensor[:, :, -offset:, :] = 0

#         # Set width boundaries
#         tensor[:, :, :, :offset] = 0
#         tensor[:, :, :, -offset:] = 0
#         return tensor


#     def forward(self, pred, gt, feat, num_classes):
#         B, C, D, H, W = feat.shape
#         pred = pred.permute(0, 2, 3, 4, 1)  # B, D, H, W, C
#         pred = F.softmax(pred, dim=-1)  # Softmax along the class dimension

#         feat = feat.permute(0, 2, 3, 4, 1)
#         pred_hard = torch.argmax(pred, dim=-1)  # B, D, H, W (for class)

#         rep = torch.zeros((num_classes, C), device=pred.device)

#         class_aware_loss = []
#         class_aware_loss1 = []
#         class_aware_loss2 = []
#         missed_boundary_loss1 = []
#         missed_boundary_loss2 = []
#         rep_loss = 0
#         #print(f"::::::::::::::NaN in feat: {torch.isnan(feat).any(), torch.isnan(pred).any(), torch.isnan(gt).any()}")

#         for i in range(num_classes):
#             mask_pred = (pred_hard == i) & (gt == i)
#             if 1 not in mask_pred:
#                 continue
#             feat_i = feat[mask_pred, :]
#             rep[i] = self.get_f_representative(feat_i)  # Representative feature for class i
            
#             self.num_accumulated[i] = self.num_accumulated[i] + 1
#             with torch.no_grad():
#                 self.accumulated_rep[i] = self.accumulated_rep[i] + (rep[i] - self.accumulated_rep[i]) / self.num_accumulated[i]
#             # self.accumulated_rep[i] = self.accumulated_rep[i] + (rep[i]-self.accumulated_rep[i])/self.num_accumulated[i]


#         # existed_cls = 0
#         # existed_cls1 = 0
#         # existed_cls2 = 0
#         for i in range(num_classes):
#             if (i not in gt) and (i not in pred_hard):
#                 continue
            
#             elif (i not in gt) and (i in pred_hard): #FP
#                 if self.num_accumulated[i]==0:
#                     continue
#                 mask_pred = (pred_hard==i)
#                 feat_i = feat[mask_pred, :]
#                 feat_accu = self.accumulated_rep[i]

#                 norm_feat_i = F.normalize(feat_i, p=2, dim=1)
#                 norm_rep = F.normalize(feat_accu.clone(), p=2, dim=0)
#                 sim = F.cosine_similarity(norm_feat_i, norm_rep)

#                 # class_aware_loss1 = class_aware_loss1 + sim.mean()
#                 class_aware_loss1.append(sim.mean())

#                 # existed_cls1 +=1
#                 # print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")
            
#             elif (i in gt) and (i not in pred_hard): #FN
#                 if self.num_accumulated[i]==0:
#                     continue
#                 mask_pred = (gt==i)
#                 feat_i = feat[mask_pred, :]
#                 feat_accu = self.accumulated_rep[i]

#                 norm_feat_i = F.normalize(feat_i, p=2, dim=1)
#                 norm_rep = F.normalize(feat_accu.clone(), p=2, dim=0)
#                 sim = F.cosine_similarity(norm_feat_i, norm_rep)

#                 # class_aware_loss2 = class_aware_loss2 + sim.mean()
#                 class_aware_loss2.append(sim.mean())

#                 # existed_cls2 +=1
#                 # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            

#             else:

#                 # existed_cls +=1
#                 # Mask where the ground truth is class `i`
#                 mask_pred = pred_hard == i  # Avoid cloning unless necessary
#                 mask_gt = gt == i  # Avoid cloning unless necessary

                
#                 #print("NaN in mask_pred") if torch.isnan(mask_pred).any() else 1
#                 #print("NaN in mask_gt") if torch.isnan(mask_gt).any() else 1

#                 # Get the features corresponding to the ground truth class `i`
#                 # feat_i = feat[mask_pred, :]
#                 feat_i_gt = feat[mask_gt, :]

                
#                 #print("NaN in feat_i_gt") if torch.isnan(feat_i_gt).any() else 1
#                 #print("NaN in feat_i") if torch.isnan(feat_i).any() else 1
#                 #print("$$$$$$$$$$$$$$$$$$$$$$$$ ", feat_i.shape, torch.sum(feat_i), torch.sum(mask_pred), torch.sum(mask_gt))
                

#                 # Compute cosine similarity between gt and representative feature
                
                
#                 #print("NaN in rep[i]") if torch.isnan(rep[i]).any() else 1

#                 norm_feat_i_gt = F.normalize(feat_i_gt, p=2, dim=1)
#                 norm_rep = F.normalize(rep[i].clone(), p=2, dim=0)  # No need to clone here
                
#                 #print("NaN in feat_i_gt after") if torch.isnan(feat_i_gt).any() else 1
                
#                 #print("NaN in rep[i] after") if torch.isnan(rep[i]).any() else 1
#                 sim = F.cosine_similarity(norm_feat_i_gt, norm_rep)
                
#                 # sim = torch.randn((feat_i_gt[0])).cuda()
#                 #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^sim0 ", sim)

#                 # Use mask_gt to select the class-specific probabilities
#                 # pred_class_i = pred[mask_gt, :][:, i]  # Select class-specific probabilities from pred

#                 # Ensure the shapes match for element-wise multiplication
#                 # sim = sim * pred_class_i  # Apply class-weighted probability
#                 #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^sim1 ", sim)

#                 # class_aware_loss += sim.mean()
#                 class_aware_loss.append(sim.mean())

#                 #-----missed class pixels
#                 # if i == 0:
#                 #     continue

#                 if i!=0: #if class is not background
#                     # Avoid inplace modification of mask_gt
#                     mask_gt_nonzero = (mask_gt != 0).long()  # Convert to float for boundary computation
#                     bound = self.get_segmentation_boundary_3d(mask_gt_nonzero.unsqueeze(1))
#                     fp_mask = (mask_pred & ~mask_gt) & bound
#                     fn_mask = (mask_gt & ~mask_pred) & bound 
#                     # ##print("-------------- ", torch.unique(fp_mask))
#                     # ##print("************* ", torch.unique(fn_mask))
#                     # mask_missed = (mask_pred ^ mask_gt) & bound

#                     # Compute missed boundary loss efficiently
#                     # if mask_missed.any():
#                     #     missed_voxels_feat = feat[mask_missed.bool(), :]
#                     #     # sim = torch.randn(missed_voxels_feat[0], device=pred.device)  # Random similarity for now
#                     #     missed_boundary_loss += sim.mean()
#                     #rep_norm = F.normalize(rep[i].clone(), p=2, dim=0, eps=1e-8)
#                     # if not torch.any(rep_norm!=0):
#                     #     ##print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^1")
#                     #     continue
#                     fn_mask =fn_mask.long()
#                     fp_mask=fp_mask.long()
#                     if 1 in torch.unique(fp_mask):
#                         missed_fp_voxels_feat = feat[fp_mask.bool(), :] 
#                         fp_norm = F.normalize(missed_fp_voxels_feat, p=2, dim=1, eps=1e-8)
#                         # if not torch.any(fp_norm!=0):
#                         #     ##print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^2")
#                         #     continue
#                         # if not is_fp:
#                         #     is_fp = True
#                         #     ##print("=======================================> FP existed <===================================")
                        
                        
#                         sim1 = F.cosine_similarity(fp_norm, norm_rep)
#                         ##print("^^^^^^^^^^^^^^^^^^^^^^^^^^^sim2 ", sim1)
#                         # sim1 = sim1 * torch.max(pred[fp_mask.bool(),:], dim=-1).values
#                         #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^sim3 ", sim1)
#                         # missed_boundary_loss1 = missed_boundary_loss1 + sim1.mean()
#                         missed_boundary_loss1.append(sim1.mean())
                    
#                     if 1 in torch.unique(fn_mask):
#                         missed_fn_voxels_feat = feat[fn_mask.bool(), :]
#                         fn_norm = F.normalize(missed_fn_voxels_feat, p=2, dim=1, eps=1e-8)
#                         # if not torch.any(fn_norm!=0):
#                         #     ##print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^3")
#                         #     continue
#                         sim2 = F.cosine_similarity(fn_norm, norm_rep)
#                         #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^sim4 ", sim2, fn_norm.shape, norm_rep.shape, sim2.mean())
#                         # p = sim2
#                         # sim2 = sim2 * pred[fn_mask.bool(), :][:,i]
#                         # #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^sim5 ", sim2)
#                         # missed_boundary_loss2 = missed_boundary_loss2 + sim2.mean()
#                         missed_boundary_loss2.append(sim2.mean())
#                 else:
#                     mask_gt_background = (mask_gt==0).long()
#                     bound = self.get_segmentation_boundary_3d(mask_gt_background.unsqueeze(1))
#                     bound = self.set_boundary_zero(bound, offset=2)
#                     fp_mask = (mask_pred & ~mask_gt) & bound
#                     fn_mask = (mask_gt & ~mask_pred) & bound

#                     fn_mask =fn_mask.long()
#                     fp_mask=fp_mask.long()
#                     if 1 in torch.unique(fp_mask):
#                         missed_fp_voxels_feat = feat[fp_mask.bool(), :]
#                         fp_norm = F.normalize(missed_fp_voxels_feat, p=2, dim=1, eps=1e-8)
#                         # if not torch.any(fp_norm!=0):
#                         #     ##print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^2")
#                         #     continue
#                         # if not is_fp:
#                         #     is_fp = True
#                         #     ##print("=======================================> FP existed <===================================")
                        
#                         missed_fp_voxels_feat = feat[fp_mask.bool(), :]
#                         sim1 = F.cosine_similarity(fp_norm, norm_rep)
#                         ##print("^^^^^^^^^^^^^^^^^^^^^^^^^^^sim6 ", sim1)
#                         # sim1 = sim1 * torch.max(pred[fp_mask.bool(),:], dim=-1).values
#                         #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^sim7 ", sim1, sim1.mean())
#                         # missed_boundary_loss1 = missed_boundary_loss1 + sim1.mean()
#                         missed_boundary_loss1.append(sim1.mean())
                    
#                     if 1 in torch.unique(fn_mask):
#                         missed_fn_voxels_feat = feat[fn_mask.bool(), :]
#                         fn_norm = F.normalize(missed_fn_voxels_feat, p=2, dim=1, eps=1e-8)
#                         # if not torch.any(fn_norm!=0):
#                         #     ##print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^3")
#                         #     continue
#                         sim2 = F.cosine_similarity(fn_norm, norm_rep)
#                         ##print("^^^^^^^^^^^^^^^^^^^^^^^^^^^sim8 ", sim2)
#                         # p = sim2
#                         # sim2 = sim2 * pred[fn_mask.bool(), :][:,i]
#                         #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^sim9 ", sim2)
#                         # missed_boundary_loss2 = missed_boundary_loss2 + sim2.mean()
#                         missed_boundary_loss2.append(sim2.mean())




#                 # ##print("++++++++++ ", (1-sim2).mean(), missed_boundary_loss, rep[i].shape)
#                 # missed_boundary_loss = missed_boundary_loss + sim1.mean() + 
#                 # Free up memory
#                 del mask_pred, mask_gt, feat_i_gt, norm_feat_i_gt, sim, bound
#                 torch.cuda.empty_cache()

#         # class_aware_loss += (1 - class_aware_loss / existed_cls) if existed_cls>0 else 0
#         # class_aware_loss += class_aware_loss1 / existed_cls1 if existed_cls1>0 else 0
#         # class_aware_loss += (1-class_aware_loss2/existed_cls2) if existed_cls2>0 else 0
#         class_aware_loss_ = 0
#         missed_boundary_loss_ = 0
#         # rep_loss_ = 0

#         class_aware_loss_ = class_aware_loss_ + ((1-torch.mean(torch.stack(class_aware_loss), dim=0)) if len(class_aware_loss)>0 else 0)
#         class_aware_loss_ = class_aware_loss_ + (torch.mean(torch.stack(class_aware_loss1), dim=0) if len(class_aware_loss1)>0 else 0)
#         class_aware_loss_ =class_aware_loss_ + ((1-torch.mean(torch.stack(class_aware_loss2), dim=0)) if len(class_aware_loss2)>0 else 0)
        
#         # missed_boundary_loss = (missed_boundary_loss1/(existed_cls) + (1-missed_boundary_loss2/ (existed_cls)))
#         # missed_boundary_loss_ = missed_boundary_loss_ + (torch.mean(torch.stack(missed_boundary_loss1), dim=0) if len(missed_boundary_loss1)>0 else 0)
#         # missed_boundary_loss_ = missed_boundary_loss_ + ((1-torch.mean(torch.stack(missed_boundary_loss2), dim=0)) if len(missed_boundary_loss2)>0 else 0)
#         # #print("&&&&&&&&&&&&&&&&&&&&&&&&&&& ", missed_boundary_loss2, torch.mean(torch.stack(missed_boundary_loss2)))
#         missed_boundary_loss1 = (torch.mean(torch.stack(missed_boundary_loss1), dim=0) if len(missed_boundary_loss1)>0 else 0)
#         missed_boundary_loss2 = ((1-torch.mean(torch.stack(missed_boundary_loss2), dim=0)) if len(missed_boundary_loss2)>0 else 0)
#         #print("&&&&&&&&&&&&&&&&&&++++++++++++++++++++++++++ ", missed_boundary_loss1, missed_boundary_loss2)

#         #-----representative
#         rep_nonzero = rep[rep.abs().sum(dim=1) != 0]
#         if rep_nonzero.shape[0]>1:
#             x_normalized = F.normalize(rep_nonzero, dim=1)
#             cosine_sim = torch.mm(x_normalized, x_normalized.t())
#             cosine_sim = cosine_sim.triu(diagonal=1)
#             rep_loss = 2 * torch.sum(cosine_sim) / (rep_nonzero.shape[0] * (rep_nonzero.shape[0] - 1))
#         #print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{class_aware_loss_, rep_loss, missed_boundary_loss_}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

#         return [torch.tensor(class_aware_loss_)/3, torch.tensor(rep_loss), torch.tensor(missed_boundary_loss1 + missed_boundary_loss2)/2] #, torch.tensor(missed_boundary_loss2)
from hymavi.network_architecture.hymavi_amos.config import config as cf
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.accumulated_rep = torch.zeros(cf.num_classes, cf.f_maps_dim[0]).to(device)
        self.num_accumulated = torch.zeros(cf.num_classes)
        self.alpha = 0.4

    def get_f_representative(self, x):
        #n_pixels, dims
        return x.mean(dim=0)

    def get_segmentation_boundary_3d(self, segmentation_mask):
        """
        Extracts the boundary of a 3D segmentation mask.

        Parameters:
        - segmentation_mask (torch.Tensor): A 4D tensor of shape (B, C, D, H, W)
        
        Returns:
        - boundary_mask (torch.Tensor): Tensor of the same shape as `segmentation_mask` containing boundaries (1s) and 0 elsewhere.
        """
        # Ensure input is 5D for batch operations
        assert segmentation_mask.ndim == 5, "Expected input shape (B, C, D, H, W)"

        # Create kernel with the correct channel size
        B, C, D, H, W = segmentation_mask.shape
        kernel = torch.ones((C, 1, 3, 3, 3), device=segmentation_mask.device)

        # Pad the input to ensure boundary is preserved
        padded_mask = F.pad(segmentation_mask.float(), (1, 1, 1, 1, 1, 1), mode='constant', value=0).clone()

        # Perform erosion by convolving the segmentation mask per channel
        eroded_mask = F.conv3d(padded_mask, kernel, stride=1, groups=C) == kernel[0].sum()

        # Extract boundary by XOR between the mask and eroded version
        boundary_mask = segmentation_mask & (~eroded_mask)


        return boundary_mask.squeeze(1)


    def set_boundary_zero(self, tensor, offset):
        """
        Set the boundary of a 4D tensor (B, D, H, W) to zero with an offset.
        Args:
            tensor: PyTorch tensor of shape (B, D, H, W)
            offset: Number of pixels/voxels from the edge to set as zero
        """
        B, D, H, W = tensor.shape

        # Set depth boundaries
        tensor[:, :offset, :, :] = 0
        tensor[:, -offset:, :, :] = 0

        # Set height boundaries
        tensor[:, :, :offset, :] = 0
        tensor[:, :, -offset:, :] = 0

        # Set width boundaries
        tensor[:, :, :, :offset] = 0
        tensor[:, :, :, -offset:] = 0
        return tensor
    
        


    def forward(self, pred, gt, feat, num_classes):
        B, C, D, H, W = feat.shape
        pred = pred.permute(0, 2, 3, 4, 1)  # B, D, H, W, C
        pred = F.softmax(pred, dim=-1)  # Softmax along the class dimension

        feat = feat.permute(0, 2, 3, 4, 1)
        pred_hard = torch.argmax(pred, dim=-1)  # B, D, H, W (for class)

        rep = torch.zeros((num_classes, C), device=pred.device)
        mean_fs = torch.zeros((num_classes, C), device=pred.device)

        class_aware_loss = []
        class_aware_loss1 = []
        class_aware_loss2 = []
        missed_boundary_loss1 = []
        missed_boundary_loss2 = []
        rep_loss = 0
        #print(f"::::::::::::::NaN in feat: {torch.isnan(feat).any(), torch.isnan(pred).any(), torch.isnan(gt).any()}")

        for i in range(num_classes):
            mask_pred = (pred_hard == i) & (gt == i)
            if 1 not in mask_pred:
                continue
            feat_i = feat[mask_pred, :]
            rep[i] = self.get_f_representative(feat_i)  # Representative feature for class i
            
            self.num_accumulated[i] = self.num_accumulated[i] + 1
            with torch.no_grad():
                # self.accumulated_rep[i] = self.accumulated_rep[i] + (rep[i] - self.accumulated_rep[i]) / self.num_accumulated[i]
                self.accumulated_rep[i] = self.alpha * self.accumulated_rep[i] + (1-self.alpha) * rep[i]
            # self.accumulated_rep[i] = self.accumulated_rep[i] + (rep[i]-self.accumulated_rep[i])/self.num_accumulated[i]
        
        for i in range(num_classes):
            if i in gt:
                mask_pred = (gt==i)
                feat_i = feat[mask_pred, :]
                # print("++++++++++++++++++++ ", feat_i.shape)
                norm_feat_i = F.normalize(feat_i, p=2, dim=1)

                mean_f = norm_feat_i.mean(dim=0, keepdim=True)
                mean_f = F.normalize(mean_f, p=2, dim=1)  # Normalize mean

                mean_fs[i] = mean_f

                if torch.any(rep[i]!=0):
                    norm_rep = F.normalize(rep[i].clone(), p=2, dim=0)
                    sim = F.cosine_similarity(norm_feat_i, norm_rep)
                elif self.num_accumulated[i]!=0:
                    norm_rep = F.normalize(self.accumulated_rep[i].clone(), p=2, dim=0)
                    sim = F.cosine_similarity(norm_feat_i, norm_rep)
                else:
                    
                    sim = F.cosine_similarity(norm_feat_i, mean_f)

                class_aware_loss.append(sim.mean())


                del mask_pred#, mask_gt, feat_i_gt, norm_feat_i_gt, sim#, bound
                torch.cuda.empty_cache()

        # class_aware_loss += (1 - class_aware_loss / existed_cls) if existed_cls>0 else 0
        # class_aware_loss += class_aware_loss1 / existed_cls1 if existed_cls1>0 else 0
        # class_aware_loss += (1-class_aware_loss2/existed_cls2) if existed_cls2>0 else 0
        class_aware_loss_ = 0
        missed_boundary_loss_ = 0
        # rep_loss_ = 0

        class_aware_loss_ = class_aware_loss_ + ((1-torch.mean(torch.stack(class_aware_loss), dim=0)) if len(class_aware_loss)>0 else 0)

        #-----representative
        rep_nonzero = rep[rep.abs().sum(dim=1) != 0]
        if rep_nonzero.shape[0]>1:
            x_normalized = F.normalize(rep_nonzero, dim=1)
            cosine_sim = torch.mm(x_normalized, x_normalized.t())
            cosine_sim = cosine_sim.triu(diagonal=1)
            rep_loss = 2 * torch.sum(cosine_sim) / (rep_nonzero.shape[0] * (rep_nonzero.shape[0] - 1))
        
        #-----mean_f
        mean_f_nonzero = mean_fs[mean_fs.abs().sum(dim=1) != 0]
        if mean_f_nonzero.shape[0]>1:
            x_normalized = F.normalize(mean_f_nonzero, dim=1)
            cosine_sim = torch.mm(x_normalized, x_normalized.t())
            cosine_sim = cosine_sim.triu(diagonal=1)
            mean_f_loss = 2 * torch.sum(cosine_sim) / (mean_f_nonzero.shape[0] * (mean_f_nonzero.shape[0] - 1))

        #print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{class_aware_loss_, rep_loss, missed_boundary_loss_}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

        return [torch.tensor(class_aware_loss_)/3, torch.tensor(rep_loss), torch.tensor(mean_f_loss)]
    
class CE_Dice_loss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4, smooth=1e-6):
        """
        Combines Dice Loss and Cross-Entropy Loss.
        
        Args:
            alpha (float): Weight for Dice Loss.
            beta (float): Weight for Cross-Entropy Loss.
            smooth (float): Smoothing constant for Dice Loss.
        """
        super(CE_Dice_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss(smooth)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, prediction, ground_truth):
        """
        Computes the combined loss.

        Args:
            prediction (torch.Tensor): Predicted tensor of shape (B, C, D, H, W).
            ground_truth (torch.Tensor): Ground truth tensor of shape (B, D, H, W).

        Returns:
            torch.Tensor: Combined Dice and Cross-Entropy loss.
        """
        # if len(ground_truth.shape)==5:
        #     ground_truth = ground_truth.squeeze(1)
        # Compute Dice Loss
        dice = self.dice_loss(prediction, ground_truth)
        
        # Compute Cross-Entropy Loss (expects ground truth to be of shape (B, D, H, W) and class indices)
        ce = self.cross_entropy_loss(prediction, ground_truth)
        
        # Combine losses
        loss = self.alpha * dice + self.beta * ce
        
        return loss
global cnt, to_print
cnt = 0
num_steps=250
num_losses = 4
to_print = torch.zeros((num_losses))
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()

        self.seg_loss = CE_Dice_loss()
        self.cbr = CustomLoss()
        self.weights = [25, 2, 1, 1]

    def forward(self, preds, masks, feat, num_classes):
        # print(">>>>>>>>>>>>>>>>>>///////////--------------------- ", torch.unique(masks), num_classes)

        loss_cst = self.cbr(preds, masks, feat, num_classes)
        loss_seg = self.seg_loss(preds, masks)

        losses = [loss_seg] + loss_cst
        total_loss = 0
        global cnt, to_print, num_steps
        for i in range(len(losses)):
            # print("*********** ", losses[i] * self.weights[i])
            tmp = losses[i] * self.weights[i]
            total_loss += tmp
            to_print[i] +=tmp.clone().cpu()
        
        cnt +=1
        if cnt==num_steps:    
            print("Component losses: ", to_print/num_steps)
            to_print=torch.zeros((num_losses))
            cnt=0
        
        return total_loss







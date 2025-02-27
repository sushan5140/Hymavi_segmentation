import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import random
from scipy.ndimage import label
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_tensor_values_given_coordinates_3d(img, x, y, z):
    """
    Utility function to get voxel value for coordinate tensors x, y, and z from a 5D tensor image (3D image).

    Input
    -----
    - img: tensor of shape (B, C, D, H, W)
    - x: tensor of shape (B, D, H, W)
    - y: tensor of shape (B, D, H, W)
    - z: tensor of shape (B, D, H, W)

    Returns
    -------
    - output: tensor of shape (B, C, D, H, W)
    """
    batch_size, C, D, H, W = img.shape

    # Create batch index for advanced indexing
    batch_idx = torch.arange(batch_size, device=img.device).view(batch_size, 1, 1, 1).expand(batch_size, D, H, W)
    
    # Stack batch and z, y, x indices for advanced indexing
    indices = torch.stack([batch_idx, z, y, x], dim=-1)  # Shape: (B, D, H, W, 4)

    # Permute img to shape (B, D, H, W, C) for advanced indexing
    output = img.permute(0, 2, 3, 4, 1)  # Shape: (B, D, H, W, C)
    output = output[indices[..., 0], indices[..., 1], indices[..., 2], indices[..., 3]]

    return output.permute(0, 4, 1, 2, 3)  # Back to shape (B, C, D, H, W)


def warp_tensor_according_to_flow(img, flow):
    """
    Function to warp a 5D image tensor according to 
    a 5D flow tensor.

    Input
    -----
    - img: tensor of shape (B, C, D, H, W) (3D image)
    - flow: tensor of shape (B, 3, D, H, W) (3D flow field)

    Returns
    -------
    - output: tensor of shape (B, C, D, H, W)
    """
    if len(img.shape)<5:
        img = img.unsqueeze(1)
    batch_size, C, D, H, W = img.shape

    # Create a grid of 3D coordinates (z, y, x)
    grid_z, grid_y, grid_x = torch.meshgrid(torch.arange(D, device=img.device), 
                                            torch.arange(H, device=img.device), 
                                            torch.arange(W, device=img.device), indexing='ij')
    
    # Stack in the correct (z, y, x) order
    grid = torch.stack((grid_z, grid_y, grid_x), dim=-1).float()  # Shape (D, H, W, 3)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # Shape (B, D, H, W, 3)

    # Add flow to the grid to obtain warped coordinates (displacements in each dimension)
    coords = grid + flow.permute(0, 2, 3, 4, 1)  # Shape (B, D, H, W, 3)

    # Split the coordinates into z, y, x to maintain consistency
    coords_z, coords_y, coords_x = coords[..., 0], coords[..., 1], coords[..., 2]

    # Compute interpolation corner points
    x0 = torch.floor(coords_x).long()
    x1 = x0 + 1
    y0 = torch.floor(coords_y).long()
    y1 = y0 + 1
    z0 = torch.floor(coords_z).long()
    z1 = z0 + 1

    # Clamp coordinates to image boundaries to avoid out-of-bound access
    x0c = x0.clamp(0, W - 1)
    x1c = x1.clamp(0, W - 1)
    y0c = y0.clamp(0, H - 1)
    y1c = y1.clamp(0, H - 1)
    z0c = z0.clamp(0, D - 1)
    z1c = z1.clamp(0, D - 1)

    # Get the values at the eight corner points
    I000 = get_tensor_values_given_coordinates_3d(img, x0c, y0c, z0c)
    I001 = get_tensor_values_given_coordinates_3d(img, x0c, y0c, z1c)
    I010 = get_tensor_values_given_coordinates_3d(img, x0c, y1c, z0c)
    I011 = get_tensor_values_given_coordinates_3d(img, x0c, y1c, z1c)
    I100 = get_tensor_values_given_coordinates_3d(img, x1c, y0c, z0c)
    I101 = get_tensor_values_given_coordinates_3d(img, x1c, y0c, z1c)
    I110 = get_tensor_values_given_coordinates_3d(img, x1c, y1c, z0c)
    I111 = get_tensor_values_given_coordinates_3d(img, x1c, y1c, z1c)

    # Calculate deltas for interpolation weights
    x0, x1 = x0.float(), x1.float()
    y0, y1 = y0.float(), y1.float()
    z0, z1 = z0.float(), z1.float()

    wa = (x1 - coords_x) * (y1 - coords_y) * (z1 - coords_z)
    wb = (x1 - coords_x) * (y1 - coords_y) * (coords_z - z0)
    wc = (x1 - coords_x) * (coords_y - y0) * (z1 - coords_z)
    wd = (x1 - coords_x) * (coords_y - y0) * (coords_z - z0)
    we = (coords_x - x0) * (y1 - coords_y) * (z1 - coords_z)
    wf = (coords_x - x0) * (y1 - coords_y) * (coords_z - z0)
    wg = (coords_x - x0) * (coords_y - y0) * (z1 - coords_z)
    wh = (coords_x - x0) * (coords_y - y0) * (coords_z - z0)

    # Expand the weights to match the channel dimension
    wa = wa.unsqueeze(1)  # Shape becomes [B, 1, D, H, W]
    wb = wb.unsqueeze(1)
    wc = wc.unsqueeze(1)
    wd = wd.unsqueeze(1)
    we = we.unsqueeze(1)
    wf = wf.unsqueeze(1)
    wg = wg.unsqueeze(1)
    wh = wh.unsqueeze(1)

    # Perform trilinear interpolation
    out = (wa * I000 + wb * I001 + wc * I010 + wd * I011 +
           we * I100 + wf * I101 + wg * I110 + wh * I111)
    
    return out

# def flow_warped_img_comparison_loss(wrapped_ED, target):
#     '''L1: unsupervised flow generation'''
#     "source, target: (B, C, D, H, W)"
#     diff = wrapped_ED - target
#     euclidean_dist = torch.sum(diff**2, dim=[1, 2, 3, 4])
#     return torch.mean(euclidean_dist, dim=0)



def process_frame(segmentation_frame, labels):
    """
    Process a single frame to get the largest connected components for each label.

    Args:
        segmentation_frame (np.ndarray): A 2D array representing a segmentation frame.
        labels (np.ndarray): An array of label values to process.

    Returns:
        np.ndarray: 2D array with largest connected components for each label in the frame.
    """
    output_frame = np.zeros_like(segmentation_frame)

    for label_val in labels:
        binary_mask = (segmentation_frame == label_val)
        labeled_array, num_features = label(binary_mask)
        
        if num_features > 0:
            component_sizes = [(i, np.sum(labeled_array == i)) for i in range(1, num_features + 1)]
            if label_val != 2:
                # Take the largest component
                largest_component = max(component_sizes, key=lambda x: x[1])[0]
                largest_mask = (labeled_array == largest_component)
                output_frame[largest_mask] = label_val
            else:
                # Sort components by size in descending order and take the two largest
                two_largest_components = sorted(component_sizes, key=lambda x: x[1], reverse=True)[:2]
                
                if len(two_largest_components) > 1 and two_largest_components[1][1] / two_largest_components[0][1] <= 1 / 5:
                    two_largest_components = [two_largest_components[0]]
                
                for component, _ in two_largest_components:
                    largest_mask = (labeled_array == component)
                    output_frame[largest_mask] = label_val

    return output_frame

def get_largest_connected_components(segmentation):
    """
    Get the largest connected components for each label in each 2D slice of a 4D segmentation using multithreading.

    Args:
        segmentation (torch.Tensor): 5D tensor of shape (B, C, D, H, W) containing labels.

    Returns:
        torch.Tensor: 5D tensor with largest connected components for each label.
    """
    B, C, D, H, W = segmentation.shape
    assert C == 1, "The channel dimension C should be 1."

    segmentation_np = segmentation.detach().cpu().numpy()
    output = np.zeros_like(segmentation_np)

    labels = np.array([1, 2, 3])  # Exclude background

    with ThreadPoolExecutor() as executor:
        futures = {}
        for b in range(B):
            for d in range(D):
                frame = segmentation_np[b, 0, d]
                futures[executor.submit(process_frame, frame, labels)] = (b, d)

        for future in as_completed(futures):
            b, d = futures[future]
            output[b, 0, d] = future.result()

    return torch.from_numpy(output).to(segmentation.device)




def smoothness_loss(flow, img):
    """
    Computes a smoothness loss to ensure the optical flow is smooth across the image,
    with improved edge-aware weighting and boundary handling.
    
    Args:
        flow (torch.Tensor): The predicted optical flow. Shape (N, C, D, H, W).
        img (torch.Tensor): The reference image for edge-aware smoothness. Shape (N, C, D, H, W).
    
    Returns:
        torch.Tensor: The computed loss.
    """
    def gradient(x):
        """
        Compute gradients of the input tensor along each axis, with boundary handling.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Gradients along x, y, and z axes.
        """
        D_dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        D_dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        D_dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        return D_dx, D_dy, D_dz
    
    flow_dx, flow_dy, flow_dz = gradient(flow)
    img_dx, img_dy, img_dz = gradient(img)

    # Compute edge-aware weights
    edge_weight_x = torch.exp(-torch.mean(torch.abs(img_dx), dim=1, keepdim=True))
    edge_weight_y = torch.exp(-torch.mean(torch.abs(img_dy), dim=1, keepdim=True))
    edge_weight_z = torch.exp(-torch.mean(torch.abs(img_dz), dim=1, keepdim=True))

    # Apply weights to the flow gradients
    weighted_flow_dx = flow_dx * edge_weight_x
    weighted_flow_dy = flow_dy * edge_weight_y
    weighted_flow_dz = flow_dz * edge_weight_z

    # Compute smoothness loss
    smoothness_loss_x = torch.mean(torch.abs(weighted_flow_dx))
    smoothness_loss_y = torch.mean(torch.abs(weighted_flow_dy))
    smoothness_loss_z = torch.mean(torch.abs(weighted_flow_dz))
    
    return smoothness_loss_x + smoothness_loss_y + smoothness_loss_z


def flow_warped_img_comparison_loss(warped_img, target_img, predicted_flow, img):
    """
    Combines multiple loss components for a total motion flow loss.
    
    Args:
        warped_img (torch.Tensor): The warped image.
        target_img (torch.Tensor): The target image.
        predicted_flow (torch.Tensor): The predicted optical flow.
        img (torch.Tensor): The reference image for smoothness calculation.
    
    Returns:
        torch.Tensor: The total motion flow loss.
    """
    # img_loss = dssim_flow_loss(target_img, warped_img)
    img_loss = F.mse_loss(warped_img, target_img)
    smooth_loss = smoothness_loss(predicted_flow, img)
    
    # Combine losses with weighting factors (e.g., 1.0 for image loss, 0.1 for smoothness loss)
    return img_loss + 0.2 * smooth_loss

def dssim_flow_loss(img1, img2, win_size=5):
    """
    Calculate dSSIM loss between two batches of 3D images.
    
    Args:
        img1 (torch.Tensor): Batch of original images. Shape (N, C, D, H, W).
        img2 (torch.Tensor): Batch of images to compare. Shape (N, C, D, H, W).
        win_size (int): Window size for SSIM calculation. Must be odd and smaller than the image dimensions.
    
    Returns:
        torch.Tensor: Mean dSSIM loss for the batch.
    """
    # Ensure the input images are in the range [0, 1]
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)

    batch_size = img1.size(0)
    dssim_loss = 0.0

    for i in range(batch_size):
        img1_np = img1[i].cpu().detach().numpy()
        img2_np = img2[i].cpu().detach().numpy()

        ssim_vals = []
        # Calculate SSIM for each slice
        for j in range(img1_np.shape[0]):  # Iterate over depth (D)
            ssim_index = ssim(
                img1_np[j, :, :], 
                img2_np[j, :, :], 
                data_range=img2_np[j, :, :].max() - img2_np[j, :, :].min(),
                win_size=win_size,
                multichannel=False  # Set to True if the image has multiple color channels
            )
            ssim_vals.append(ssim_index)
        
        mean_ssim = torch.tensor(ssim_vals).mean()
        dssim_loss += 1 - mean_ssim
    
    return dssim_loss / batch_size

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

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_f_representative(self, x):
        #B, C, D, H, W
        # B, C, D, H, W = x.shape
        # x = rearrange(x, 'b c d h w -> (b d h w) c', b=B, d=D, h=H, w=W)
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
    
    def forward(self, pred, gt, feat, num_classes):
        B, C, D, H, W = feat.shape
        pred = pred.permute(0, 2, 3, 4, 1)  # B, D, H, W, C
        pred = F.softmax(pred, dim=-1)  # Softmax along the class dimension

        feat = feat.permute(0, 2, 3, 4, 1)
        pred_hard = torch.argmax(pred, dim=-1)  # B, D, H, W (for class)

        rep = torch.zeros((num_classes, C), device=pred.device)

        class_aware_loss = 0
        missed_boundary_loss1 = 0
        missed_boundary_loss2 = 0
        rep_loss = 0

        for i in range(num_classes):
            # Mask where the ground truth is class `i`
            mask_pred = pred_hard == i  # Avoid cloning unless necessary
            mask_gt = gt == i  # Avoid cloning unless necessary

            # Get the features corresponding to the ground truth class `i`
            feat_i = feat[mask_pred, :]
            feat_i_gt = feat[mask_gt, :]

            rep[i] = self.get_f_representative(feat_i)  # Representative feature for class i

            # Compute cosine similarity between gt and representative feature
            norm_feat_i_gt = F.normalize(feat_i_gt, p=2, dim=1)
            norm_rep = F.normalize(rep[i].clone(), p=2, dim=0)  # No need to clone here
            sim = F.cosine_similarity(norm_feat_i_gt, norm_rep)
            # sim = torch.randn((feat_i_gt.shape[0])).cuda()

            # Use mask_gt to select the class-specific probabilities
            pred_class_i = pred[mask_gt, :][:, i]  # Select class-specific probabilities from pred

            # Ensure the shapes match for element-wise multiplication
            sim = sim * pred_class_i  # Apply class-weighted probability

            class_aware_loss += sim.mean()

            #-----missed class pixels
            if i == 0:
                continue

            # Avoid inplace modification of mask_gt
            mask_gt_nonzero = (mask_gt != 0).long()  # Convert to float for boundary computation
            bound = self.get_segmentation_boundary_3d(mask_gt_nonzero.unsqueeze(1))
            fp_mask = (mask_pred & ~mask_gt) & bound
            fn_mask = (mask_gt & ~mask_pred) & bound 
            # print("-------------- ", torch.unique(fp_mask))
            # print("************* ", torch.unique(fn_mask))
            # mask_missed = (mask_pred ^ mask_gt) & bound

            # Compute missed boundary loss efficiently
            # if mask_missed.any():
            #     missed_voxels_feat = feat[mask_missed.bool(), :]
            #     # sim = torch.randn(missed_voxels_feat.shape[0], device=pred.device)  # Random similarity for now
            #     missed_boundary_loss += sim.mean()
            #rep_norm = F.normalize(rep[i].clone(), p=2, dim=0, eps=1e-8)
            # if not torch.any(rep_norm!=0):
            #     print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^1")
            #     continue
            fn_mask =fn_mask.long()
            fp_mask=fp_mask.long()
            if 1 in torch.unique(fp_mask):
                fp_norm = F.normalize(missed_fp_voxels_feat, p=2, dim=1, eps=1e-8)
                # if not torch.any(fp_norm!=0):
                #     print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^2")
                #     continue
                if not is_fp:
                    is_fp = True
                    print("=======================================> FP existed <===================================")
                
                missed_fp_voxels_feat = feat[fp_mask.bool(), :]
                sim1 = F.cosine_similarity(fp_norm, norm_rep)
                # sim1 = sim1 * torch.max(pred[fp_mask.bool(),:], dim=-1).values
                missed_boundary_loss1 = missed_boundary_loss1 + sim1.mean()
            
            if 1 in torch.unique(fn_mask):
                missed_fn_voxels_feat = feat[fn_mask.bool(), :]
                fn_norm = F.normalize(missed_fn_voxels_feat, p=2, dim=1, eps=1e-8)
                # if not torch.any(fn_norm!=0):
                #     print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^3")
                #     continue
                sim2 = F.cosine_similarity(fn_norm, norm_rep)
                # p = sim2
                # sim2 = sim2 * 1/(pred[fn_mask.bool(), :][:,i]+1e-3)
                # sim2 = sim2 * pred[fn_mask.bool(), :][:,i]
                missed_boundary_loss2 = missed_boundary_loss2 + sim2.mean()

            # print("++++++++++ ", (1-sim2).mean(), missed_boundary_loss, rep[i].shape)
            # missed_boundary_loss = missed_boundary_loss + sim1.mean() + 
            # Free up memory
            del mask_pred, mask_gt, feat_i, feat_i_gt, norm_feat_i_gt, sim, pred_class_i, mask_gt_nonzero, bound
            torch.cuda.empty_cache()

        class_aware_loss = 1 - class_aware_loss / num_classes
        missed_boundary_loss = (missed_boundary_loss1/(num_classes-1) + (1-missed_boundary_loss2/ (num_classes-1)))

        #-----representative
        x_normalized = F.normalize(rep, dim=1)
        cosine_sim = torch.mm(x_normalized, x_normalized.t())
        cosine_sim = cosine_sim.triu(diagonal=1)
        rep_loss = 2 * torch.sum(cosine_sim) / (num_classes * (num_classes - 1))

        return class_aware_loss, rep_loss, missed_boundary_loss


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2, smooth=1e-6):
        """
        Combines Dice Loss and Cross-Entropy Loss.
        
        Args:
            alpha (float): Weight for Dice Loss.
            beta (float): Weight for Cross-Entropy Loss.
            smooth (float): Smoothing constant for Dice Loss.
        """
        super(CombinedLoss, self).__init__()
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
        combined_loss = self.alpha * dice + self.beta * ce
        
        return combined_loss
seg_loss_ce_dice = CombinedLoss()

class DiceLoss2D(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss2D, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, ground_truth):
        """
        Computes the Dice Loss for 2D data.
        
        Args:
            prediction (torch.Tensor): Predicted tensor of shape (B, C, H, W).
            ground_truth (torch.Tensor): Ground truth tensor of shape (B, H, W).

        Returns:
            torch.Tensor: Dice loss for each class, averaged across the batch.
        """
        # Convert prediction to probability using softmax
        prediction = F.softmax(prediction, dim=1)  # Shape: (B, C, H, W)

        # Convert ground truth to one-hot encoding
        B, C, H, W = prediction.shape
        ground_truth_one_hot = F.one_hot(ground_truth, num_classes=C).permute(0, 3, 1, 2).float()  # Shape: (B, C, H, W)

        # Compute Dice score
        intersection = torch.sum(prediction * ground_truth_one_hot, dim=(2, 3))
        union = torch.sum(prediction, dim=(2, 3)) + torch.sum(ground_truth_one_hot, dim=(2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score  # Dice Loss is 1 - Dice Coefficient

        return dice_loss.mean()  # Mean Dice loss over batch

class CombinedLoss2D(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2, smooth=1e-6):
        """
        Combines Dice Loss and Cross-Entropy Loss for 2D data.
        
        Args:
            alpha (float): Weight for Dice Loss.
            beta (float): Weight for Cross-Entropy Loss.
            smooth (float): Smoothing constant for Dice Loss.
        """
        super(CombinedLoss2D, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss2D(smooth)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, prediction, ground_truth):
        """
        Computes the combined loss.

        Args:
            prediction (torch.Tensor): Predicted tensor of shape (B, C, H, W).
            ground_truth (torch.Tensor): Ground truth tensor of shape (B, H, W).

        Returns:
            torch.Tensor: Combined Dice and Cross-Entropy loss.
        """
        # Compute Dice Loss
        dice = self.dice_loss(prediction, ground_truth)
        
        # Compute Cross-Entropy Loss (expects ground truth to be of shape (B, H, W) and class indices)
        ce = self.cross_entropy_loss(prediction, ground_truth)
        
        # Combine losses
        combined_loss = self.alpha * dice + self.beta * ce
        
        return combined_loss
    
def dice_loss_warpEDES(y_true, y_pred, smooth=1.0):
    """
    Dice loss for binary masks.
    """
    # y_true_flat = y_true.view(-1)
    # y_pred_flat = y_pred.view(-1)
    # intersection = (y_true_flat * y_pred_flat).sum()
    # dice = (2. * intersection + smooth) / (y_true_flat.sum() + y_pred_flat.sum() + smooth)
    # return 1 - dice

    # Flatten the tensors along all dimensions except the batch
    y_true_flat = y_true.view(y_true.size(0), -1)
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    
    # Compute the intersection and union
    intersection = (y_true_flat * y_pred_flat).sum(1)
    denominator = y_true_flat.sum(1) + y_pred_flat.sum(1)
    
    # Calculate Dice coefficient
    dice_coeff = (2. * intersection + smooth) / (denominator + smooth)
    
    # Compute Dice loss
    dice_loss_value = 1-dice_coeff.mean()
    
    return dice_loss_value

def flow_warped_gt_comparison_dice_loss(ED_mask, ES_mask, flow):
    """
    Calculate Dice loss between the warped ES_mask and ED_mask in PyTorch.
    
    Args:
        ED_mask (torch.Tensor): Ground truth mask for the ED frame, with classes 1, 2, 3.
        ES_mask (torch.Tensor): Ground truth mask for the ES frame, with classes 1, 2, 3.
        flow (torch.Tensor): Flow field to warp ES_mask toward ED_mask.
        warp_func (function): Function to warp tensor according to flow.
    
    Returns:
        torch.Tensor: Total Dice loss for each class summed up.
    """
    # Extract class-specific masks for ES
    ES_mask_1 = torch.where(ES_mask == 1, torch.ones_like(ES_mask), torch.zeros_like(ES_mask))
    ES_mask_2 = torch.where(ES_mask == 2, torch.ones_like(ES_mask), torch.zeros_like(ES_mask))
    ES_mask_3 = torch.where(ES_mask == 3, torch.ones_like(ES_mask), torch.zeros_like(ES_mask))
    
    # Extract class-specific masks for ED
    ED_mask_1 = torch.where(ED_mask == 1, torch.ones_like(ED_mask), torch.zeros_like(ED_mask))
    ED_mask_2 = torch.where(ED_mask == 2, torch.ones_like(ED_mask), torch.zeros_like(ED_mask))
    ED_mask_3 = torch.where(ED_mask == 3, torch.ones_like(ED_mask), torch.zeros_like(ED_mask))
    
    # Warp each class of ES mask according to the flow, num_classes = 2 because there are only class 0 and 1 now
    # warped_ED_mask_1 = warp_tensor_according_to_flow(F.one_hot(ED_mask_1.cuda(), num_classes=2).permute(0,4,1,2,3), flow)
    # warped_ED_mask_2 = warp_tensor_according_to_flow(F.one_hot(ED_mask_2.cuda(), num_classes=2).permute(0,4,1,2,3), flow)
    # warped_ED_mask_3 = warp_tensor_according_to_flow(F.one_hot(ED_mask_3.cuda(), num_classes=2).permute(0,4,1,2,3), flow)
    #print("///////// ", ED_mask_1.shape, ES_mask_1.shape)
    warped_ED_mask_1 = warp_tensor_according_to_flow(ED_mask_1, flow).squeeze(1)
    warped_ED_mask_2 = warp_tensor_according_to_flow(ED_mask_2, flow).squeeze(1)
    warped_ED_mask_3 = warp_tensor_according_to_flow(ED_mask_3, flow).squeeze(1)

    #print("--------- ", warped_ED_mask_1.shape)


    # warped_ED_mask_1 = torch.argmax(warped_ED_mask_1, dim=1)
    # warped_ED_mask_2 = torch.argmax(warped_ED_mask_2, dim=1)
    # warped_ED_mask_3 = torch.argmax(warped_ED_mask_3, dim=1)
    
    # Calculate Dice loss for each class between warped ES mask and ED mask
    loss1 = dice_loss_warpEDES(ES_mask_1, warped_ED_mask_1, smooth=1.0)
    loss2 = dice_loss_warpEDES(ES_mask_2, warped_ED_mask_2, smooth=1.0)
    loss3 = dice_loss_warpEDES(ES_mask_3, warped_ED_mask_3, smooth=1.0)

    #print("+++++++++ ", loss1, loss2, loss3)
    
    return (loss1 + loss2 + loss3)/3

dice_loss_3d = DiceLoss(smooth=1.0)
def t_seg_loss(t_seg, ED_seg, wrapped_ED_seg_gt, wrapped_ED_seg, is_ES, masks, curr_epoch, n_warm_up =20, criterion = nn.CrossEntropyLoss(), flow_ED2t=None, num='train'):
    """
    Calculate the segmentation loss.
    Inputs:
    - t_seg, ED_seg, wrapped_ED_seg_gt, wrapped_ED_seg: 5D tensors (B, C, D, H, W)
    - is_ES: Binary tensor indicating if the frame is ES or not
    - masks: Ground truth segmentation masks
    - curr_epoch: Current training epoch
    - n_warm_up: Number of epochs for warm-up
    """
    criterion1 = nn.KLDivLoss(reduction="batchmean")
    criterion2 = nn.MSELoss()
    criterion3 = nn.L1Loss()
    
    if True in is_ES:  # if the current batch contains ES frame
        ES_seg = t_seg[is_ES==True]
        ES_masks = masks[is_ES==True][:, 1, :, :, :].cuda()
        

        l1 = seg_loss_ce_dice(ED_seg, masks[:, 0, :, :, :])
        

        #l2 = seg_loss_ce_dice(ES_seg, ES_masks) + seg_loss_ce_dice(wrapped_ED_seg[is_ES==True], ES_masks).
        b,c,d,h,w = ED_seg.shape
        # print(f"::::::{num}:::: ", ES_seg.shape, ES_masks.shape, masks.shape)
        
        l3 = seg_loss_ce_dice(ES_seg, ES_masks)
        #l4 = seg_loss_ce_dice(wrapped_ED_seg[is_ES==True], ES_masks)
        #print("+++++++++++++++----------****** ",  flow_ED2t.shape)
        '''mask_ED2ES = warp_tensor_according_to_flow(F.one_hot(masks[:,0, :,:,:][is_ES==True], num_classes=4).permute(0,4,1,2,3), flow_ED2t.contiguous()[is_ES==True])
        
        #mask_ED2ES = mask_ED2ES.contiguous().view(-1, d, c, h, w).permute(0,2,1,3,4)
        #print(">>>>>>>>>> ", mask_ED2ES.shape)
        mask_ED2ES=torch.argmax(mask_ED2ES, dim=1)
        
        l4= dice_loss_3d(mask_ED2ES.float(), masks[:,1,:,:,:][is_ES==True])'''
        l4 = flow_warped_gt_comparison_dice_loss(masks[:,0, :,:,:][is_ES==True], masks[:,1, :,:,:][is_ES==True], flow_ED2t.contiguous()[is_ES==True])
        # l4= torch.tensor(0).cuda()


    else:
        l1 = seg_loss_ce_dice(ED_seg, masks[:, 0, :, :, :])
        
        # if curr_epoch >= n_warm_up:
        #     l2 = seg_loss_ce_dice(t_seg, wrapped_ED_seg_gt[is_ES==False])
        # else:
        #     l2 = torch.tensor(0.0).cuda()
        l3, l4 = torch.tensor(0).cuda(), torch.tensor(0).cuda()
    
    #print("::::::::::::::: ", wrapped_ED_seg_gt.shape, wrapped_ED_seg_gt[is_ES==False].shape)
    
    if False in is_ES:
        t_seg = t_seg[is_ES==False]
        l2 = seg_loss_ce_dice(t_seg, wrapped_ED_seg_gt[is_ES==False].long()) if curr_epoch>=n_warm_up else torch.tensor(0.0).cuda()
    else:
        l2 = torch.tensor(0.0).cuda()
        
    return l1, l4, l2, l3

def flow_diffeomorphism_loss(flow):
    batch_size, _, H, W = flow.shape
    flow = flow.permute(0,2,3,1)

    y, x = torch.meshgrid(torch.arange(H, device=flow.device), torch.arange(W, device=flow.device), indexing='ij')
    x = x.unsqueeze(0).unsqueeze(-1).float()
    y = y.unsqueeze(0).unsqueeze(-1).float()

    grid = torch.concat([x, y], dim=-1)  # Shape: (batch_size, H, W, 2)
    coords = grid + flow

    


    coords_x = coords[..., 0].unsqueeze(-1)
    coords_y = coords[..., 1].unsqueeze(-1)
    

    coords_x0 = coords_x[..., :-1, :]
    coords_x1 = coords_x[..., 1:, :]
    coords_y0 = coords_y[:, :-1, :, :]
    coords_y1 = coords_y[:, 1:, :, :]


    x_adjacent_diff_clip = torch.clamp(coords_x1 - coords_x0, max=0)
    y_adjacent_diff_clip = torch.clamp(coords_y1 - coords_y0, max=0)

    global id0, id1, id2, id3, id4, id5
    id0 = coords_x0
    id1 = coords_x1
    id2 = coords_y0
    id3 = coords_y1
    id4= x_adjacent_diff_clip
    id5 = y_adjacent_diff_clip

    x_adjacent_diff_clip = x_adjacent_diff_clip.pow(2)
    y_adjacent_diff_clip = y_adjacent_diff_clip.pow(2)

    loss = torch.mean(torch.sum(x_adjacent_diff_clip, dim=[1, 2, 3]) + torch.sum(y_adjacent_diff_clip, dim=[1, 2, 3]))
    
    return loss

def shuffle_with_seed(arr, seed=2024):
    random.seed(seed)
    random.shuffle(arr)
    return arr


def get_all_subs(gt_path, mode ='train', dataset='both'): #both ACDC and UKB
    gt_path += '/labelsTr'
    if mode == 'train':
        list_subs_ukb = []
        list_subs_acdc = []
        for f in os.listdir(gt_path):
            sub = f.split('_')[0]
            if (sub[:7]!= 'patient') and (sub not in list_subs_ukb):
                list_subs_ukb.append(sub)
            if (sub[:7]== 'patient') and (sub not in list_subs_acdc):
                list_subs_acdc.append(sub)
        
        list_subs_ukb = shuffle_with_seed(list_subs_ukb)
        list_subs_acdc = shuffle_with_seed(list_subs_acdc)

        if dataset=='both':
            train_subs = list_subs_ukb[:int(0.8*len(list_subs_ukb))] + list_subs_acdc[:int(0.8*len(list_subs_acdc))]
            val_subs = list_subs_ukb[int(0.8*len(list_subs_ukb)):] + list_subs_acdc[int(0.8*len(list_subs_acdc)):]
        elif dataset=='ACDC':
            train_subs = list_subs_acdc[:int(0.8*len(list_subs_acdc))]
            val_subs = list_subs_acdc[int(0.8*len(list_subs_acdc)):]
        elif dataset=='UKB':
            train_subs = list_subs_ukb[:int(0.8*len(list_subs_ukb))] 
            val_subs = list_subs_ukb[int(0.8*len(list_subs_ukb)):] 

        random.shuffle(train_subs)
        random.shuffle(val_subs)
            
        return train_subs, val_subs
    
    else:
        list_subs_ukb = []
        list_subs_acdc = []
        for f in os.listdir(gt_path):
            sub = f.split('_')[0]
            if (sub[:7]!= 'patient') and (sub not in list_subs_ukb):
                list_subs_ukb.append(sub)
            if (sub[:7]== 'patient') and (sub not in list_subs_acdc):
                list_subs_acdc.append(sub)
        
        if dataset=='both':
            test_subs = list_subs_ukb + list_subs_acdc 
        elif dataset=='ACDC':
            test_subs = list_subs_acdc
        elif dataset=='UKB':
            test_subs=list_subs_ukb
            
        return test_subs
    

def c2f_loss(fine_hr, fine_shr, is_ES, masks, masks_shr, mode):
    if mode == 'train':
        # target_mask = masks[:, 0, :, :, :]
        # if True in is_ES:
        #     target_mask = torch.cat([target_mask, masks[is_ES==True][:, 1, :, :, :]], dim=0)
        hr_loss = seg_loss_ce_dice(fine_hr, masks)
        
        shr_loss = seg_loss_ce_dice(fine_shr, masks_shr)

        return hr_loss, shr_loss
    else:
        # target_mask = masks[:, 0, :, :, :]
        # if True in is_ES:
        #     target_mask = torch.cat([target_mask, masks[is_ES==True][:, 1, :, :, :]], dim=0)
        hr_loss = seg_loss_ce_dice(fine_hr, masks)

        return hr_loss, torch.tensor(0).cuda()


def dice_score(prediction, ground_truth, smooth=1e-6):
    """
    Compute the Dice score for 3D image segmentation.
    
    Args:
        prediction (torch.Tensor): Predicted tensor of shape (B, C, D, H, W).
        ground_truth (torch.Tensor): Ground truth tensor of shape (B, D, H, W).
        smooth (float): A small smoothing constant to avoid division by zero.
    
    Returns:
        torch.Tensor: Dice score for each class and average Dice score across all classes.
    """
    # Get number of classes from prediction
    B, C, D, H, W = prediction.shape
    
    # Convert prediction to binary (one-hot encoded format)
    prediction = torch.argmax(prediction, dim=1)  # Shape: (B, D, H, W)
    
    # Initialize dice scores
    dice_scores = []
    # print("++++++++++++++ ", prediction.shape)
    
    # Calculate Dice score for each class
    for c in range(C):
        if c==0: #-------------------------------------------------------------------------------------
            continue
        # Create binary mask for class `c` in predictions and ground truth
        pred_c = (prediction == c).float()  # Shape: (B, D, H, W)
        gt_c = (ground_truth == c).float()  # Shape: (B, D, H, W)
        
        # Compute intersection and union
        intersection = torch.sum(pred_c * gt_c, dim=(1, 2, 3))  # Sum over D, H, W for each batch
        union = torch.sum(pred_c, dim=(1, 2, 3)) + torch.sum(gt_c, dim=(1, 2, 3))  # Sum over D, H, W for each batch
        
        # Compute Dice score for this class
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
    
    # Stack Dice scores for each class: Shape (C, B)
    dice_scores = torch.stack(dice_scores, dim=0)  # Shape: (C, B)
    
    # Mean Dice score for each class (average over batch)
    mean_dice_per_class = torch.mean(dice_scores, dim=1)  # Shape: (C,)
    
    # Mean Dice score across all classes
    mean_dice_all_classes = torch.mean(mean_dice_per_class)  # Scalar
    
    return mean_dice_per_class.cpu().numpy(), mean_dice_all_classes.cpu().numpy()

def dice_score_2d(prediction, ground_truth, smooth=1e-6):
    """
    Compute the Dice score for 2D image segmentation.
    
    Args:
        prediction (torch.Tensor): Predicted tensor of shape (B, C, H, W).
        ground_truth (torch.Tensor): Ground truth tensor of shape (B, H, W).
        smooth (float): A small smoothing constant to avoid division by zero.
    
    Returns:
        torch.Tensor: Dice score for each class and average Dice score across all classes.
    """
    # Get number of classes from prediction
    B, C, H, W = prediction.shape
    
    # Convert prediction to binary (one-hot encoded format)
    prediction = torch.argmax(prediction, dim=1)  # Shape: (B, H, W)
    
    # Initialize dice scores
    dice_scores = []
    
    # Calculate Dice score for each class
    for c in range(C):
        # Create binary mask for class `c` in predictions and ground truth
        pred_c = (prediction == c).float()  # Shape: (B, H, W)
        gt_c = (ground_truth == c).float()  # Shape: (B, H, W)
        
        # Compute intersection and union
        intersection = torch.sum(pred_c * gt_c, dim=(1, 2))  # Sum over H, W for each batch
        union = torch.sum(pred_c, dim=(1, 2)) + torch.sum(gt_c, dim=(1, 2))  # Sum over H, W for each batch
        
        # Compute Dice score for this class
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
    
    # Stack Dice scores for each class: Shape (C, B)
    dice_scores = torch.stack(dice_scores, dim=0)  # Shape: (C, B)
    
    # Mean Dice score for each class (average over batch)
    mean_dice_per_class = torch.mean(dice_scores, dim=1)  # Shape: (C,)
    
    # Mean Dice score across all classes
    mean_dice_all_classes = torch.mean(mean_dice_per_class)  # Scalar
    
    return mean_dice_per_class, mean_dice_all_classes

def calculate_metrics(t_seg, masks):
    #---------------------------dice score------------------------------

    # dice_score_seg = dice_coefficient(t_seg, masks)
    # dice_score_seg_per_class = dice_coefficient_per_class(t_seg, masks)

    dice_score_seg_per_class, dice_score_seg = dice_score(t_seg, masks)


    return dice_score_seg, dice_score_seg_per_class

def torch3D_Hausdorff_distance(x, y):
    """
    Calculate the Hausdorff distance for 3D volumes.

    Args:
        x (torch.Tensor): Tensor of shape (B, D, H, W)
        y (torch.Tensor): Tensor of shape (B, D, H, W)

    Returns:
        torch.Tensor: Hausdorff distance between x and y
    """
    x = x.float()
    y = y.float()
    distance_matrix = torch.cdist(x, y, p=2)  # Euclidean distance

    value1 = distance_matrix.min(3)[0].max(2)[0].max(1, keepdim=True)[0]
    value2 = distance_matrix.min(2)[0].max(2)[0].max(1, keepdim=True)[0]

    value = torch.cat((value1, value2), dim=1)
    value = value.max(1)[0].mean(dim=0).mean()  # mean over batches and classes

    return value

def calculate_segment_metrics(ED_seg, t_seg, masks, is_ES, ED_frs, t_frames, num='Train'):
    #---------------------------dice score------------------------------
    dice_score_ES_seg = 0
    dice_score_ES_seg_per_class = 0

    # dice_score_ED_seg = dice_coefficient(ED_seg, masks[:, 0, :, :, :])
    # dice_score_ED_seg_per_class = dice_coefficient_per_class(ED_seg, masks[:, 0, :, :, :])
    dice_score_ED_seg, dice_score_ED_seg_per_class = calculate_metrics(ED_seg, masks[:, 0, :, :, :])

    if True in is_ES:
        # try:
        #     ES_seg = t_seg[is_ES==True]
        # except:
        #     print("^^^^^^^^ ", ED_seg.shape, masks.shape)
        #     print(">>>>>>>> error: ", t_seg.shape)
        #     print(">>>>>>1: ",  is_ES)
        #     print("/////3: ", is_ES.shape)
        #     print("/////4: ", is_ES==True)
        #     print(t_seg)
        #     print(">>>>>>>5: ", t_seg[is_ES==True].shape)

        try:

            ES_seg = t_seg[is_ES==True]
        except Exception as e:

            print(f"^^^^^^^^^^^^^^^^^^{num}--------------- ")
            print(f"------{num}------Indexing failed: {e}")
            print(f"-------{num}----t_seg shape: {t_seg.shape}, is_ES shape: {is_ES.shape} ", t_seg.dtype )
            print(f"^^^^^^^^{num} ", ED_seg.shape, masks.shape)
            print(f">>>>>>>>{num} error: ", t_seg.shape)
            print(f">>>{num}>>>1: ",  is_ES)
            print(f"///{num}//3: ", is_ES.shape)
            print(f"//{num}///4: ", is_ES==True)
            print(f"@@{num}@@@ ", ED_frs.shape, t_frames.shape)
            print(f"/^^^{num}&&&& ", torch.min(masks), torch.max(masks))
            print(t_seg)
            
            # print("<<<<<2: ", torch.min(t_seg.view(-1)), torch.max(t_seg.view(-1)))
    
    if False in is_ES:
        t_seg = t_seg[is_ES==False]
        dice_score_t_seg, _ = calculate_metrics(t_seg, masks[is_ES==False][:, 1, :, :, :])
    
    if True in is_ES:
        # print("+++++++++++++++ ", ES_seg.shape, masks.shape, ED_seg.shape)
        # print(masks[:, 1, :, :, :].shape)
        dice_score_ES_seg, _ = calculate_metrics(ES_seg, masks[:, 1, :, :, :][is_ES==True])
        
        _, dice_score_ES_seg_per_class = calculate_metrics(ES_seg, masks[:, 1, :, :, :][is_ES==True])
        dice_score_average2_per_class = (dice_score_ED_seg_per_class + dice_score_ES_seg_per_class) / 2
        dice_score_average3 = (dice_score_ED_seg * 1 + dice_score_ES_seg * 1 + dice_score_t_seg) / 3 if False in is_ES else (dice_score_ED_seg * 1 + dice_score_ES_seg * 1)/2
        dice_score_average2 = (dice_score_ED_seg * 1 + dice_score_ES_seg) / 2
    else:
        #dice_score_t_seg = dice_coefficient(t_seg, masks[:, 1, :, :, :])
        dice_score_average2_per_class = dice_score_ED_seg_per_class / 1
        dice_score_average3 = (dice_score_ED_seg * 1 + dice_score_t_seg * 1) / 2
        dice_score_average2 = (dice_score_ED_seg * 1) / 1

    #---------------------------Hausdorff Distance------------------------------
    ED_seg = torch.argmax(ED_seg, dim=1)
    t_seg = torch.argmax(t_seg, dim=1)
    if True in is_ES:
        ES_seg = torch.argmax(ES_seg, dim=1)

    ED_seg_1 = (ED_seg == 1) * ED_seg
    ED_seg_2 = (ED_seg == 2) * ED_seg
    ED_seg_3 = (ED_seg == 3) * ED_seg

    hd_ED_seg1 = torch3D_Hausdorff_distance(ED_seg_1, (masks[:, 0, :, :, :] == 1) * masks[:, 0, :, :, :])
    hd_ED_seg2 = torch3D_Hausdorff_distance(ED_seg_2, (masks[:, 0, :, :, :] == 2) * masks[:, 0, :, :, :])
    hd_ED_seg3 = torch3D_Hausdorff_distance(ED_seg_3, (masks[:, 0, :, :, :] == 3) * masks[:, 0, :, :, :])
    hd_ED_seg = (hd_ED_seg1 + hd_ED_seg2 + hd_ED_seg3) / 3

    if True in is_ES:
        ES_mask = masks[is_ES == True][:, 1, :, :, :]
        hd_ES_seg1 = torch3D_Hausdorff_distance((ES_seg == 1) * ES_seg, (ES_mask == 1) * ES_mask)
        hd_ES_seg2 = torch3D_Hausdorff_distance((ES_seg == 2) * ES_seg, (ES_mask == 2) * ES_mask)
        hd_ES_seg3 = torch3D_Hausdorff_distance((ES_seg == 3) * ES_seg, (ES_mask == 3) * ES_mask)
    else:
        hd_ES_seg1, hd_ES_seg2, hd_ES_seg3 = 0, 0, 0
    hd_ES_seg = (hd_ES_seg1 + hd_ES_seg2 + hd_ES_seg3) / 3

    if False in is_ES:
        t_mask = masks[is_ES == False][:, 1, :, :, :]
        hd_t_seg1 = torch3D_Hausdorff_distance((t_seg == 1) * t_seg, (t_mask == 1) * t_mask)
        hd_t_seg2 = torch3D_Hausdorff_distance((t_seg == 2) * t_seg, (t_mask == 2) * t_mask)
        hd_t_seg3 = torch3D_Hausdorff_distance((t_seg == 3) * t_seg, (t_mask == 3) * t_mask)
        hd_t_seg = (hd_t_seg1 + hd_t_seg2 + hd_t_seg3) / 3

    if True in is_ES:
        hd_dis_average3 = (hd_ED_seg + hd_ES_seg) / 2
        hd_dis_average2 = hd_ED_seg / 1
    else:
        hd_dis_average3 = (hd_ED_seg + hd_t_seg) / 2
        hd_dis_average2 = hd_ED_seg / 1

    return dice_score_average3, hd_dis_average3, dice_score_average2, hd_dis_average2, dice_score_average2_per_class
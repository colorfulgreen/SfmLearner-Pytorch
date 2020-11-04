from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from inverse_warp import inverse_warp


def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics,
                                    depth, explainability_mask, pose,
                                    rotation_mode='euler', padding_mode='zeros'):
    '''
    tgt_img.size = torch.Size([4, 3, 128, 416])
    ref_imgs.len 2
    intrinsics.size = torch.Size([4, 3, 3])
    depth.len = 4
    explainability_mask.len = 4
    pose.size = torch.Size([4, 2, 6])       # 从 target 到两个 ref_imgs(sources) 的 6DoF 位姿变换
    '''

    def one_scale(depth, explainability_mask):
        assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))

        ##### 将 depth 图和原始图 scale 到同样大小的尺寸, 注意相机内参也需要随之 scale

        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area') # TODO mode='area'
        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)  # TODO intrinsics 为什么要这么排列
        '''
        (Pdb) intrinsics
        tensor([[[261.5696,   0.0000, 217.9755],
                 [  0.0000, 272.7059,  62.3303],
                 [  0.0000,   0.0000,   1.0000]],

                ...

                [[270.3754,   0.0000, 211.4148],
                 [  0.0000, 265.8237,  55.6816],
                 [  0.0000,   0.0000,   1.0000]]], device='cuda:0')

        (Pdb) intrinsics[:, 2:]
        tensor([[[0., 0., 1.]],

                ...

                [[0., 0., 1.]]], device='cuda:0')

        (Pdb) intrinsics[:, 0:2]
        tensor([[[261.5696,   0.0000, 217.9755],
                 [  0.0000, 272.7059,  62.3303]],

                ...

                [[270.3754,   0.0000, 211.4148],
                 [  0.0000, 265.8237,  55.6816]]], device='cuda:0')

        '''

        warped_imgs = []
        diff_maps = []

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped, valid_points = inverse_warp(ref_img, depth[:,0], current_pose,
                                                        intrinsics_scaled,
                                                        rotation_mode, padding_mode)
            diff = (tgt_img_scaled - ref_img_warped) * valid_points.unsqueeze(1).float()

            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i:i+1].expand_as(diff)

            reconstruction_loss += diff.abs().mean()
            # TODO 这里 assert 是为了检查什么
            assert((reconstruction_loss == reconstruction_loss).item() == 1)

            # TODO 一个 batch 里 4 张图片，这里为什么只取第一张
            warped_imgs.append(ref_img_warped[0])
            diff_maps.append(diff[0])

        return reconstruction_loss, warped_imgs, diff_maps

    warped_results, diff_results = [], []
    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    total_loss = 0
    for d, mask in zip(depth, explainability_mask):
        loss, warped, diff = one_scale(d, mask)
        total_loss += loss
        warped_results.append(warped)
        diff_results.append(diff)
    return total_loss, warped_results, diff_results


def explainability_loss(mask):
    '''
    Args:
        mask: batch_size 个 Tensor, 每个 Tensor 包含
    '''
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = torch.ones_like(mask_scaled)
        # TODO binary_cross_entropy
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss


def smooth_loss(pred_map):
    '''overcoming the gradient locality

    问题的具体描述：photometric reconstruction loss 仅依赖局部的双线性插值，当坐标位于 low-texture 区域时，即便预测错误，该部分 loss 仍为 0.
    为了在损失函数中体现这部分的错误，作者引入一个 smoothness loss 项，约束相邻位置的深度差.
    另一种解释是，引入 smoothness loss 可以借助高纹理区域的正确结果，帮助矫正低纹理区域.

    Args:
        pred_map: 4 个不同 scales 的深度图，对应 Fig4
    '''
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # don't ask me why it works better
    return loss


@torch.no_grad()
def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]

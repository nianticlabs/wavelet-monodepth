# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def depth_to_disp(depth, min_depth, max_depth):
    """Convert depth map back to disparity.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth

    disp = 1 / (depth+1e-5)
    disp = (disp - min_disp) / (max_disp - min_disp)
    disp[depth<=0] = 0
    disp[disp<=0] = 0

    return disp


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_layer=None, use_refl=False):
        super(ConvBlock, self).__init__()

        if kernel_size == 3:
            self.conv = Conv3x3(in_channels, out_channels, use_refl=use_refl)
        elif kernel_size == 1:
            self.conv = Conv1x1(in_channels, out_channels)
        else:
            raise NotImplementedError

        self.nonlin = nn.ELU(inplace=True)
        if norm_layer is not None:
            self.norm_layer = norm_layer(out_channels)
        else:
            self.norm_layer = nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm_layer(out)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True, stride=1, use_bias=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, stride=stride, bias=use_bias)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class Conv1x1(nn.Module):
    """Conv1x1
    """
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv(x)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode='nearest')


def get_smooth_loss(disp, img, gamma=2):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-gamma * grad_img_x)
    grad_disp_y *= torch.exp(-gamma * grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def get_grad_map(img, reduce=False):
    """
    Returns the gradient map of an image with two channels (x, y) per input channel
    :param img: input image
    :param reduce: average gradients over input channels

    Example:
    img.shape --> (B, 3, H, W)
    grad1 = get_grad_map(img, False)
    grad2 = get_grad_map(img, True)
    grad1.shape --> (B, 6, H, W)
    grad2.shape --> (B, 2, H, W)
    :return:
    """
    grad_img_x = F.pad(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), (0, 1, 0, 0), mode='reflect')
    grad_img_y = F.pad(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), (0, 0, 0, 1), mode='reflect')

    if reduce:
        grad_img_x = torch.mean(grad_img_x, 1, keepdim=True)
        grad_img_y = torch.mean(grad_img_y, 1, keepdim=True)

    grad = torch.cat([grad_img_x, grad_img_y], dim=1)

    return grad


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# Sparse convolutions

def sparse_select(xvals, xchn, xidxmap, ymask, ufactor=1, pad=False):
    xheight, xwidth = xidxmap.shape[2:]
    yheight, ywidth = ymask.shape[2:]
    assert xheight * ufactor == yheight
    assert xwidth * ufactor == ywidth
    numel = xvals.shape[0] // xchn

    coors = mask2yx(ymask)
    # yidx = coors[0] * ywidth + coors[1]
    if ufactor == 2:
        coors = coors // 2
    idx = coors[0] * xwidth + coors[1]

    if pad:
        xidxmap = xidxmap + 1
        numel = numel + 1
        xvals = torch.cat([torch.zeros(xchn, 1, dtype=xvals.dtype, device=xvals.device),
                           xvals.reshape(xchn, -1)], 1).reshape(-1)

    idx = xidxmap.reshape(-1).gather(index=idx, dim=0).reshape(1, -1).expand(xchn, -1)

    idx = torch.arange(xchn, dtype=torch.long, device=xvals.device).reshape(-1, 1) * numel + idx
    idx = idx.reshape(-1)

    vals = xvals.reshape(-1).gather(index=idx, dim=0)
    return vals


def make_result(xvals, xchn, mask):
    result = torch.zeros(1, xchn, mask.shape[2], mask.shape[3], dtype=xvals.dtype, device=xvals.device)
    result[mask.bool().expand(-1, xchn, -1, -1)] = xvals
    return result


def mask2yx(mask):
    assert mask.shape[0] == 1
    assert mask.shape[1] == 1
    height, width = mask.shape[2:]
    grid_y, grid_x = torch.meshgrid(torch.arange(height, dtype=torch.long, device=mask.device),
                                    torch.arange(width, dtype=torch.long, device=mask.device))
    y = grid_y.reshape(1, 1, height, width)[mask.bool()]
    x = grid_x.reshape(1, 1, height, width)[mask.bool()]
    return torch.stack([y, x], 0)


def mask2idxmap(xmask):
    assert xmask.shape[0] == 1
    assert xmask.shape[1] == 1
    numel = xmask.sum()
    xidxmap = -torch.ones(1, 1, xmask.shape[2], xmask.shape[3], device=xmask.device, dtype=torch.long)
    xidxmap[xmask.bool()] = torch.arange(numel, dtype=torch.long, device=xmask.device)
    ops = xmask.shape[2] * xmask.shape[3]
    return xidxmap, ops


def sparse_conv1x1(conv_layer, xvals, nonlin):
    if isinstance(conv_layer, Conv1x1):
        ochn, ichn = conv_layer.conv.weight.shape[0:2]
        w = conv_layer.conv.weight.reshape(ochn, -1)
        bias = conv_layer.conv.bias.reshape(ochn, 1)
    else:
        raise NotImplementedError()

    numel = xvals.reshape(-1).shape[0] // ichn

    vals = xvals.reshape(ichn, numel)
    new_vals = torch.matmul(w, vals) + bias
    new_vals = nonlin(new_vals)
    ops = numel * ichn * ochn + numel * ochn
    return new_vals, ochn, ops


def sparse_conv3x3(conv_layer, xvals, xidxmap,
                   mask, nonlin=nn.Identity(), padding='reflect', make_result=True):
    """
    Sparse 3x3 convolution operation
    """
    if isinstance(conv_layer, Conv3x3):
        ochn, ichn = conv_layer.conv.weight.shape[0:2]
        w = conv_layer.conv.weight.reshape(ochn, -1)
        bias = conv_layer.conv.bias.reshape(ochn, 1)
        ops = 0
    elif isinstance(conv_layer, ConvBlock):
        nonlin = conv_layer.nonlin
        conv_layer = conv_layer.conv
        ochn, ichn = conv_layer.conv.weight.shape[0:2]
        w = conv_layer.conv.weight.reshape(ochn, -1)
        bias = conv_layer.conv.bias.reshape(ochn, 1)
        ops = 0
    elif isinstance(conv_layer, nn.Sequential):
        if isinstance(conv_layer[0], Conv1x1):
            xvals, ichn, ops = sparse_conv1x1(conv_layer[0], xvals, conv_layer[1])
        ochn = conv_layer[2].conv.weight.shape[0]
        w = conv_layer[2].conv.weight.reshape(ochn, -1)
        bias = conv_layer[2].conv.bias.reshape(ochn, 1)
    else:
        raise NotImplementedError()

    height, width = mask.shape[2:]
    numel = xvals.reshape(-1).shape[0] // ichn

    # pad xvals
    xvals = torch.cat([torch.zeros(ichn, 1, dtype=xvals.dtype, device=xvals.device),
                       xvals.reshape(ichn, -1)], 1).reshape(-1)
    xidxmap = xidxmap + 1
    numel = numel + 1

    xidxmap = F.pad(xidxmap.float(), pad=(1, 1, 1, 1), mode=padding).long()
    pmask = F.pad(mask, pad=(2, 2, 2, 2), mode='constant', value=0) > 0.5

    idxs = []
    for i in range(9):
        dy = 2 - i // 3
        dx = 2 - i % 3
        smask = pmask[:, :, dy:(dy + height + 2), dx:(dx + width + 2)]
        idx = xidxmap[smask]
        idxs.append(idx)

    idx = torch.stack(idxs, 0)

    idx = idx.reshape(1, -1).expand(ichn, -1)
    idx = torch.arange(ichn, dtype=torch.long, device=xvals.device).reshape(-1, 1) * numel + idx
    idx = idx.reshape(-1)
    vals = xvals.reshape(-1).gather(index=idx, dim=0).reshape(ichn, -1)

    ops += vals.reshape(-1).shape[0]

    vals = vals.reshape(ichn * 3 * 3, -1)

    update = torch.matmul(w, vals)
    update = update + bias

    ops += (1 + 3 * 3 * ichn) * vals.shape[1] * ochn

    update = nonlin(update)

    if make_result:
        result = torch.zeros(1, ochn, height, width,
                             dtype=xvals.dtype, device=xvals.device)

        result[mask.bool().expand(-1, ochn, -1, -1)] = update.reshape(-1)
        return result, ops
    else:
        return update.reshape(-1), ochn, ops


def sparse_upsample(xvals, xchn, xidxmap, skip, mask, make_result=True):
    ochn = xchn + skip.shape[1]
    xheight = xidxmap.shape[2]
    xwidth = xidxmap.shape[3]
    oheight = 2 * xheight
    owidth = 2 * xwidth
    xnumel = xvals.shape[0] // xchn

    coors = mask2yx(mask)
    coors = coors // 2
    idx = coors[0] * xwidth + coors[1]
    idx = xidxmap.reshape(-1).gather(index=idx, dim=0)
    idx = idx.reshape(1, -1).expand(xchn, -1)
    idx = torch.arange(xchn, dtype=torch.long, device=xvals.device).reshape(-1, 1) * xnumel + idx
    idx = idx.reshape(-1)
    xvals = xvals.reshape(-1).gather(index=idx, dim=0)

    svals = skip[mask.bool().expand(-1, skip.shape[1], -1, -1)]
    vals = torch.cat([xvals, svals], 0)

    if make_result:
        result = torch.zeros(1, ochn, oheight, owidth, dtype=vals.dtype, device=vals.device)
        result[mask.bool().expand(-1, ochn, -1, -1)] = vals
        return result
    else:
        return vals, ochn
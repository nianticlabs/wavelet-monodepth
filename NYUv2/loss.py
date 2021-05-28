# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
from math import exp
import torch.nn.functional as F
import torch.nn as nn

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs

    return ret


class SpatialGradientsLoss(nn.Module):
    def __init__(self, kernel_size=3, size_average=False):
        super(SpatialGradientsLoss, self).__init__()

        self.size_average = size_average
        self.kernel_size = kernel_size

        if gradient_loss_on:
            self.masked_huber_loss = HuberLoss(sigma=3)

    def forward(self, input, target):

        repeat_channels = target.shape[1]

        sobel_x = torch.Tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])

        sobel_x = sobel_x.view((1, 1, 3, 3))
        sobel_x = torch.autograd.Variable(sobel_x.cuda())

        sobel_y = torch.Tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]])

        sobel_y = sobel_y.view((1, 1, 3, 3))
        sobel_y = torch.autograd.Variable(sobel_y.cuda())
        if repeat_channels != 1:
            sobel_x = sobel_x.repeat(1, repeat_channels, 1, 1)
            sobel_y = sobel_y.repeat(1, repeat_channels, 1, 1)

        gx_input = F.conv2d(input, (1.0 / 8.0) * sobel_x, padding=1)
        gy_input = F.conv2d(input, (1.0 / 8.0) * sobel_y, padding=1)

        gx_target = F.conv2d(target, (1.0 / 8.0) * sobel_x, padding=1)
        gy_target = F.conv2d(target, (1.0 / 8.0) * sobel_y, padding=1)

        gradients_input = torch.pow(gx_input, 2) + torch.pow(gy_input, 2)
        gradients_target = torch.pow(gx_target, 2) + torch.pow(gy_target, 2)

        grad_loss = self.masked_huber_loss(gradients_input, gradients_target, mask)

        return grad_loss


class LainaBerHuLoss(nn.Module):
    # Based on Laina et al.

    def __init__(self, size_average=True):
        super(LainaBerHuLoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        diff = input - target

        diff = torch.abs(diff)

        diff = diff.squeeze(1)
        c = 0.2 * diff.max()
        cond = diff < c
        loss = torch.where(cond, diff, (diff ** 2 + c ** 2) / (2 * c + 1e-9))

        return loss.mean()

class CroppedL1Loss(nn.Module):
    def __init__(self, size_average=True, crop_border=2):
        super(CroppedL1Loss, self).__init__()
        self.size_average = size_average
        self.crop_border = crop_border

    def forward(self, input, target):
        diff = torch.abs(input - target)
        mask = torch.zeros_like(target)
        _, _, H, W = target.shape
        mask[..., self.crop_border:H-self.crop_border, self.crop_border:W-self.crop_border] = 1

        return (diff * mask).mean()
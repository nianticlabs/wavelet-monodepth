# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, padding="zero", stride=1, is_depthwise=False):
        super(Conv3x3, self).__init__()

        if padding == "reflection":
            self.pad = nn.ReflectionPad2d(1)
        elif padding == "replicate":
            self.pad = nn.ReplicationPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        if is_depthwise:
            self.conv = nn.Sequential(depthwise(int(in_channels), kernel_size=3),
                                      pointwise(int(in_channels), int(out_channels)))
        else:
            self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3, stride=stride, padding=0)#padding_mode=padding)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    return F.interpolate(x, scale_factor=2, mode='nearest')


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features, align_corners=False, padding="zero"):
        super(UpSample, self).__init__()
        # self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.convA = Conv3x3(skip_input, output_features, padding=padding)
        self.leakyreluA = nn.LeakyReLU(0.2)
        # self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.convB = Conv3x3(output_features, output_features, padding=padding)
        self.leakyreluB = nn.LeakyReLU(0.2)

        self.align_corners = align_corners

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='nearest',
                             align_corners=self.align_corners)
        return self.leakyreluB( self.convB( self.leakyreluA(self.convA( torch.cat([up_x, concat_with], dim=1) ) ) )  )


class UpSampleBlock(nn.Sequential):
    def __init__(self, skip_input, output_features, padding="zero", is_depthwise=False):
        super(UpSampleBlock, self).__init__()
        self.convA = Conv3x3(skip_input, output_features, padding=padding, is_depthwise=is_depthwise)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, concat_with):
        # up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='nearest')
        up_x = self.upsample(x)
        return self.leakyreluA(self.convA( torch.cat([up_x, concat_with], dim=1) ) )


def depthwise(in_channels, kernel_size):
    padding = 0
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding, bias=False, groups=in_channels),
        nn.ReLU(inplace=True),
    )


def pointwise(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)


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

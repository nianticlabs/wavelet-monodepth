# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
from pytorch_wavelets import IDWT


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(
                    self.convs[("dispconv", i)](x)
                )

        return self.outputs


class DepthWaveProgressiveDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthWaveProgressiveDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])  # the last layer is removed since it is the full scale

        self.J = 1
        self.inverse_wt = IDWT(wave="haar", mode="zero")

        # decoder
        self.convs = OrderedDict()
        for i in range(4, 0, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]

            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out, use_refl=True)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, use_refl=True)

            # [0, 1, 2, 3] as scale options, will use scales [1, 2, 3, 4] to compute wavelet coefficients
            if i == 4:
                # LL
                self.convs[("waveconv", i, 0)] = nn.Sequential(*[Conv1x1(self.num_ch_dec[i], self.num_ch_dec[i] // 4),
                                                                 nn.LeakyReLU(0.1, inplace=True),
                                                                 Conv3x3(self.num_ch_dec[i] // 4, 1,
                                                                         use_refl=True)])  # low frequency

            self.convs[("waveconv", i, 1)] = nn.Sequential(*[Conv1x1(self.num_ch_dec[i], self.num_ch_dec[i]),
                                                             nn.LeakyReLU(0.1, inplace=True),
                                                             Conv3x3(self.num_ch_dec[i], 3,
                                                                     use_refl=True)])

            # split between positive and negative parts
            self.convs[("waveconv", i, -1)] = nn.Sequential(*[Conv1x1(self.num_ch_dec[i], self.num_ch_dec[i]),
                                                              nn.LeakyReLU(0.1, inplace=True),
                                                              Conv3x3(self.num_ch_dec[i], 3,
                                                                      use_refl=True)])

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def get_coefficients(self, input_features, scale=1, return_ll=False):
        """
        Takes features maps at scale s as input and returns tuple (LL, [LH, HL, HH])
        """

        yl = None
        if return_ll:
            yl = 2**scale * self.sigmoid(self.convs[("waveconv", scale, 0)](input_features))
        yh = 2**(scale-1) * self.sigmoid(self.convs[("waveconv", scale, 1)](input_features)).unsqueeze(1) - \
             2**(scale-1) * self.sigmoid(self.convs[("waveconv", scale, -1)](input_features)).unsqueeze(1)
        return yl, yh

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]

        for i in range(4, 0, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i == 4:
                # LL, (LH, HL, HH)
                yl, yh = self.get_coefficients(x, scale=i, return_ll=True)
            else:
                # compute only high coefficients (LH, HL, HH) and keep previous low LL
                _, yh = self.get_coefficients(x, scale=i, return_ll=False)

            # log coefficients
            self.outputs[("wavelets", i - 1, "LL")] = yl
            self.outputs[("wavelets", i - 1, "LH")] = yh[:, :, 0]
            self.outputs[("wavelets", i - 1, "HL")] = yh[:, :, 1]
            self.outputs[("wavelets", i - 1, "HH")] = yh[:, :, 2]

            yl = self.inverse_wt((yl, list([yh])))

            self.outputs[("disp", i - 1)] = torch.clamp(yl / 2**(i-1), 0, 1)

        return self.outputs


class SparseDepthWaveProgressiveDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(SparseDepthWaveProgressiveDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])  # the last layer is removed since it is the full scale

        self.J = 1
        self.inverse_wt = IDWT(wave="haar", mode="zero")

        # decoder
        self.convs = OrderedDict()
        for i in range(4, 0, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out, use_refl=True, kernel_size=3)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, use_refl=True,
                                                     kernel_size=3)
            # [0, 1, 2, 3] as scale options, will use scales [1, 2, 3, 4] to compute wavelet coefficients
            if i == 4:
                # LL
                self.convs[("waveconv", i, 0)] = nn.Sequential(
                    *[Conv1x1(self.num_ch_dec[i], self.num_ch_dec[i] // 4),
                      nn.LeakyReLU(0.1, inplace=True),
                      Conv3x3(self.num_ch_dec[i] // 4, 1,
                              use_refl=True)])  # low frequency

            self.convs[("waveconv", i, 1)] = nn.Sequential(*[Conv1x1(self.num_ch_dec[i], self.num_ch_dec[i]),
                                                             nn.LeakyReLU(0.1, inplace=True),
                                                             Conv3x3(self.num_ch_dec[i], 3,
                                                                     use_refl=True)])  # high-freq lvl1

            self.convs[("waveconv", i, -1)] = nn.Sequential(*[Conv1x1(self.num_ch_dec[i], self.num_ch_dec[i]),
                                                              nn.LeakyReLU(0.1, inplace=True),
                                                              Conv3x3(self.num_ch_dec[i], 3,
                                                                      use_refl=True)])  # high-freq lvl1

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.maxpool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(5, stride=1, padding=2)
        self.maxpool7 = nn.MaxPool2d(7, stride=1, padding=3)

    @staticmethod
    def my_iwt_once(coeffs):
        yl, [yh] = coeffs
        """ IWT once """
        recon = [None] * 4
        hf = yh / 2
        lf = yl / 2
        recon[0] = hf[:, :, 0] + hf[:, :, 1] + hf[:, :, 2]
        recon[2] = -hf[:, :, 0] + hf[:, :, 1] - hf[:, :, 2]
        recon[1] = hf[:, :, 0] - hf[:, :, 1] - hf[:, :, 2]
        recon[3] = -hf[:, :, 0] - hf[:, :, 1] + hf[:, :, 2]

        recon = lf.expand(-1, 4, -1, -1) + torch.cat(recon, 1)
        recon = F.pixel_shuffle(recon, 2)
        return recon


    def get_coefficients(self, input_features, mask, s=1, scale_pow=1, use_ll=False):
        """
        Takes features maps at scale s as input and returns tuple (LL, [LH, HL, HH])
        """
        total_ops = 0
        if use_ll:
            total_ops += (1 + 1 * 1 * self.convs[("waveconv", s, 0)][0].conv.weight.shape[1] * input_features.shape[
                2] * input_features.shape[3]) * \
                         self.convs[("waveconv", s, 0)][0].conv.weight.shape[0]
            total_ops += (
                                 1 + 3 * 3 * self.convs[("waveconv", s, 0)][2].conv.weight.shape[1] *
                                 input_features.shape[2] * input_features.shape[3]
                         ) * self.convs[("waveconv", s, 0)][2].conv.weight.shape[0]
            yl = self.sigmoid(self.convs[("waveconv", s, 0)](input_features))
            yl = 2 ** scale_pow * yl
        else:
            yl = None
        for j in [-1, 1]:
            total_ops += (1 + 1 * 1 * self.convs[("waveconv", s, j)][0].conv.weight.shape[1] *
                          input_features.shape[2] * input_features.shape[3]) * \
                         self.convs[("waveconv", s, j)][0].conv.weight.shape[0]
            total_ops += (
                                 1 + 3 * 3 * self.convs[("waveconv", s, j)][2].conv.weight.shape[1] *
                                 input_features.shape[2] * input_features.shape[3]
                         ) * self.convs[("waveconv", s, j)][2].conv.weight.shape[0]

        yh_1 = self.sigmoid(self.convs[("waveconv", s, 1)](input_features))
        yh_2 = self.sigmoid(self.convs[("waveconv", s, -1)](input_features))

        yh = 2 ** (scale_pow - 1) * (yh_1 - yh_2)
        yh = yh * mask

        return yl, yh.unsqueeze(1), total_ops

    def get_sparse_coefficients(self, xvals, xidxmap,
                                mask, s=1, scale_pow=1):
        """
        Takes features maps at scale s as input and returns tuple (LL, [LH, HL, HH])
        """
        total_ops = 0
        yh_1, ops = sparse_conv3x3(self.convs[("waveconv", s, 1)], xvals, xidxmap,
                                   mask, nonlin=self.sigmoid)
        total_ops += ops
        yh_2, ops = sparse_conv3x3(self.convs[("waveconv", s, -1)], xvals, xidxmap,
                                   mask, nonlin=self.sigmoid)
        total_ops += ops
        yh = 2 ** (scale_pow - 1) * (yh_1 - yh_2)

        return yh.unsqueeze(1), total_ops

    def forward(self, input_features, thresh_ratio=0.05, sparse_scales=[0, 1, 2, 3]):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        assert x.shape[0] == 1, 'works with single input only'

        total_ops = 0
        for i in range(4, -1, -1):
            # print(i)

            scale_ops = 0

            if i == 4:
                mask = torch.ones_like(x[:, 0:1])
            else:
                thresh = (yl.max() - yl.min()) * thresh_ratio
                mask = (torch.abs(yh).max(2)[0] > thresh).float()
                scale_ops += 3 * mask.shape[2] * mask.shape[3]

            # dilating masks
            umask = upsample(mask)
            wavelet_mask = umask.bool()

            lowres_mask = self.maxpool3(mask).bool()
            upconv0_mask = self.maxpool5(mask).bool()
            upsample_mask = self.maxpool5(umask).bool()
            upconv1_mask = self.maxpool3(umask).bool()

            # maxpool ops
            scale_ops += 5 * 5 * mask.shape[2] * mask.shape[3]
            scale_ops += 5 * 5 * 2 * 2 * mask.shape[2] * mask.shape[3]

            self.outputs[("lowres_mask", i - 1)] = lowres_mask.clone()
            self.outputs[("upconv0_mask", i - 1)] = upconv0_mask.clone()
            self.outputs[("upsample_mask", i - 1)] = upsample_mask.clone()
            self.outputs[("upconv1_mask", i - 1)] = upconv1_mask.clone()
            self.outputs[("wavelet_mask", i - 1)] = wavelet_mask.clone()

            if i in sparse_scales:  # do 1, 2 and 3 sparsely, others densely

                lowres_idxmap, ops = mask2idxmap(lowres_mask)
                scale_ops += ops
                upconv0_idxmap, ops = mask2idxmap(upconv0_mask)
                scale_ops += ops
                upsample_idxmap, ops = mask2idxmap(upsample_mask)
                scale_ops += ops
                upconv1_idxmap, ops = mask2idxmap(upconv1_mask)
                scale_ops += ops

                print('sparse:', i)
                assert self.use_skips and i > 0
                assert 'yl' in locals()

                if i == max(sparse_scales):
                    xvals = x[lowres_mask.bool().expand(-1, x.shape[1], -1, -1)]
                    xchn = x.shape[1]
                else:
                    xvals = sparse_select(xvals, xchn, prev_idxmap, lowres_mask, pad=True)

                xvals, xchn, ops = sparse_conv3x3(self.convs[("upconv", i, 0)], xvals, lowres_idxmap, upconv0_mask,
                                                  make_result=False)
                scale_ops += ops
                xvals, xchn = sparse_upsample(xvals, xchn, upconv0_idxmap, input_features[i - 1],
                                              upsample_mask, make_result=False)
                xvals, xchn, ops = sparse_conv3x3(self.convs[("upconv", i, 1)], xvals, upsample_idxmap, upconv1_mask,
                                                  make_result=False)
                scale_ops += ops

                # compute only high coefficients and keep previous low
                yh, ops = self.get_sparse_coefficients(xvals, upconv1_idxmap,
                                                       wavelet_mask, s=i, scale_pow=i)
                scale_ops += ops

                # log coefficients
                self.outputs[("wavelets", i - 1, "LL")] = yl
                self.outputs[("wavelets", i - 1, "LH")] = yh[:, :, 0]
                self.outputs[("wavelets", i - 1, "HL")] = yh[:, :, 1]
                self.outputs[("wavelets", i - 1, "HH")] = yh[:, :, 2]

                yl = self.inverse_wt((yl, list([yh])))
                ops = 4 * yl.shape[2] * yl.shape[3]  # DT bugfix: missed a factor of 4 here previously
                scale_ops += ops

                self.outputs[("disp", i - 1)] = torch.clamp(yl / 2 ** (i - 1), 0, 1)

                total_ops += scale_ops
                self.outputs[("total_ops", i - 1)] = scale_ops
                if i == 1:
                    break

                prev_idxmap = upconv1_idxmap
            else:  # dense operations
                # print('dense:', i)
                ops = (1 + 3 * 3 * x.shape[1] * x.shape[2] * x.shape[3]
                       ) * self.convs[("upconv", i, 0)].conv.conv.weight.shape[0]
                scale_ops += ops
                x = self.convs[("upconv", i, 0)](x)

                ux = [upsample(x)]
                if self.use_skips and i > 0:
                    ux += [input_features[i - 1]]

                ux = torch.cat(ux, 1)
                ops = (1 + 3 * 3 * ux.shape[1] * ux.shape[2] * ux.shape[3]
                       ) * self.convs[("upconv", i, 1)].conv.conv.weight.shape[0]
                scale_ops += ops

                ux = self.convs[("upconv", i, 1)](ux)

                if i == 4:
                    yl, yh, ops = self.get_coefficients(ux, wavelet_mask, s=i, scale_pow=i, use_ll=True)
                else:
                    # compute only high coefficients and keep previous low
                    _, yh, ops = self.get_coefficients(ux, wavelet_mask, s=i, scale_pow=i, use_ll=False)

                scale_ops += ops

                # log coefficients
                self.outputs[("wavelets", i - 1, "LL")] = yl
                self.outputs[("wavelets", i - 1, "LH")] = yh[:, :, 0]
                self.outputs[("wavelets", i - 1, "HL")] = yh[:, :, 1]
                self.outputs[("wavelets", i - 1, "HH")] = yh[:, :, 2]

                yl = self.inverse_wt((yl, list([yh])))
                ops = 4 * yl.shape[2] * yl.shape[3] #IDWT
                scale_ops += ops

                self.outputs[("disp", i - 1)] = torch.clamp(yl / 2 ** (i - 1), 0, 1)

                total_ops += scale_ops
                self.outputs[("total_ops", i - 1)] = scale_ops
                if i == 1:
                    break
                x = ux
        self.outputs["total_ops"] = total_ops
        return self.outputs
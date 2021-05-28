# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
import torch.nn as nn
from networks.layers import *
from pytorch_wavelets import IDWT

from collections import OrderedDict


class Decoder(nn.Module):
    def __init__(self, enc_features=[96, 96, 192, 384, 2208], decoder_width = 0.5, is_depthwise=False):
        super(Decoder, self).__init__()
        features = int(enc_features[-1] * decoder_width)

        padding = "zero"
        self.conv2 = Conv3x3(enc_features[-1], features, padding='zero')

        self.up1 = UpSampleBlock(skip_input=features//1 + enc_features[-2], output_features=features//2, padding=padding,
                                 is_depthwise=is_depthwise)
        self.up2 = UpSampleBlock(skip_input=features//2 + enc_features[-3], output_features=features//4, padding=padding,
                                 is_depthwise=is_depthwise)
        self.up3 = UpSampleBlock(skip_input=features//4 + enc_features[-4], output_features=features//8, padding=padding,
                                 is_depthwise=is_depthwise)
        self.up4 = UpSampleBlock(skip_input=features//8 + enc_features[-5], output_features=features//16, padding=padding,
                                 is_depthwise=is_depthwise)

        if not is_depthwise:
            self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        else:
            self.conv3 = Conv3x3(features // 16, 1, is_depthwise=is_depthwise)

    def forward(self, features):
        outputs = {}
        x_block0, x_block1, x_block2, x_block3, x_block4 = tuple(features)
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)

        outputs[("disp", 0)] = self.conv3(x_d4)
        return outputs


class Decoder224(nn.Module):
    def __init__(self, enc_features=[96, 96, 192, 384, 2208], decoder_width = 0.5, is_depthwise=False):
        super(Decoder224, self).__init__()
        features = int(enc_features[-1] * decoder_width)

        padding = "zero"
        # self.conv2 = nn.Conv2d(enc_features[-1], features, kernel_size=1, stride=1, padding=1, padding_mode='zeros')
        self.conv2 = Conv3x3(enc_features[-1], features, padding='zero')

        self.up1 = UpSampleBlock(skip_input=features//1 + enc_features[-2], output_features=features//2, padding=padding,
                                 is_depthwise=is_depthwise)
        self.up2 = UpSampleBlock(skip_input=features//2 + enc_features[-3], output_features=features//4, padding=padding,
                                 is_depthwise=is_depthwise)
        self.up3 = UpSampleBlock(skip_input=features//4 + enc_features[-4], output_features=features//8, padding=padding,
                                 is_depthwise=is_depthwise)
        self.up4 = UpSampleBlock(skip_input=features//8 + enc_features[-5], output_features=features//16, padding=padding,
                                 is_depthwise=is_depthwise)
        self.conv5 = nn.Sequential(*[Conv3x3(features//16, features//32, is_depthwise=is_depthwise),
                                     nn.LeakyReLU(0.2)])

        if not is_depthwise:
            self.conv3 = nn.Conv2d(features//32, 1, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        else:
            self.conv3 = Conv3x3(features // 32, 1, is_depthwise=is_depthwise)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, features):
        outputs = {}
        x_block0, x_block1, x_block2, x_block3, x_block4 = tuple(features)
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)

        x_d4 = self.upsample(x_d4)
        x_d5 = self.conv5(x_d4)
        outputs[("disp", 0)] = self.conv3(x_d5)
        return outputs


class DecoderWave(nn.Module):
    def __init__(self, enc_features=[96, 96, 192, 384, 2208], decoder_width=0.5, dw_waveconv=False, dw_upconv=False):
        super(DecoderWave, self).__init__()
        features = int(enc_features[-1] * decoder_width)

        wave_pad = "zero"
        padding = "reflection"
        self.iwt = IDWT(wave='haar', mode=wave_pad)

        self.iwt_LL = IDWT(wave='haar', mode='zero')
        self.conv2 = Conv3x3(enc_features[-1], features, padding='replicate')
        self.up1 = UpSampleBlock(skip_input=features // 1 + enc_features[-2], output_features=features // 2,
                                 padding=padding, is_depthwise=dw_upconv)

        self.wave1_ll = Conv3x3(features // 2, 1, padding='replicate')
        self.wave1 = Conv3x3(features // 2, 3, padding=wave_pad, is_depthwise=dw_waveconv)

        self.up2 = UpSampleBlock(skip_input=features // 2 + enc_features[-3], output_features=features // 4,
                                 padding=padding, is_depthwise=dw_upconv)
        self.wave2 = Conv3x3(features // 4, 3, padding=wave_pad, is_depthwise=dw_waveconv)

        self.up3 = UpSampleBlock(skip_input=features // 4 + enc_features[-4], output_features=features // 8,
                                 padding=padding, is_depthwise=dw_upconv)
        self.wave3 = Conv3x3(features // 8, 3, padding=wave_pad, is_depthwise=dw_waveconv)

    def forward(self, x_blocks):
        outputs = {}
        x_d0 = self.conv2(x_blocks[-1])

        x_d1 = self.up1(x_d0, x_blocks[-2])
        ll = (2 ** 3) * self.wave1_ll(x_d1)
        outputs[("disp", 3)] = ll / (2 ** 3)
        h = (2 ** 2) * self.wave1(x_d1).unsqueeze(1)
        outputs[("wavelets", 2, "LL")] = ll
        outputs[("wavelets", 2, "LH")] = h[:, :, 0]
        outputs[("wavelets", 2, "HL")] = h[:, :, 1]
        outputs[("wavelets", 2, "HH")] = h[:, :, 2]
        ll = self.iwt((ll, list([h])))
        outputs[("disp", 2)] = ll / (2 ** 2)

        x_d2 = self.up2(x_d1, x_blocks[-3])
        h = (2 ** 1) * self.wave2(x_d2).unsqueeze(1)
        outputs[("wavelets", 1, "LH")] = h[:, :, 0]
        outputs[("wavelets", 1, "HL")] = h[:, :, 1]
        outputs[("wavelets", 1, "HH")] = h[:, :, 2]
        ll = self.iwt((ll, list([h])))
        outputs[("disp", 1)] = ll / (2 ** 1)

        x_d3 = self.up3(x_d2, x_blocks[-4])
        h = self.wave3(x_d3).unsqueeze(1)
        outputs[("wavelets", 0, "LH")] = h[:, :, 0]
        outputs[("wavelets", 0, "HL")] = h[:, :, 1]
        outputs[("wavelets", 0, "HH")] = h[:, :, 2]
        ll = self.iwt((ll, list([h])))
        outputs[("disp", 0)] = ll

        return outputs


class DecoderWave224(nn.Module):
    def __init__(self, enc_features=[96, 96, 192, 384, 2208], decoder_width=0.5, dw_waveconv=False, dw_upconv=False):
        super(DecoderWave224, self).__init__()
        features = int(enc_features[-1] * decoder_width)

        wave_pad = "zero"
        padding = "reflection"
        self.iwt = IDWT(wave='haar', mode=wave_pad)

        self.iwt_LL = IDWT(wave='haar', mode='zero')
        self.conv2 = Conv3x3(enc_features[-1], features, padding='replicate')
        self.up1 = UpSampleBlock(skip_input=features // 1 + enc_features[-2], output_features=features // 2,
                                 padding=padding, is_depthwise=dw_upconv)

        self.wave1_ll = Conv3x3(features // 2, 1, padding='replicate')
        self.wave1 = Conv3x3(features // 2, 3, padding=wave_pad, is_depthwise=dw_waveconv)

        self.up2 = UpSampleBlock(skip_input=features // 2 + enc_features[-3], output_features=features // 4,
                                 padding=padding, is_depthwise=dw_upconv)
        self.wave2 = Conv3x3(features // 4, 3, padding=wave_pad, is_depthwise=dw_waveconv)

        self.up3 = UpSampleBlock(skip_input=features // 4 + enc_features[-4], output_features=features // 8,
                                 padding=padding, is_depthwise=dw_upconv)
        self.wave3 = Conv3x3(features // 8, 3, padding=wave_pad, is_depthwise=dw_waveconv)
        self.up4 = UpSampleBlock(skip_input=features // 8 + enc_features[-5], output_features=features // 16,
                                 padding=padding, is_depthwise=dw_upconv)
        self.wave4 = Conv3x3(features // 16, 3, padding=wave_pad, is_depthwise=dw_waveconv)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_blocks):
        outputs = {}
        x_d0 = self.conv2(x_blocks[-1])
        x_d1 = self.up1(x_d0, x_blocks[-2])

        ll = (2 ** 4) * (self.wave1_ll(x_d1))
        h = self.wave1(x_d1).unsqueeze(1)
        h = (2 ** 3) * h
        outputs[("wavelets", 3, "LL")] = ll
        outputs[("wavelets", 3, "LH")] = h[:, :, 0]
        outputs[("wavelets", 3, "HL")] = h[:, :, 1]
        outputs[("wavelets", 3, "HH")] = h[:, :, 2]
        ll = self.iwt((ll, list([h])))
        outputs[("disp", 3)] = ll / (2 ** 3)

        x_d2 = self.up2(x_d1, x_blocks[-3])
        h = self.wave2(x_d2).unsqueeze(1)
        h = (2 ** 2) * h
        outputs[("wavelets", 2, "LH")] = h[:, :, 0]
        outputs[("wavelets", 2, "HL")] = h[:, :, 1]
        outputs[("wavelets", 2, "HH")] = h[:, :, 2]
        ll = self.iwt((ll, list([h])))
        outputs[("disp", 2)] = ll / (2 ** 2)

        x_d3 = self.up3(x_d2, x_blocks[-4])
        h = self.wave3(x_d3).unsqueeze(1)
        h = (2 ** 1) * h
        outputs[("wavelets", 1, "LH")] = h[:, :, 0]
        outputs[("wavelets", 1, "HL")] = h[:, :, 1]
        outputs[("wavelets", 1, "HH")] = h[:, :, 2]
        ll = self.iwt((ll, list([h])))
        outputs[("disp", 1)] = ll // (2 ** 1)

        x_d4 = self.up4(x_d3, x_blocks[-5])
        h = self.wave4(x_d4).unsqueeze(1)
        outputs[("wavelets", 0, "LH")] = h[:, :, 0]
        outputs[("wavelets", 0, "HL")] = h[:, :, 1]
        outputs[("wavelets", 0, "HH")] = h[:, :, 2]
        ll = self.iwt((ll, list([h])))
        outputs[("disp", 0)] = ll
        return outputs


class SparseDecoderWave(nn.Module):
    def __init__(self, enc_features=[96, 96, 192, 384, 2208], decoder_width=0.5):
        super(SparseDecoderWave, self).__init__()

        print("Using Sparse DenseDepth Decoder")

        features = int(enc_features[-1] * decoder_width)

        wave_pad = "zero"
        padding = "reflection"
        self.iwt = IDWT(wave='haar', mode=wave_pad)

        self.iwt_LL = IDWT(wave='haar', mode='zero')
        self.conv2 = Conv3x3(enc_features[-1], features, padding='replicate')
        self.up1 = UpSampleBlock(skip_input=features // 1 + enc_features[-2], output_features=features // 2,
                                 padding=padding)

        self.wave1_ll = Conv3x3(features // 2, 1, padding='replicate')
        self.wave1 = Conv3x3(features // 2, 3, padding=wave_pad)

        self.up2 = UpSampleBlock(skip_input=features // 2 + enc_features[-3], output_features=features // 4,
                                 padding=padding)
        self.wave2 = Conv3x3(features // 4, 3, padding=wave_pad)

        self.up3 = UpSampleBlock(skip_input=features // 4 + enc_features[-4], output_features=features // 8,
                                 padding=padding)
        self.wave3 = Conv3x3(features // 8, 3, padding=wave_pad)

        if padding == 'reflection':
            self.sparse_padding = 'reflect'
        elif padding == 'replicate':
            self.sparse_padding = 'replicate'
        else:
            self.sparse_padding = 'constant'

        if wave_pad == 'reflection':
            self.sparse_wave_pad = 'reflect'
        elif wave_pad == 'replicate':
            self.sparse_wave_pad = 'replicate'
        else:
            self.sparse_wave_pad = 'constant'

        self.leakyreluA = nn.LeakyReLU(0.2)
        self.maxpool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.maxpool5 = nn.MaxPool2d(5, stride=1, padding=2)
        self.maxpool7 = nn.MaxPool2d(7, stride=1, padding=3)

    def forward(self, x_blocks, thresh_ratio=0.1):
        total_ops = 0

        outputs = {}

        ops = (1 + 3 * 3 * x_blocks[-1].shape[1]) * x_blocks[-1].shape[2] * x_blocks[-1].shape[3] * \
              self.conv2.conv.weight.shape[0]
        total_ops += ops

        x_d0 = self.conv2(x_blocks[-1])

        x_d1 = self.up1(x_d0, x_blocks[-2])

        chn = x_d0.shape[1] + x_blocks[-2].shape[1]
        ops = (1 + 3 * 3 * chn) * x_d1.shape[2] * x_d1.shape[3] * x_d1.shape[1]
        total_ops += ops

        ll = (2 ** 3) * self.wave1_ll(x_d1)

        # get threshold relative to LL
        # thresh = (ll.max() - ll.min()) / (2 ** 3) * thresh_ratio

        outputs[("disp", 3)] = ll / (2 ** 3)
        h = (2**2) * self.wave1(x_d1)

        chn = x_d1.shape[1]
        ops = (1 + 3 * 3 * chn) * x_d1.shape[2] * x_d1.shape[3] * 4
        total_ops += ops

        # h = self.pre_activation(h.unsqueeze(1))
        h = h.unsqueeze(1)
        mask = torch.ones_like(h[:, 0])
        outputs[('wavelet_mask', 2)] = mask

        outputs[("wavelets", 2, "LL")] = ll
        outputs[("wavelets", 2, "LH")] = h[:, :, 0]
        outputs[("wavelets", 2, "HL")] = h[:, :, 1]
        outputs[("wavelets", 2, "HH")] = h[:, :, 2]
        ll = self.iwt((ll, list([h])))

        total_ops += ll.shape[2] * ll.shape[3]
        outputs[("disp", 2)] = ll / (2 ** 2)

        ## start sparse
        # scale 2
        thresh = (ll.max() - ll.min()) * thresh_ratio
        mask = (torch.abs(h).max(2)[0] > thresh).float()
        total_ops += 3 * mask.shape[2] * mask.shape[3]
        up_mask = self.maxpool5(mask).bool()
        conva_mask = self.maxpool5(upsample(mask)).bool()
        wave_mask = self.maxpool3(upsample(mask)).bool()
        wavelet_mask = upsample(mask)
        # ops
        total_ops += 5 * 5 * mask.shape[2] * mask.shape[3]
        total_ops += 5 * 5 * 2 * 2 * mask.shape[2] * mask.shape[3]

        wavelet_idxmap, ops = mask2idxmap(wavelet_mask)
        total_ops += ops
        conva_idxmap, ops = mask2idxmap(conva_mask)
        total_ops += ops
        wave_idxmap, ops = mask2idxmap(wave_mask)
        total_ops += ops
        up_idxmap, ops = mask2idxmap(up_mask)
        total_ops += ops

        outputs[('wavelet_mask', 1)] = wavelet_mask

        xchn = x_d1.shape[1]
        xvals = x_d1[up_mask.bool().expand(-1, xchn, -1, -1)]

        #         x_d2 = self.up2(x_d1, x_blocks[-3])
        #         h = self.wave2(x_d2)
        xvals, xchn = sparse_upsample(xvals, xchn, up_idxmap, x_blocks[-3], conva_mask, make_result=False)
        xvals, xchn, ops = sparse_conv3x3(self.up2.convA, xvals, conva_idxmap,
                                          wave_mask, nonlin=self.leakyreluA,
                                          padding=self.sparse_padding, make_result=False)
        total_ops += ops
        h, ops = sparse_conv3x3(self.wave2, xvals, wave_idxmap,
                                wavelet_mask, nonlin=nn.Identity(), padding=self.sparse_wave_pad, make_result=True)
        total_ops += ops

        # h = self.pre_activation(h.unsqueeze(1))
        h = (2 ** 1) * h.unsqueeze(1)
        outputs[("wavelets", 1, "LH")] = h[:, :, 0]
        outputs[("wavelets", 1, "HL")] = h[:, :, 1]
        outputs[("wavelets", 1, "HH")] = h[:, :, 2]
        ll = self.iwt((ll, list([wavelet_mask.unsqueeze(2) * h])))
        total_ops += ll.shape[2] * ll.shape[3]
        outputs[("disp", 1)] = ll / (2 ** 1)

        # scale 1
        prev_idxmap = wave_idxmap
        thresh = (ll.max() - ll.min()) * thresh_ratio
        mask = (torch.abs(h).max(2)[0] > thresh).float()
        total_ops += 3 * mask.shape[2] * mask.shape[3]
        up_mask = self.maxpool5(mask).bool()
        conva_mask = self.maxpool5(upsample(mask)).bool()
        wave_mask = self.maxpool3(upsample(mask)).bool()
        wavelet_mask = upsample(mask)
        total_ops += 5 * 5 * mask.shape[2] * mask.shape[3]
        total_ops += 5 * 5 * 2 * 2 * mask.shape[2] * mask.shape[3]

        wavelet_idxmap, ops = mask2idxmap(wavelet_mask)
        total_ops += ops
        conva_idxmap, ops = mask2idxmap(conva_mask)
        total_ops += ops
        wave_idxmap, ops = mask2idxmap(wave_mask)
        total_ops += ops
        up_idxmap, ops = mask2idxmap(up_mask)
        total_ops += ops
        wave_idxmap, ops = mask2idxmap(wave_mask)
        total_ops += ops

        outputs[('wavelet_mask', 0)] = wavelet_mask

        xvals = sparse_select(xvals, xchn, prev_idxmap, up_mask, pad=True)
        #         x_d3 = self.up3(x_d2, x_blocks[-4])
        #         h = self.wave3(x_d3)
        xvals, xchn = sparse_upsample(xvals, xchn, up_idxmap, x_blocks[-4], conva_mask, make_result=False)

        xvals, xchn, ops = sparse_conv3x3(self.up3.convA, xvals, conva_idxmap,
                                          wave_mask, nonlin=self.leakyreluA,
                                          padding=self.sparse_padding, make_result=False)
        total_ops += ops
        h, ops = sparse_conv3x3(self.wave3, xvals, wave_idxmap,
                                wavelet_mask, nonlin=nn.Identity(), padding=self.sparse_wave_pad, make_result=True)
        total_ops += ops

        # h = self.pre_activation(h.unsqueeze(1))
        h = h.unsqueeze(1)
        outputs[("wavelets", 0, "LH")] = h[:, :, 0]
        outputs[("wavelets", 0, "HL")] = h[:, :, 1]
        outputs[("wavelets", 0, "HH")] = h[:, :, 2]
        ll = self.iwt((ll, list([wavelet_mask.unsqueeze(2) * h])))
        total_ops += ll.shape[2] * ll.shape[3]
        outputs[("disp", 0)] = ll / (2 ** 0)

        outputs['total_ops'] = total_ops
        return outputs
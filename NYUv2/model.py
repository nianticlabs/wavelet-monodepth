# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch.nn as nn

from networks.encoders import ResnetEncoder, DenseEncoder, MobileNetV2Encoder
from networks.decoders import DecoderWave, DecoderWave224, Decoder, Decoder224, SparseDecoderWave

class Model(nn.Module):
    def __init__(self, opts):
        super(Model, self).__init__()

        print("Building model ", end="")

        decoder_width = 0.5
        if opts.encoder_type == "densenet":
            self.encoder = DenseEncoder(normalize_input=opts.normalize_input, pretrained=opts.pretrained_encoder)
        elif opts.encoder_type == "resnet":
            self.encoder = ResnetEncoder(num_layers=opts.num_layers, pretrained=opts.pretrained_encoder,
                                         normalize_input=opts.normalize_input)
        elif opts.encoder_type == "mobilenet":
            self.encoder = MobileNetV2Encoder(pretrained=opts.pretrained_encoder, use_last_layer=True,
                                              normalize_input=opts.normalize_input)
        elif opts.encoder_type == "mobilenet_light":
            self.encoder = MobileNetV2Encoder(pretrained=opts.pretrained_encoder, use_last_layer=False,
                                              normalize_input=opts.normalize_input)
        else:
            raise NotImplementedError

        print("using {} encoder".format(opts.encoder_type))

        self.use_sparse = False

        if opts.use_wavelets:
            try:
                if opts.use_sparse:
                    self.use_sparse = True
                    if opts.use_224:
                        raise NotImplementedError
            except AttributeError:
                opts.use_sparse = False
                self.use_sparse = False

            if opts.use_sparse:
                self.decoder = SparseDecoderWave(enc_features=self.encoder.num_ch_enc, decoder_width=decoder_width)
            else:
                if opts.use_224:
                    decoder_wave = DecoderWave224
                else:
                    decoder_wave = DecoderWave

                self.decoder = decoder_wave(enc_features=self.encoder.num_ch_enc, decoder_width=decoder_width,
                                            dw_waveconv=opts.dw_waveconv,
                                            dw_upconv=opts.dw_upconv)
        else:
            if opts.use_224:
                decoder = Decoder224
            else:
                decoder = Decoder
            self.decoder = decoder(enc_features=self.encoder.num_ch_enc,
                                   is_depthwise=(opts.dw_waveconv or opts.dw_upconv))

    def forward(self, x, threshold=-1):
        x = self.encoder(x)
        if self.use_sparse:
            return self.decoder(x, threshold)
        else:
            return self.decoder(x)

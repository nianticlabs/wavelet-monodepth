# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import networks.encoders as encoders
import networks.decoders as decoders
from networks.pose_cnn import PoseCNN


def make_depth_encoder(opts):
    print(" Building {}... \t".format(opts.encoder_type), end="")
    use_pretrained = opts.weights_init == "pretrained"
    if use_pretrained:
        print(" Loading pretrained weights... \t", end="")
    if opts.encoder_type == "resnet":
        encoder = encoders.ResnetEncoder(opts.num_layers, pretrained=use_pretrained)
    elif opts.encoder_type == "mobilenet":
        encoder = encoders.MobileNetV2Encoder(pretrained=use_pretrained,
                                              use_last_layer=True)
    elif opts.encoder_type == "mobilenet_light":
        encoder = encoders.MobileNetV2Encoder(pretrained=use_pretrained,
                                              use_last_layer=False)
    else:
        raise NotImplementedError
    return encoder


def make_depth_decoder(encoder, opts):
    print("Building Decoder... ", end="")
    if opts.use_wavelets:
        if opts.use_sparse:
            decoder = decoders.SparseDepthWaveProgressiveDecoder(encoder.num_ch_enc)
        else:
            decoder = decoders.DepthWaveProgressiveDecoder(
                encoder.num_ch_enc, opts.scales)
    else:
        decoder = decoders.DepthDecoder(encoder.num_ch_enc, opts.scales)
    return decoder


def make_posenet(opts, depth_encoder, num_pose_frames, num_input_frames):
    pose_encoder = None
    if opts.pose_model_type == "separate_resnet":
        pose_encoder = encoders.ResnetEncoder(
            opts.num_layers,
            opts.weights_init == "pretrained",
            num_input_images=num_pose_frames)

        pose_decoder = decoders.PoseDecoder(
            pose_encoder.num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)

    elif opts.pose_model_type == "shared":
        pose_decoder = decoders.PoseDecoder(
            depth_encoder.num_ch_enc, num_pose_frames)

    elif opts.pose_model_type == "posecnn":
        pose_decoder = PoseCNN(
            num_input_frames if opts.pose_model_input == "all" else 2)

    return pose_encoder, pose_decoder
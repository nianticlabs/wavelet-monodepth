# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import pickle as pkl
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

from networks.network_constructors import *

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

def compute_density(sparse_outputs):
    numerator = 0
    denominator = 0
    for i in range(4):
        try:
            numerator += torch.sum(sparse_outputs[("wavelet_mask", i)], (1,2,3))
            _, _, height, width = sparse_outputs[("wavelet_mask", i)].shape
            denominator += (height * width)
        except KeyError:
            pass
    return float(numerator) / float(denominator)


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           opt.height, opt.width,
                                           [0], opt.scales, is_train=False)
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        # Build network
        print("Building network")
        print("Encoder...", end='')
        encoder = make_depth_encoder(opt)
        encoder.cuda()
        encoder.eval()
        print("\t Done.")

        print("Decoder...", end='')
        depth_decoder = make_depth_decoder(encoder, opt)
        depth_decoder.cuda()
        depth_decoder.eval()
        print("\t Done.")

        sparse_decoding = (opt.use_sparse and opt.use_wavelets)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path), strict=False)

        N = len(dataset)

        pred_disps = np.zeros((N, opt.height, opt.width))
        print("pred_disps", pred_disps.shape)
        input_images = np.zeros((N, opt.height, opt.width, 3))

        if opt.use_wavelets:
            coeff_names = {"LL": 0,
                           "LH": 1,
                           "HL": 2,
                           "HH": 3,
                           }
            pred_coeffs = {}
            for i in range(4):
                pred_coeffs[i] = np.zeros((len(dataset), 4, opt.height//(2**(i+1)), opt.width//(2**(i+1))))

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        i = 0
        start_idx = 0

        total_ops = []
        densities = []

        with torch.no_grad():

            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                end_idx = input_color.shape[0] + start_idx

                print("Computing pred {}/{}".format(i+1, len(dataloader)))

                if opt.post_process and not sparse_decoding:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                if opt.save_pred_disps:
                    input_images[start_idx:end_idx] = input_color.cpu().numpy().transpose(0, 2, 3, 1)

                if sparse_decoding:
                    # sparse_decoding only implemented for batch_size=1
                    output = encoder(input_color)
                    output = depth_decoder(output, opt.threshold)
                    pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)

                    total_ops.append(output['total_ops'])
                    densities.append(compute_density(output))

                    # flipped version
                    output = encoder(torch.flip(input_color, [3]))
                    output = depth_decoder(output, opt.threshold)
                    pred_disp_flip, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                    pred_disp = torch.cat((pred_disp, pred_disp_flip), 0)

                    total_ops.append(output['total_ops'])

                else:
                    output = encoder(input_color)
                    output = depth_decoder(output)
                    pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)

                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.use_wavelets and opt.save_pred_disps:
                    for scale in range(4):
                        for c in ["LL", "LH", "HL", "HH"]:
                            pred_coeffs[scale][start_idx:end_idx, coeff_names[c]] = output[("wavelets", scale, c)].cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps[start_idx:end_idx] = pred_disp
                start_idx = end_idx
                i += 1

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

        output_path = os.path.join(
            opt.load_weights_folder, "inputs_rgb_{}_split.npy".format(opt.eval_split))
        np.save(output_path, input_images)

        if opt.use_wavelets:
            for scale in range(4):
                output_path = os.path.join(
                    opt.load_weights_folder, "disps_coeffs_s{}_{}_split.npy".format(scale, opt.eval_split))
                np.save(output_path, pred_coeffs[scale])

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.4f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

    if sparse_decoding:
        total_ops = np.array(total_ops)
        print("total_ops: mean {: 2.3f}, std {:2.3f}".format(np.mean(total_ops)/1e9, np.std(total_ops))/1e9)
        print("density: mean {:.3f}, std {:.3f}".format(100*np.mean(densities), 100*np.std(densities)))

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())

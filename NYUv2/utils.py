# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import matplotlib
import matplotlib.cm
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch
from imageio import imsave, imread
from skimage import feature

import pickle as pkl

import scipy.io as io
from scipy import ndimage
import os


def DepthNorm(depth, maxDepth=1000.0/100.0):
    return maxDepth / depth


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def scale_up_stack(scale, images):
    from skimage.transform import resize
    scaled = []

    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append(resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True))

    return np.stack(scaled)


def scale_up(scale, image):
    from skimage.transform import resize
    output_shape = (scale * image.shape[0], scale * image.shape[1])
    return resize(image, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True)


def colorize(value, vmin=0.1, vmax=10, cmap='plasma'):
    value = value.cpu().numpy()[0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2,0,1))


def compute_errors_nyu(pred, gt):
    # x = pred[crop]
    # y = gt[crop]
    y = gt
    x = pred
    thresh = torch.max((y / x), (x / y))
    a1 = (thresh < 1.25   ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    abs_rel = torch.mean(torch.abs(y - x) / y)
    rmse = (y - x) ** 2
    rmse = torch.sqrt(rmse.mean())
    log_10 = (torch.abs(torch.log10(y)-torch.log10(x))).mean()
    return abs_rel, rmse, log_10, a1, a2, a3


def compute_errors_kitti(gt, pred):
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


def compute_depth_boundary_error(edges_gt, pred, mask=None, low_thresh=0.15, high_thresh=0.3):
    # skip dbe if there is no ground truth distinct edge
    if np.sum(edges_gt) == 0:
        dbe_acc = np.nan
        dbe_com = np.nan
        edges_est = np.empty(pred.shape).astype(int)
    else:

        # normalize est depth map from 0 to 1
        pred_normalized = pred.copy().astype('f')
        pred_normalized[pred_normalized == 0] = np.nan
        pred_normalized = pred_normalized - np.nanmin(pred_normalized)
        pred_normalized = pred_normalized / np.nanmax(pred_normalized)

        # apply canny filter
        edges_est = feature.canny(pred_normalized, sigma=np.sqrt(2), low_threshold=low_thresh,
                                  high_threshold=high_thresh)

        # compute distance transform for chamfer metric
        D_gt = ndimage.distance_transform_edt(1 - edges_gt)
        D_est = ndimage.distance_transform_edt(1 - edges_est)

        max_dist_thr = 10.  # Threshold for local neighborhood

        mask_D_gt = D_gt < max_dist_thr  # truncate distance transform map

        E_fin_est_filt = edges_est * mask_D_gt  # compute shortest distance for all predicted edges
        if mask is None:
            mask = np.ones(shape=E_fin_est_filt.shape)
        E_fin_est_filt = E_fin_est_filt * mask
        D_gt = D_gt * mask

        if np.sum(E_fin_est_filt) == 0:  # assign MAX value if no edges could be detected in prediction
            dbe_acc = max_dist_thr
            dbe_com = max_dist_thr
        else:
            # accuracy: directed chamfer distance of predicted edges towards gt edges
            dbe_acc = np.nansum(D_gt * E_fin_est_filt) / np.nansum(E_fin_est_filt)

            # completeness: sum of undirected chamfer distances of predicted and gt edges
            ch1 = D_gt * edges_est  # dist(predicted,gt)
            ch1[ch1 > max_dist_thr] = max_dist_thr  # truncate distances
            ch2 = D_est * edges_gt  # dist(gt, predicted)
            ch2[ch2 > max_dist_thr] = max_dist_thr  # truncate distances
            res = ch1 + ch2  # summed distances
            dbe_com = np.nansum(res) / (np.nansum(edges_est) + np.nansum(edges_gt))  # normalized

    return dbe_acc, dbe_com, edges_est, D_est


def idx2nyu(idx, splits_idx):
    return int(splits_idx[idx])-1


def torch_fliplr(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def add_results(model, results_dict, rgb, depth, crop,
                start_idx, end_idx, border_crop_size=16, save_wavelets_dir=None, use_disparity=False,
                use_sparse=False, threshold=-1,
                edges_gt=None, use_224=False, nyu_idx_list=None):
    outdict = results_dict
    rgbs = rgb[start_idx:end_idx, :, :, :]

    replicate = nn.ReplicationPad2d(border_crop_size//2)

    # crop the image border
    x = rgbs[:, border_crop_size:-border_crop_size, border_crop_size:-border_crop_size, :]
    x = x.transpose(0, 3, 1, 2)

    x = torch.from_numpy(x).float().cuda() / 255

    # resize
    if use_224:
        x = F.interpolate(x, (224, 224), mode='bilinear', align_corners=True)
    else:
        x = F.interpolate(x, (480, 640), mode='bilinear', align_corners=True)

    # Compute results
    true_y = depth[start_idx:end_idx]

    # Compute predictions
    with torch.no_grad():
        model.eval()
        if use_sparse:
            outputs = model(x, threshold)
        else:
            outputs = model(x)
        pred_y = outputs[("disp", 0)]

        if use_disparity:
            pred_y = DepthNorm(pred_y, maxDepth=1000) / 10000
        else:
            pred_y /= 100

        # shrink back to the original scale before initial crop
        if not use_224:
            pred_y = F.interpolate(pred_y, (240 - border_crop_size, 320 - border_crop_size), mode='bilinear', align_corners=True)
            # pad borders
            pred_y = replicate(pred_y)
            # scale back to full scale
            pred_y = F.interpolate(pred_y, scale_factor=2, mode='bilinear', align_corners=True)

        pred_y = torch.clamp(pred_y, min=0.4, max=10)

    if save_wavelets_dir is not None:
        for j in range(start_idx, end_idx):
            save_outputs = {}
            if nyu_idx_list is None:
                save_idx = j
            else:
                save_idx = nyu_idx_list[j]
            with torch.no_grad():
                save_outputs[("wavelets", 2, "LL")] = outputs[("wavelets", 2, "LL")][j-start_idx].cpu().numpy()
                save_outputs[("disp", 0)] = outputs[("disp", 0)][j - start_idx].cpu().numpy()
                for scale in range(3):
                    for c in ["LH", "HL", "HH"]:
                        save_outputs[("wavelets", scale, c)] = outputs[("wavelets", scale, c)][j-start_idx].cpu()[0].numpy()

            output_path = os.path.join(save_wavelets_dir,
                                       "results_{}.pickle".format(save_idx))
            with open(output_path, "wb") as f:
                pkl.dump(save_outputs, f)

    if not use_224:
        # Crop based on Eigen et al. crop
        true_y = true_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
        pred_y = pred_y[:, 0, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
        outdict["depth_gts"].append(torch.from_numpy(true_y).float().cuda())
    else:
        outdict["depth_gts"].append(true_y.cuda())
    outdict["predictions"].append(pred_y)

    if edges_gt is not None:
        eigen_crop = np.zeros((480, 640), dtype=np.uint8)
        eigen_crop[crop[0]:crop[1] + 1, crop[2]:crop[3] + 1] = 1
        for idx in range(start_idx, end_idx):
            gt_edge = edges_gt[idx]
            with torch.no_grad():
                dbe_acc, dbe_comp, _, _ = compute_depth_boundary_error(gt_edge[crop[0]:crop[1] + 1, crop[2]:crop[3] + 1],
                                                                       pred_y[idx-start_idx].cpu().numpy())

            num_scores = outdict["edges_scores"].shape[0]
            outdict["edges_scores"][idx % num_scores, 0] = dbe_acc
            outdict["edges_scores"][idx % num_scores, 1] = dbe_comp

    return outdict


def evaluate(model, rgb, depth, crop, edges=None, verbose=False,
             use_disparity=False,
             save_npy=False, save_figs=False, save_wavelets=False,
             save_dir=None, use_224=False,
             use_sparse=False, threshold=-1,
             nyu_idx_list=None):
    N = len(rgb)

    results_dict = {"predictions": [], "depth_gts": []}

    border_crop_size = 16
    input_shape = (480, 640)

    if use_224:
        depth = depth[:, border_crop_size:-border_crop_size, border_crop_size:-border_crop_size]
        depth = torch.from_numpy(depth).float().cuda().unsqueeze(1)
        depth = F.interpolate(depth, (224, 224), mode='bilinear', align_corners=True)

    num_batches = N
    remainder = N % 1

    if save_wavelets:
        wavelets_savedir = os.path.join(save_dir, "results_wavelets")
        if not os.path.exists(wavelets_savedir):
            os.makedirs(wavelets_savedir)
    else:
        wavelets_savedir = None

    if edges is not None:
        results_dict["edges_scores"] = np.zeros((N, 2), dtype=np.float)

    model = model.eval()
    with torch.no_grad():
        for i in range(num_batches):
            print("Computing batch {}/{}".format(i, num_batches))
            start_idx = i
            end_idx = (i+1)

            results_dict = add_results(model, results_dict, rgb, depth, crop,
                                       start_idx, end_idx, border_crop_size=border_crop_size,
                                       save_wavelets_dir=wavelets_savedir, edges_gt=edges,
                                       use_disparity=use_disparity,
                                       use_sparse=use_sparse, threshold=threshold,
                                       use_224=use_224, nyu_idx_list=nyu_idx_list)

        predictions = torch.cat(results_dict["predictions"], dim=0)
        testSetDepths = torch.cat(results_dict["depth_gts"], dim=0)

        remaining_results = {"predictions": [], "depth_gts":[]}
        if N != end_idx:
            if edges is not None:
                remaining_results["edges_scores"] = np.zeros((N-end_idx, 2), dtype=np.float)
            remaining_results = add_results(model, remaining_results, rgb, depth, crop,
                                            end_idx, N, border_crop_size=border_crop_size,
                                            save_wavelets_dir=wavelets_savedir, edges_gt=edges,
                                            use_disparity=use_disparity, use_224=use_224, nyu_idx_list=nyu_idx_list)

            predictions = torch.cat([predictions, torch.cat(remaining_results["predictions"], dim=0)], dim=0)
            testSetDepths = torch.cat([testSetDepths, torch.cat(remaining_results["depth_gts"], axis=0)], axis=0)

    e = compute_errors_nyu(predictions, testSetDepths)
    if edges is not None:
        edges_scores = results_dict["edges_scores"]
        if N!= end_idx:
            edges_scores = np.vstack((results_dict["edges_scores"],
                                      remaining_results["edges_scores"]))
        e_edges = edges_scores.mean(0)
    else:
        e_edges = None

    if save_npy:
        if not os.path.exists(os.path.join(save_dir, "results_npy")):
            os.makedirs(os.path.join(save_dir, "results_npy"))
        np.save(os.path.join(save_dir, "results_npy", "eigen_rgb.npy"), rgb)
        np.save(os.path.join(save_dir, "results_npy", "eigen_preds.npy"), predictions.cpu().numpy())
        np.save(os.path.join(save_dir, "results_npy", "eigen_gts.npy"), testSetDepths.cpu().numpy())

    if save_figs:
        if not os.path.exists(os.path.join(save_dir, "results")):
            os.makedirs(os.path.join(save_dir, "results"))

        print("saving {} files".format(predictions.shape[0]))
        for i in range(predictions.shape[0]):
            save_idx = i if nyu_idx_list is None else nyu_idx_list[i]
            imsave(os.path.join(save_dir, "results", "{}_pred.png".format(save_idx)),
                   colorize(predictions[i].unsqueeze(0)).transpose(1,2,0))
            imsave(os.path.join(save_dir, "results", "{}_gt.png".format(save_idx)),
                   colorize(testSetDepths[i].unsqueeze(0)).transpose(1, 2, 0))
            imsave(os.path.join(save_dir, "results", "{}_rgb.png".format(i)), rgb[save_idx])

    if verbose:
        if edges is not None:
            print(("&{:>10}  " * 8).format('rel', 'rms', 'log_10', 'a1', 'a2', 'a3', 'e_acc', 'e_comp'))
            print(("&{:10.4f}  " * 8).format(e[0], e[1], e[2], e[3], e[4], e[5], e_edges[0], e_edges[1]))
        else:
            print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rms', 'log_10', 'a1', 'a2', 'a3'))
            print(("&{:10.4f}  " * 6).format(e[0], e[1], e[2], e[3], e[4], e[5]))
    return e, e_edges


def to_multichannel(i):
    if i.shape[2] == 3: return i
    i = i[:,:,0]
    return np.stack((i,i,i), axis=2)


def display_images(outputs, inputs=None, gt=None, is_colormap=True, is_rescale=True):
    import matplotlib.pyplot as plt
    import skimage
    from skimage.transform import resize

    plasma = plt.get_cmap('plasma')

    shape = (outputs[0].shape[0], outputs[0].shape[1], 3)

    all_images = []

    for i in range(outputs.shape[0]):
        imgs = []

        if isinstance(inputs, (list, tuple, np.ndarray)):
            x = to_multichannel(inputs[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True)
            imgs.append(x)

        if isinstance(gt, (list, tuple, np.ndarray)):
            x = to_multichannel(gt[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True)
            imgs.append(x)

        if is_colormap:
            rescaled = outputs[i][:, :, 0]
            if is_rescale:
                rescaled = rescaled - np.min(rescaled)
                rescaled = rescaled / np.max(rescaled)
            imgs.append(plasma(rescaled)[:, :, :3])
        else:
            imgs.append(to_multichannel(outputs[i]))

        img_set = np.hstack(imgs)
        all_images.append(img_set)

    all_images = np.stack(all_images)

    return skimage.util.montage(all_images, multichannel=True, fill=(0, 0, 0))

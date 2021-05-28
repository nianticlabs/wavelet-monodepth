# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import time
import argparse
import datetime

import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from model import Model
from data import getTrainingTestingData
from utils import AverageMeter, DepthNorm

from load_save_utils import *

from pytorch_wavelets import DWT

import os
import sys
import shutil

def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

def set_train(model):
    """Convert all models to training mode
    """
    for m in model.modules():
        m.train()

def set_eval(model):
    """Convert all models to testing/evaluation mode
    """
    for m in model.modules():
        m.eval()


def val(model, val_iter, val_loader, dwt, l1_criterion, all_writers, niter, args, LL_criterion):
    """Validate the model on a single minibatch
    """
    set_eval(model)
    try:
        sample_batched = val_iter.next()
    except StopIteration:
        val_iter = iter(val_loader)
        sample_batched = val_iter.next()

    with torch.no_grad():
        inputs = {}
        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
        if args.disparity:
            depth_n = DepthNorm(depth)
        else:
            depth_n = depth
        inputs["image"] = image.detach().cpu()  # .data

        inputs["disp_gt"] = depth_n
        yl_gt, yh_gt = dwt(depth_n)

        outputs = model(image)
        losses = {}
        total_loss = 0

        if args.use_wavelets:
            pad_lowestscale = nn.Identity()
            inputs[("wavelets", 3, "LL")] = pad_lowestscale(yl_gt)

        for scale in range(4):
            if args.use_wavelets:
                inputs[("wavelets", scale, "LH")] = yh_gt[scale][:, 0, 0]
                inputs[("wavelets", scale, "HL")] = yh_gt[scale][:, 0, 1]
                inputs[("wavelets", scale, "HH")] = yh_gt[scale][:, 0, 2]

            if scale in args.output_scales:
                pred = F.interpolate(outputs[("disp", scale)],
                                     scale_factor=2 ** scale, mode='bilinear', align_corners=True)
                l_depth = l1_criterion(pred, depth_n)

                loss = (0.1 * l_depth)
                if scale in args.loss_scales:
                    total_loss += loss
                losses["loss/{}".format(scale)] = loss
                losses["loss_depth/{}".format(scale)] = l_depth

        if args.use_wavelets:
            try:
                l_LL = LL_criterion(outputs[("wavelets", 3, "LL")], yl_gt) / 2**4
                losses["loss_LL3"] = l_LL
                if args.supervise_LL:
                    total_loss += l_LL
            except KeyError:
                pass
        losses["loss"] = total_loss

        log("val", inputs, outputs, losses, all_writers, niter, args)
        del inputs, outputs, losses

    set_train(model)
    return val_iter


def log(mode, inputs, outputs, losses, all_writers, niter, args):
    writer = all_writers[mode]
    for l, v in losses.items():
        writer.add_scalar("{}".format(l), v, niter)

    for j in range(min(args.bs, 4)):
        writer.add_image(
            "color/{}".format(j), inputs["image"][j], niter)

        if args.use_wavelets:
            # log LL
            try:
                writer.add_image(
                    "{}_{}_pred/{}".format("LL", 3, j),
                    normalize_image(outputs[("wavelets", 3, "LL")][j]), niter)
                if args.log_histogram:
                    writer.add_histogram("hist_{}_{}_pred/{}".format("LL", 3, j),
                                         outputs[("wavelets", 3, "LL")][j], niter)
                writer.add_image(
                    "{}_{}_gt/{}".format("LL", 3, j),
                    normalize_image(inputs[("wavelets", 3, "LL")][j]), niter)
                if args.log_histogram:
                    writer.add_histogram("hist_{}_{}_gt/{}".format("LL", 3, j),
                                         inputs[("wavelets", 3, "LL")][j], niter)
            except KeyError:
                pass

        writer.add_image(
            "disp_0_gt/{}".format(j),
            normalize_image(inputs["disp_gt"][j]), niter)
        for scale in range(4):
            if scale in args.output_scales:
                writer.add_image(
                    "disp_{}_pred/{}".format(scale, j),
                    normalize_image(outputs[("disp", scale)][j]), niter)
            if args.use_wavelets:
                for c, coeff in enumerate(["LH", "HL", "HH"]):
                    if ("wavelets", scale, coeff) in outputs:
                        # print("wavelets", scale, coeff)
                        writer.add_image(
                            "{}_{}_pred/{}".format(coeff, scale, j),
                            normalize_image(outputs[("wavelets", scale, coeff)][j]), niter)
                        if args.log_histogram:
                            writer.add_histogram("hist_{}_{}_pred/{}".format(coeff, scale, j),
                                                 outputs[("wavelets", scale, coeff)][j], niter)
                        writer.add_image(
                            "{}_{}_gt/{}".format(coeff, scale, j),
                            normalize_image(inputs[("wavelets", scale, coeff)][j]), niter)
                        if args.log_histogram:
                            writer.add_histogram("hist_{}_{}_gt/{}".format(coeff, scale, j),
                                                 inputs[("wavelets", scale, coeff)][j], niter)

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Single Image Depth Prediction with Wavelet Decomposition')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--model_name', type=str, default='tmp')
    parser.add_argument('--disparity', action="store_true")
    parser.add_argument("--loss_scales",
                        nargs="+",
                        type=int,
                        help="scales at which outputs are computed",
                        default=[0, 1, 2, 3])
    parser.add_argument("--output_scales",
                        nargs="+",
                        type=int,
                        help="scales used in the loss",
                        default=[0, 1, 2, 3])
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='batch size')
    parser.add_argument('--log_histogram', action="store_true")
    parser.add_argument('--load_weights_folder', type=str, help="Folder for continued training")
    parser.add_argument('--normalize_input', action="store_true")
    parser.add_argument('--supervise_LL', action="store_true")
    parser.add_argument('--num_layers', type=int, default=50)
    parser.add_argument('--encoder_type', type=str, choices=["resnet", "densenet", "mobilenet", "mobilenet_light"],
                        default="densenet")
    parser.add_argument('--use_wavelets', action="store_true")
    parser.add_argument('--no_pretrained', action="store_true")
    parser.add_argument('--dw_waveconv', action="store_true")
    parser.add_argument('--dw_upconv', action="store_true")
    parser.add_argument('--use_224', action="store_true")


    args = parser.parse_args()

    # Logging
    logpath = os.path.join(args.logdir, args.model_name)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    if os.path.exists(logpath):
        answer = None
        while answer not in ("yes", "no"):
            answer = input("A model has already been trained there, overwrite? yes or no: ")
            if answer == "yes":
                try:
                    shutil.rmtree(logpath)
                except OSError as e:
                    print("Error: %s : %s" % (logpath, e.strerror))
                    sys.exit(0)
                break
            elif answer == "no":
                print("Not overwriting. Training aborted.")
                sys.exit(0)
            else:
                print("Please enter yes or no.")

    os.makedirs(logpath)
    save_opts(logpath, args)
    with open(os.path.join(logpath, 'commandline_args.txt'), 'w') as f:
        f.write('\n '.join(sys.argv[1:]))

    # Create model
    if args.no_pretrained:
        args.pretrained_encoder = False
    else:
        args.pretrained_encoder = True

    print("Creating model...", end="")
    model = Model(args).cuda()
    print(' ...Model created.')

    if args.load_weights_folder is not None:
        model = load_model(model, args.load_weights_folder)

    # Training parameters
    optimizer = torch.optim.Adam( model.parameters(), args.lr )
    batch_size = args.bs
    prefix = 'densenet_' + str(batch_size)

    # Load data
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size, num_workers=args.num_workers,
                                                       is_224=args.use_224)
    test_iter = iter(test_loader)

    writers = {}
    for mode in ["train", "val"]:
        writers[mode] = SummaryWriter(os.path.join(logpath, mode))

    # Loss
    l1_criterion = nn.L1Loss()

    # DWT
    forward_dwt = DWT(J=4, wave='haar', mode='reflect').cuda()

    # Start training...
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        # losses = AverageMeter()
        losses = {}
        N = len(train_loader)

        # Switch to train mode
        model.train()

        end = time.time()

        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            # Normalize depth
            if args.disparity:
                depth_n = DepthNorm( depth )
            else:
                depth_n = depth
            inputs = {"disp_gt": depth_n}
            inputs["image"] = image.detach().cpu()

            pad_lowestscale = nn.Identity()
            if args.use_wavelets:
                yl_gt, yh_gt = forward_dwt(depth_n)
                inputs[("wavelets", 3, "LL")] = pad_lowestscale(yl_gt)

            # Predict
            outputs = model(image)

            # Compute the loss
            total_loss = 0

            for scale in range(4):
                if args.use_wavelets:
                    inputs[("wavelets", scale, "LH")] = yh_gt[scale][:, 0, 0]
                    inputs[("wavelets", scale, "HL")] = yh_gt[scale][:, 0, 1]
                    inputs[("wavelets", scale, "HH")] = yh_gt[scale][:, 0, 2]

                if scale in args.output_scales:
                    pred = F.interpolate(outputs[("disp", scale)],
                                         scale_factor=2**scale, mode='bilinear', align_corners=True)

                    l_depth = l1_criterion(pred, depth_n)

                    loss = (0.1 * l_depth)
                    if scale in args.loss_scales:
                        total_loss += loss
                    losses["loss/{}".format(scale)] = loss
                    losses["loss_depth/{}".format(scale)] = l_depth

            if args.use_wavelets:
                try:
                    l_LL = l1_criterion(outputs[("wavelets", 3, "LL")], yl_gt) / (2**4)
                    losses["loss_LL3"] = l_LL
                    if args.supervise_LL:
                        total_loss += l_LL
                except KeyError:
                    pass

            # Update step
            losses["loss"] = total_loss
            total_loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))
        
            # Log progress
            niter = epoch*N+i
            if i % 5 == 0:
                # Print to console
                print("Epoch: [{}][{}/{}]\t Time batch_time {:.3f} ({:.3f})\t ETA {}\t Loss {:.4f}"
                      .format(epoch, i, N, batch_time.val, batch_time.sum, eta, losses["loss"]))

            if i % 300 == 0:
                # Log to tensorboard
                log("train", inputs, outputs, losses, writers, niter, args)
                test_iter = val(model, test_iter, test_loader, forward_dwt, l1_criterion, writers, niter, args, l1_criterion)

        # save epoch
        save_model(model, logpath, epoch)


if __name__ == '__main__':
    main()

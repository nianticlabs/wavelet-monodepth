# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import time

import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from pyt_utils import *
from kitti_utils import *
from layers import *
from networks.network_constructors import *

import datasets


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.use_sparse is False, \
            "Training with sparse convolution is not implemented, please remove the --use_sparse flag for training"

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = (not (self.opt.use_stereo and self.opt.frame_ids == [0]))

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        if self.opt.use_depth_hints:
            assert 's' in self.opt.frame_ids, "Can't use depth hints without training from stereo" \
                                              "images - either add --use_stereo or remove " \
                                              "--use_depth_hints."

        # Build network
        print("Building network")
        print("Encoder...", end='')
        self.models["encoder"] = make_depth_encoder(self.opt)
        self.models["encoder"].to(self.device)
        self.parameters_to_train.append(dict(params=self.models["encoder"].parameters(), lr=self.opt.learning_rate))
        print("\t Done.")

        print("Decoder...", end='')
        self.models["depth"] = make_depth_decoder(self.models["encoder"], self.opt)
        self.models["depth"].to(self.device)
        for _, weights in self.models["depth"].convs.items():
            group_weight(self.parameters_to_train, weights, self.opt.learning_rate)
        print("\t Done.")

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"], self.models["pose"] = make_posenet(self.opt, self.models["encoder"],
                                                                                self.num_pose_frames,
                                                                                self.num_input_frames)
                self.models["pose_encoder"].to(self.device)
            else:
                _, self.models["pose"] = make_posenet(self.opt, self.models["encoder"],
                                                      self.num_pose_frames,
                                                      self.num_input_frames)
            self.models["pose"].to(self.device)

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.parameters_to_train.append(dict(params=self.models["pose_encoder"].parameters(),
                                                     lr=self.opt.learning_rate))
            self.parameters_to_train.append(dict(params=self.models["pose"].parameters(), lr=self.opt.learning_rate))

        self.model_optimizer = optim.Adam(self.parameters_to_train,
                                          self.opt.learning_rate,
                                          weight_decay=1e-5)

        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs #- self.opt.start_epoch)

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.opt.scales, is_train=True, img_ext=img_ext,
            use_depth_hints=self.opt.use_depth_hints, depth_hint_path=self.opt.depth_hint_path)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.opt.scales, is_train=False, img_ext=img_ext,
            use_depth_hints=self.opt.use_depth_hints, depth_hint_path=self.opt.depth_hint_path)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = int(self.opt.height // (2 ** scale))
            w = int(self.opt.width // (2 ** scale))

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = self.opt.start_epoch
        self.step = self.opt.start_epoch * (len(self.train_loader) // self.opt.batch_size)
        self.start_time = time.time()
        for epoch in range(self.opt.start_epoch, self.opt.num_epochs, 1):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
            self.epoch += 1

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        before_dataload_time = time.time()
        for batch_idx, inputs in enumerate(self.train_loader):
            durations = {}
            durations["dataloading"] = time.time() - before_dataload_time
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            durations["batch_process"] = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 1000
            late_phase = self.step % 1000 == 0

            if early_phase or late_phase:

                before_log_time = time.time()

                self.log("train", inputs, outputs, losses)
                durations["logging"] = time.time() - before_log_time
                self.log_time(batch_idx, durations, losses["loss"].cpu().data)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            try:
                inputs[key] = ipt.to(self.device)
            except AttributeError:
                pass

        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        if self.opt.use_depth_hints:
            losses = self.compute_losses_hints(inputs, outputs)
        else:
            losses = self.compute_losses_mdp(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.loss_scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

        if self.opt.use_depth_hints:
            if "s" in self.opt.frame_ids[1:]:
                T = inputs["stereo_T"]
                source_scale = 0
                depth = inputs['depth_hint']
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("color_depth_hint", "s", source_scale)] = F.grid_sample(
                    inputs[("color", "s", source_scale)],
                    pix_coords, padding_mode="border")

    def compute_reprojection_loss(self, pred, target, use_ssim=True):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim or not use_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    @staticmethod
    def compute_proxy_supervised_loss(pred, target, valid_pixels, loss_mask):
        """ Compute proxy supervised loss (depth hint loss) for prediction.
            - valid_pixels is a mask of valid depth hint pixels (i.e. non-zero depth values).
            - loss_mask is a mask of where to apply the proxy supervision (i.e. the depth hint gave
            the smallest reprojection error)"""

        # first compute proxy supervised loss for all valid pixels
        depth_hint_loss = torch.log(torch.abs(target - pred) + 1) * valid_pixels

        # only keep pixels where depth hints reprojection loss is smallest
        depth_hint_loss = depth_hint_loss * loss_mask

        return depth_hint_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss,
                           depth_hint_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection.
        identity_reprojections_loss and/or depth_hint_reprojection_loss can be None"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

            if depth_hint_reprojection_loss:
                all_losses = torch.cat([reprojection_loss, depth_hint_reprojection_loss], dim=1)
                idxs = torch.argmin(all_losses, dim=1, keepdim=True)
                depth_hint_loss_mask = (idxs == 1).float()

        else:
            # we are using automasking
            if depth_hint_reprojection_loss is not None:
                all_losses = torch.cat([reprojection_loss, identity_reprojection_loss,
                                        depth_hint_reprojection_loss], dim=1)
            else:
                all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)

            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs != 1).float()  # automask has index '1'
            depth_hint_loss_mask = (idxs == 2).float()  # will be zeros if not using depth hints

        # just set depth hint mask to None if not using depth hints
        depth_hint_loss_mask = \
            None if depth_hint_reprojection_loss is None else depth_hint_loss_mask

        return reprojection_loss_mask, depth_hint_loss_mask

    def compute_losses_hints(self, inputs, outputs):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0

        # compute depth hint reprojection loss
        if self.opt.use_depth_hints:
            pred = outputs[("color_depth_hint", 's', 0)]
            depth_hint_reproj_loss = self.compute_reprojection_loss(pred, inputs[("color", 0, 0)])
            # set loss for missing pixels to be high so they are never chosen as minimum
            depth_hint_reproj_loss += 1000 * (1 - inputs['depth_hint_mask'])
        else:
            depth_hint_reproj_loss = None

        for scale in self.opt.loss_scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # differently to Monodepth2, compute mins as we go
                    identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                              keepdim=True)
            else:
                raise NotImplementedError

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                # differently to Monodepth2, compute mins as we go
                reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

            # find minimum losses from [reprojection, identity, depth hints reprojection]
            reprojection_loss_mask, depth_hint_loss_mask = \
                self.compute_loss_masks(reprojection_loss,
                                        identity_reprojection_loss,
                                        depth_hint_reproj_loss)

            # standard reprojection loss
            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

            outputs["identity_selection/{}".format(scale)] = (1 - reprojection_loss_mask).float()
            losses['reproj_loss/{}'.format(scale)] = reprojection_loss

            # proxy supervision loss
            depth_hint_loss = 0
            if self.opt.use_depth_hints:
                target = inputs['depth_hint']
                pred = outputs[('depth', 0, scale)]
                valid_pixels = inputs['depth_hint_mask']

                depth_hint_loss = self.compute_proxy_supervised_loss(pred, target, valid_pixels,
                                                                     depth_hint_loss_mask)
                depth_hint_loss = depth_hint_loss.sum() / (depth_hint_loss_mask.sum() + 1e-7)
                # save for logging
                outputs["depth_hint_pixels/{}".format(scale)] = depth_hint_loss_mask
                losses['depth_hint_loss/{}'.format(scale)] = depth_hint_loss

            loss += reprojection_loss + depth_hint_loss

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses

    def compute_losses_mdp(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        total_sparsity_loss = 0

        for scale in self.opt.loss_scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale or scale <= 0:
                source_scale = scale
            else:
                source_scale = 0

            if ("disp", scale) not in outputs:
                continue
            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            else:
                raise NotImplementedError

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            if self.opt.disparity_smoothness != 0:
                smooth_loss = get_smooth_loss(norm_disp, color)
            else:
                smooth_loss = 0

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses

    def log_time(self, batch_idx, durations, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / durations["batch_process"]
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | forward: {:2.1f}s (examples/s: {:5.1f}) |" + \
            " logging: {:2.1f}s | dataloading: {:2.1f}s | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, durations["batch_process"],
                                  samples_per_sec, durations["logging"], durations["dataloading"], loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for frame_id in self.opt.frame_ids:
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, 0, j),
                    inputs[("color", frame_id, 0)][j].data, self.step)

                try:
                    writer.add_image(
                        "color_pred_{}_{}/{}".format(frame_id, 0, j),
                        outputs[("color", frame_id, 0)][j].data, self.step)
                except KeyError:
                    pass
            for scale in self.opt.scales:
                if scale in self.opt.loss_scales:
                    writer.add_image(
                        "automask_{}/{}".format(scale, j),
                        outputs["identity_selection/{}".format(scale)][j][None, ...], self.step)
                    if self.opt.use_depth_hints:
                        # try:
                        if frame_id == "s" and scale==0:
                            writer.add_image(
                                "depth_hints_mask{}/{}".format(scale, j),
                                outputs["depth_hint_pixels/{}".format(scale)][j][None, ...], self.step)
                            writer.add_image(
                                "disp_hints/{}".format(j),
                                normalize_image(inputs["disp_hint"][j][None, ...]), self.step)
                            writer.add_image(
                                "color_depth_hint{}/{}".format(scale, j),
                                outputs[("color_depth_hint", frame_id, scale)][j][None, ...], self.step)

                if self.opt.use_wavelets:
                    for coeff in ["LL", "LH", "HL", "HH"]:
                        if ("wavelets", scale, coeff) in outputs:
                            writer.add_image(
                                "{}_{}/{}".format(coeff, scale, j),
                                normalize_image(torch.mean(outputs[("wavelets", scale, coeff)][j], 0,
                                                           keepdim=True)),
                                self.step)
                            writer.add_histogram("hist_{}_{}/{}".format(coeff, scale, j),
                                                 outputs[("wavelets", scale, coeff)][j], self.step)
                            m = (outputs[("wavelets", scale, coeff)][j]).min()
                            M = (outputs[("wavelets", scale, coeff)][j]).max()
                            writer.add_scalar("min/{}_{}/{}".format(coeff, scale, j), m, self.step)
                            writer.add_scalar("max/{}_{}/{}".format(coeff, scale, j), M, self.step)

                try:
                    m = (outputs[('disp', scale)][j]).min()
                    M = (outputs[('disp', scale)][j]).max()
                    writer.add_image(
                        "disp_{}/{}".format(scale, j),
                        (outputs[('disp', scale)][j]-m) / ((M - m) if m != M else 1e5), self.step)
                except KeyError:
                    pass

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            if n == "adam":
                continue
            if n in ["pose_encoder", "pose"] and not self.use_pose_net:
                continue
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if "adam" in self.opt.models_to_load:
            try:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except Exception as e:
                print(e)
                print("Cannot find Adam weights so Adam is randomly initialized")

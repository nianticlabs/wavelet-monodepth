# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random

import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms as T
import torchvision.transforms.functional as TF

from layers import depth_to_disp

MIN_DEPTH=0.1
MAX_DEPTH=100.0

cv2.setNumThreads(0)

def pil_rgb_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def pil_depth_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 target_scales=[0, 1, 2, 3],
                 use_depth_hints=False,
                 depth_hint_path=None,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.target_scales = target_scales
        self.num_scales = len(self.target_scales)
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.rgb_loader = pil_rgb_loader
        self.depth_loader = pil_rgb_loader
        self.to_tensor = T.ToTensor()

        self.use_depth_hints = use_depth_hints
        if self.use_depth_hints:
            self.with_hints = 0
            self.without_hints = 0
        if depth_hint_path is None:
            self.depth_hint_path = os.path.join(self.data_path, 'depth_hints')
        else:
            self.depth_hint_path = depth_hint_path

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            T.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for s in self.target_scales:
            scale = 2 ** s
            interp = self.interp if s > -1 else Image.BICUBIC
            self.resize[s] = T.Resize((int(self.height // scale), int(self.width // scale)),
                                      interpolation=interp)

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """

        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k

                last_scale = (n, im, -1)
                for j in range(self.num_scales):
                    s = self.target_scales[j]
                    if s == -1:
                        continue
                    else:
                        inputs[(n, im, s)] = self.resize[s](inputs[last_scale])
                        last_scale = (n, im, s)

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
                inputs["image_path"] = self.get_image_path(folder, frame_index, other_side).split(self.data_path, 1)[-1]
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)
                inputs["image_path"] = self.get_image_path(folder, frame_index + i, side).split(self.data_path, 1)[-1]

        # adjusting intrinsics to match each scale in the pyramid
        for scale in self.target_scales:
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = T.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            if ("color_aug", i, -1) in inputs:
                del inputs[("color_aug", i, -1)]
            if ("color", i, -1) in inputs:
                del inputs[("color", i, -1)]


        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

            # load depth hint
            if self.use_depth_hints:
                side_folder = 'image_02' if side == 'l' else 'image_03'
                depth_folder = os.path.join(self.depth_hint_path, folder, side_folder,
                                            str(frame_index).zfill(10) + '.npy')
                try:
                    depth = np.load(depth_folder)[0]
                    file_found = True
                    self.with_hints += 1
                    # print("Found hints")
                except FileNotFoundError:
                    # raise FileNotFoundError("Warning - cannot find depth hint for {} {} {}! "
                    #                         "Either specify the correct path in option "
                    #                         "--depth_hint_path, or run precompute_depth_hints.py to"
                    #                         "train with depth hints".format(folder, side_folder,
                    #                                                         frame_index))
                    file_found = False
                    self.without_hints += 1
                    # print("Not found")

                if file_found:
                    if do_flip:
                        depth = np.fliplr(depth)

                    depth = cv2.resize(depth, dsize=(self.width, self.height),
                                       interpolation=cv2.INTER_NEAREST)
                    disp = depth_to_disp(depth, MIN_DEPTH, MAX_DEPTH)
                    inputs['disp_hint'] = torch.from_numpy(disp).float().unsqueeze(0)
                    inputs['depth_hint'] = torch.from_numpy(depth).float().unsqueeze(0)
                    inputs['depth_hint_mask'] = (inputs['depth_hint'] > 0).float()

                else:
                    inputs['depth_hint'] = torch.zeros((self.height, self.width)).float().unsqueeze(0)
                    inputs['depth_hint_mask'] = inputs['depth_hint']

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_image_path(self, folder, frame_index, side):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    @staticmethod
    def get_color_transform(gamma=None,
                            contrast=None,
                            brightness=None, saturation=None, hue=None,
                            color_brightness=None):

        transforms = []

        # perform gamma transform first
        if gamma is not None:
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(T.Lambda(lambda img: TF.adjust_gamma(img, gamma_factor)))

        color_transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            color_transforms.append(T.Lambda(lambda img: TF.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            color_transforms.append(T.Lambda(lambda img: TF.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            color_transforms.append(T.Lambda(lambda img: TF.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            color_transforms.append(T.Lambda(lambda img: TF.adjust_hue(img, hue_factor)))

        # all of the above color transforms can be shuffled
        random.shuffle(color_transforms)

        if color_brightness is not None:
            random_colors = np.random.uniform(color_brightness[0], color_brightness[1], 3)
            # randomly shift color
            # color_transforms.append(T.Lambda(lambda img: img * random_colors))
            color_transforms.append(T.Lambda(lambda img: adjust_color_brightness(img, random_colors)))

        transforms += color_transforms
        transform = T.Compose(transforms)
        return transform


def adjust_color_brightness(img, color_factors):
    """
    Color Brightness Jitter
    """
    if not TF._is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    color_factors = np.array(color_factors)

    input_mode = img.mode
    img = img.convert('RGB')

    r, g, b = img.split()
    # use PIL's point-function to accelerate this part
    r = r.point(lambda c: c * color_factors[0])
    g = g.point(lambda c: c * color_factors[1])
    b = b.point(lambda c: c * color_factors[2])

    img = Image.merge('RGB', (r, g, b))
    img = img.convert(input_mode)
    return img

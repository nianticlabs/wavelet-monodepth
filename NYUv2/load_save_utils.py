# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import torch
import json

def save_model(model, log_path, epoch):
    """Save model weights to disk
    """
    save_folder = os.path.join(log_path, "models", "weights_{}".format(epoch))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = os.path.join(save_folder, "model.pth")
    to_save = model.state_dict()
    torch.save(to_save, save_path)


def load_model(model, load_weights_folder):
    """Load model(s) from disk
    """
    load_weights_folder = os.path.expanduser(load_weights_folder)

    assert os.path.isdir(load_weights_folder), \
        "Cannot find folder {}".format(load_weights_folder)
    print("loading model from folder {}".format(load_weights_folder))

    path = os.path.join(load_weights_folder, "model.pth")
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def save_opts(log_path, opts):
    """Save options to disk so we know what we ran this experiment with
    """
    models_dir = os.path.join(log_path, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    to_save = opts.__dict__.copy()

    with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
        json.dump(to_save, f, indent=2)

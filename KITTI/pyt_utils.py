# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

# Modified from TorchSeg
# https://github.com/ycszen/TorchSeg/blob/master/furnace/utils/pyt_utils.py

import torch.nn as nn

def group_weight(weight_group, module, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(
        group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group
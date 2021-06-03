# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the WaveletMonoDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from utils import evaluate
import argparse
from model import Model
from load_save_utils import *
import numpy as np
import os
import scipy.io as io
import h5py
from imageio import imread


# Argument Parser
parser = argparse.ArgumentParser(description='Single Image Depth Prediction with Wavelet Decomposition')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
parser.add_argument('--data_path', type=str, help="Folder containing weights NYUv2 data", default="data")
parser.add_argument('--load_weights_folder', type=str, help="Folder containing weights for evaluation")
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--logdir', type=str, default='log')
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
parser.add_argument('--disparity', action="store_true")
parser.add_argument('--normalize_input', action="store_true")
parser.add_argument('--num_layers', type=int, default=50)
parser.add_argument('--encoder_type', type=str, choices=["resnet", "densenet", "mobilenet", "mobilenet_light"],
                    default="densenet")
parser.add_argument('--use_wavelets', action="store_true")
parser.add_argument('--use_sparse', action="store_true")
parser.add_argument('--threshold', type=float, default=0.04)
parser.add_argument('--save_preds', action="store_true")
parser.add_argument('--save_npy', action="store_true")
parser.add_argument('--dw_waveconv', action="store_true")
parser.add_argument('--dw_upconv', action="store_true")

parser.add_argument('--eval_edges', action="store_true")
parser.add_argument('--use_224', action="store_true")

args = parser.parse_args()
# Load test data
print('Loading test data...', end='')


EIGEN_CROP = [20, 459, 24, 615]

nyu_splits = io.loadmat(os.path.join(args.data_path, 'nyuv2_splits.mat'))
data = h5py.File(os.path.join(args.data_path, 'nyu_depth_v2_labeled.mat'), 'r', libver='latest', swmr=True)
rgb = np.array(data['images']).transpose(0, 3, 2, 1)
depth = np.array(data['depths']).transpose(0, 2, 1)

idx2nyu_dict = {}
nyu_idx_list = []
for i in range(len(nyu_splits['testNdxs'])):
    nyu_idx_list.append(int(nyu_splits['testNdxs'][i])-1)
rgb = rgb[nyu_idx_list]
depth = depth[nyu_idx_list]

edge_gt_dir = None
edges = None
if args.eval_edges:
    print(" Loading edges GT...\t", end="")
    edge_gt_dir = os.path.join(args.data_path, 'NYUv2_OCpp')
    edges = np.zeros_like(depth)
    for i in range(len(nyu_idx_list)):
        edges[i] = np.array(imread(os.path.join(edge_gt_dir, "oc", "{:04d}_oc.png".format(nyu_idx_list[i]))),
                            dtype=np.float) / 255.0
    print("Done")

print('Test data loaded.\n')

print("Building model...\t", end='')

args.pretrained_encoder = False
model = Model(args).cuda()
print("Done.")

if args.load_weights_folder is not None:
    model = load_model(model, args.load_weights_folder)

model.eval()

e, e_edges = evaluate(model, rgb, depth, EIGEN_CROP, edges=edges,
                      save_figs=args.save_preds, save_dir=args.load_weights_folder,
                      save_npy=args.save_npy,
                      save_wavelets=(args.use_wavelets and args.save_npy), use_disparity=args.disparity,
                      use_sparse=args.use_sparse, threshold=args.threshold,
                      use_224=args.use_224, nyu_idx_list=nyu_idx_list)

print('Testing...')
if not args.eval_edges:
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rms', 'log_10', 'a1', 'a2', 'a3'))
    print(("&{:10.4f}  " * 6).format(e[0], e[1], e[2], e[3], e[4], e[5]))
else:
    print(("&{:>10}  " * 8).format('rel', 'rms', 'log_10', 'a1', 'a2', 'a3', 'e_acc', 'e_comp'))
    print(("&{:10.4f}  " * 8).format(e[0], e[1], e[2], e[3], e[4], e[5], e_edges[0], e_edges[1]))


# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.FINETUNE = False
_C.MODEL.HTM_MODEL_PATH = ''

# hourglass
_C.MODEL.NET = CN()
_C.MODEL.NET.NCHANNELS = 256
_C.MODEL.NET.NSTACKS = 2
_C.MODEL.NET.NMODULES = 1
_C.MODEL.NET.NREDUCTIONS = 4

# auto_semantic_augmentation
_C.ASA = CN()
_C.ASA.INTERVAL = 1
_C.ASA.LR = 0.05
_C.ASA.NUM_MODELS = 4
_C.ASA.THETA_GRAD = True
_C.ASA.USE_SA = True
_C.ASA.LOAD_THETA = False
_C.ASA.THETA_PATH = ''
_C.ASA.BETA1 = 0.5
_C.ASA.BETA2 = 0.999
_C.ASA.NUM_AUG = (0,2,4,6,8)
_C.ASA.SCALE_RANGE = (0.7, 1.3)
_C.ASA.PART_ANN_FILE = './lip/parts_filter_done/part_anns.json'
_C.ASA.PART_ROOT_DIR = './lip/parts_filter_done/'
_C.ASA.ERODE_KERNEL = (13, 13)
_C.ASA.GAUSSIAN_KERNEL = (15, 15)
_C.ASA.BOTH_HF_SA = True

_C.STN = CN()
_C.STN.LR = 0.01
_C.STN.STN_FIRST = 0
_C.STN.NG = 1
_C.STN.ND = 1

#gcn
_C.GCN = CN()
_C.GCN.A = [[1, 2], [2, 0, 3], [1, 0, 4], [1, 5], [2, 6], [11, 6, 7, 3],[12, 5, 8, 4], [5, 9], [6, 10], [7], [8], [13, 12, 5], [14, 11, 6],[15, 11], [16, 12], [13], [14]]
_C.GCN.NODE_DEPTH=8
_C.GCN.NUM_NODE=17
_C.GCN.NUM_LAYERS=3
_C.GCN.NUM_HEAD=1
_C.GCN.GAT_ACT_FUNC='sigmoid'

_C.NON_LOCAL = CN()
_C.NON_LOCAL.DEPTH =  16
_C.NON_LOCAL.EMBED = True
_C.NON_LOCAL.POOL = 1
_C.NON_LOCAL.RESIDUAL = False
_C.NON_LOCAL.NUM_HEAD = 1
_C.NON_LOCAL.EBR = True
_C.NON_LOCAL.ER = True
_C.NON_LOCAL.WZR = True
_C.NON_LOCAL.FUSE = 'add'
_C.NON_LOCAL.FUSE_CAT_KSIZE = 1

_C.FUSE = CN()
_C.FUSE.MANNER = 'cat'
_C.FUSE.CAT_KSIZE = 1
_C.FUSE.CAT_LAYERS = 1
_C.FUSE.ADD3_KSIZE = 1
_C.FUSE.ADD3_LAYERS = 1


_C.LOSS = CN()
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'mpii'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.SELECT_DATA = False

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.PROB_HALF_BODY = 0.0
_C.DATASET.NUM_JOINTS_HALF_BODY = 8
_C.DATASET.COLOR_RGB = False

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False

_C.TEST.USE_GT_BBOX = False

# nms
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.NMS_THRE = 0.6
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0
_C.TEST.MODEL_FILE = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    cfg.DATASET.ROOT = os.path.join(
        cfg.DATA_DIR, cfg.DATASET.ROOT
    )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)


# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models
import cv2
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def copy_prev_models(prev_models_dir, model_dir):
    import shutil

    vc_folder = '/hdfs/' \
        + '/' + os.environ['PHILLY_VC']
    source = prev_models_dir
    # If path is set as "sys/jobs/application_1533861538020_2366/models" prefix with the location of vc folder
    source = vc_folder + '/' + source if not source.startswith(vc_folder) \
        else source
    destination = model_dir

    if os.path.exists(source) and os.path.exists(destination):
        for file in os.listdir(source):
            source_file = os.path.join(source, file)
            destination_file = os.path.join(destination, file)
            if not os.path.exists(destination_file):
                print("=> copying {0} to {1}".format(
                    source_file, destination_file))
            shutil.copytree(source_file, destination_file)
    else:
        print('=> {} or {} does not exist'.format(source, destination))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(233)
    args = parse_args()
    update_config(cfg, args)

    if args.prevModelDir and args.modelDir:
        # copy pre models for philly
        copy_prev_models(args.prevModelDir, args.modelDir)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    stn = eval('models.'+cfg.MODEL.NAME+'.get_stn_net')(
        cfg, is_train=True
    )
    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    stn = torch.nn.DataParallel(stn, device_ids=cfg.GPUS).cuda()


    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, False, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    stn_optimizer = torch.optim.Adam(stn.parameters(), lr=cfg.STN.LR)
    ouput_dir = os.path.join(final_output_dir, 'stn_vis')
    if not os.path.exists(ouput_dir):
        os.makedirs(ouput_dir)
    for epoch in range(40, 20000, 40):
        stn_file_path = os.path.join(final_output_dir, 'stn_{}.pth'.format(epoch+1))
        if not os.path.exists(stn_file_path):
            break

        #set_seed(2333)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )

        stn_checkpoint = torch.load(stn_file_path)
        stn.load_state_dict(stn_checkpoint['stn_state_dict'])
        stn_optimizer.load_state_dict(stn_checkpoint['stn_optimizer'])
        print('loading {}'.format(stn_file_path))
        vis_stn(train_loader, stn, ouput_dir, epoch+1, 100000000)
        

def vis_stn(train_loader, stn, ouput_dir, epoch, img_idx=0):
    stn.eval()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        N = input.size(0)
        if i < img_idx:
            pass#continue
        elif i > img_idx:
            break
        # stn
        part_imgs = meta['part_imgs']
        part_masks = meta['part_masks']
        init_thetas = meta['init_thetas']
        part_idxes = meta['part_idxes']
        input = stn(input, part_imgs, part_masks, part_idxes, init_thetas)

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        III = input.detach().cpu().numpy()
        for iI, I in enumerate(III):
            I = I.transpose(1,2,0)
            I = (I*np.array(std) + np.array(mean))*255
            I = I.astype(np.uint8)
            
            I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(ouput_dir, '{}_{}.jpg'.format(iI+i*N, epoch)), I)

if __name__ == '__main__':
    main()

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

import torch.nn as nn
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
from dataset.semantic_augmentation import SemanticAugmentation
import dataset
import models

from torch.multiprocessing import Process, Pipe, set_start_method
from collections import defaultdict, OrderedDict
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass
import time
import numpy as np
import random
import json
import cv2
import matplotlib.pyplot as plt

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


def main():
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

    def model_optimizer_lrSchedule_dataset(device_ids):
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=True
        )

        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()

        theta_pool = None
        theta_optimizer_state_dict = None
        all_best_perf = 0.0
        best_model = False
        last_epoch = -1
        optimizer = get_optimizer(cfg, model)
        begin_epoch = cfg.TRAIN.BEGIN_EPOCH
        checkpoint_file = os.path.join(
            final_output_dir, 'checkpoint.pth'
        )

        if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer'])

            theta_pool = checkpoint['theta_pool']
            theta_optimizer_state_dict = checkpoint['theta_optimizer']
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch
        )

        

        return model, optimizer, lr_scheduler, theta_pool, theta_optimizer_state_dict, begin_epoch, all_best_perf

    num_models = cfg.ASA.NUM_MODELS
    devices_per_model = [[0,1,2,3], [4,5,6,7],
                        [0,1,2,3], [4,5,6,7]]
    num_parts_per_augmentation = [0, 2, 4, 6, 8]
    vertical_translate = [0.0, 0.1, 0.2, 0.3]
    horizontal_translate = [0.0, 0.1, 0.2, 0.3]
    part_scale = np.arange(0.7, 1.4, 0.1)
    part_type = np.arange(0, 25)

    semantic_augmentation_core = SemanticAugmentation(0,0,0,0)
    semantic_augmentation_param_pool = OrderedDict()
    semantic_augmentation_param_pool['st'] = []
    for s in part_scale:
        for t in part_type:
            semantic_augmentation_param_pool['st'].append({'scale_factor':s, 'part_type':t})
    semantic_augmentation_param_pool['num_augmentation'] = num_parts_per_augmentation
    semantic_augmentation_param_pool['vertical_translate'] = vertical_translate
    semantic_augmentation_param_pool['horizontal_translate'] = horizontal_translate

    
    # initialize network model
    optimizers = []
    lr_schedulers = []
    models = []
    for i in range(num_models):
        model, optimizer, lr_scheduler, theta_pool, theta_optimizer_state_dict, begin_epoch, all_best_perf = model_optimizer_lrSchedule_dataset(cfg.GPUS)
        optimizers.append(optimizer)
        models.append(model)
        lr_schedulers.append(lr_scheduler)

    # load theta
    if cfg.ASA.LOAD_THETA:
        assert os.path.exists(cfg.ASA.THETA_PATH), 'error path {}'.format(cfg.ASA.THETA_PATH)
        logger.info("=> loaded theta {}".format(cfg.ASA.THETA_PATH))
        checkpoint = torch.load(cfg.ASA.THETA_PATH)
        theta_pool = checkpoint['theta_pool']

    # sample augmentation for different instances and models
    num_distributions = len(semantic_augmentation_param_pool.keys())
    if theta_pool is None:
        theta_pool = []
        for k, v in semantic_augmentation_param_pool.items():
            theta_pool.append(
                nn.Parameter(
                    torch.ones(len(v), dtype=torch.float32) / len(v), 
                    requires_grad=cfg.ASA.THETA_GRAD
                    )
                )
    else:
        theta_pool = [nn.Parameter(t, requires_grad=cfg.ASA.THETA_GRAD) for t in theta_pool]
    # initialize theta optimizer
    theta_optimizer = torch.optim.Adam(theta_pool, lr=cfg.ASA.LR, betas=(0.5, 0.999))
    if theta_optimizer_state_dict is not None:
        theta_optimizer.load_state_dict(theta_optimizer_state_dict)



    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    if cfg.ASA.USE_SA:
        setattr(train_dataset, 'semantic_augmentation_param_pool', semantic_augmentation_param_pool)
        setattr(train_dataset, 'semantic_augmentation_core', semantic_augmentation_core)

    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

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

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    # writer_dict['writer'].add_graph(model, (dump_input, ))

    logger.info(get_model_summary(model, dump_input))

    

    interval = cfg.ASA.INTERVAL
    begin_interval = begin_epoch // interval
    end_interval = cfg.TRAIN.END_EPOCH // interval

    # initial multiprocess
    # conn_pool = []
    # process_pool = []
    # for ith_Process in range(num_models):
    #     conn_pool.append(Pipe())
    #     args=(conn_pool[ith_Process][1], devices_per_model[ith_Process], interval, begin_epoch, all_best_perf, lr_scheduler, cfg, 
    #             train_dataset, valid_loader, valid_dataset, criterion, final_output_dir, 
    #             tb_log_dir, writer_dict)
    #     process_pool.append(Process(target=train_validate, args=args))
    #     process_pool[ith_Process].start()


    end = time.time()
    for ith_interval in range(begin_interval, end_interval):
        # save theta
        with open(os.path.join(final_output_dir, 'theta.json'), 'a') as f:
            theta_pool_to_save = [t.detach().cpu().numpy().tolist() for t in theta_pool]
            f.write('\n')
            f.write(json.dumps(theta_pool_to_save))
        # summary theta
        summary_theta(writer_dict['writer'], ith_interval, theta_pool, semantic_augmentation_param_pool, len(part_scale), len(part_type))
        end = time.time()
        aug_prob_distributions = [torch.nn.functional.softmax(t) for t in theta_pool]
        semantic_augmentation_idxes_pool = []
        perf_pool = []
        best_perf_pool = []
        for i in range(num_models):
            tmp = []
            for jth in range(len(aug_prob_distributions)):
                tmp.append(torch.multinomial(aug_prob_distributions[jth], len(train_dataset)*interval, replacement=True))

            semantic_augmentation_idxes_pool.append(tmp)
            output = train_validate(interval, ith_interval*interval, all_best_perf, lr_schedulers[i], semantic_augmentation_idxes_pool[i], cfg, 
                        train_dataset, valid_loader, valid_dataset, models[i], criterion, optimizers[i], final_output_dir, 
                        tb_log_dir, writer_dict)
            model, optimizer, perf_indicator, best_perf = output
            # model_pool.append(model)
            # optimizer_pool.append(optimizer)
            perf_pool.append(perf_indicator)
            best_perf_pool.append(best_perf)
            
            

        # calc loss to update Theta
        acc = torch.Tensor(perf_pool)
        acc = acc - acc.mean()
        P_tra_pool = []
        for i in range(num_models):
            tmp = []
            for jth in range(len(semantic_augmentation_idxes_pool[i])):
                tmp.append(aug_prob_distributions[jth][semantic_augmentation_idxes_pool[i][jth]])
            tmp = torch.stack(tmp, dim=0)
            P_tra_pool.append(tmp)
        P_tra_pool = torch.stack(P_tra_pool, dim=0)
        P_tra_pool = torch.log(P_tra_pool).sum(dim=2).sum(dim=1)
        theta_loss = P_tra_pool * acc
        theta_loss = -theta_loss.mean()

        # compute gradient and do update step
        if cfg.ASA.THETA_GRAD:
            print('theta lr: {}'.format(theta_optimizer.param_groups()[0]['lr']))
            theta_optimizer.zero_grad()
            theta_loss.backward()
            theta_optimizer.step()



        cur_best_perf = max(perf_pool)
        best_perf = max(best_perf_pool)

        # import pdb
        # pdb.set_trace()
        next_idx = perf_pool.index(cur_best_perf)
        update_model_optimizer(models, optimizers, next_idx)


        if all_best_perf < cur_best_perf:
            all_best_perf = cur_best_perf
            best_model = True
        else:
            best_model = False
        
        # print
        msg = 'Outer loop: [{}]\t' \
                  'Time {}s \t' \
                  'Loss {} \t' \
                  'current PCKh@0.5 {} \t' \
                  'best PCKh@0.5 {}'.format(
                      ith_interval, round(time.time()-end, 2), theta_loss.cpu().detach().numpy(), cur_best_perf, all_best_perf)
        logger.info(msg)



        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        theta_pool_to_save = [t.data for t in theta_pool]
        save_checkpoint({
            'epoch': ith_interval + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.state_dict(),
            'perf': cur_best_perf,
            'optimizer': optimizer.state_dict(),
            'theta_pool':theta_pool_to_save,
            'theta_optimizer': theta_optimizer.state_dict()
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


def cvt_BGR2RBG(img):
    B, G, R = cv2.split(img)
    img = cv2.merge([R, B, G])
    return img

def norm_fm(fm):
    min = np.amin(fm)
    max = np.amax(fm)
    fm = (fm-min)/(max-min+0.00001) * 255
    return fm.astype(np.uint8)


def summary_theta(writer, step, theta_pool, semantic_augmentation_param_pool, num_scales, num_parts):
    distributions = [torch.nn.functional.softmax(t) for t in theta_pool]
    for i, k in enumerate(semantic_augmentation_param_pool.keys()):
        to_be_vis = distributions[i].detach().cpu().numpy()
        if k == 'st':
            to_be_vis = to_be_vis.reshape(num_scales, num_parts)
            to_be_vis = norm_fm(to_be_vis)
            to_be_vis = cv2.applyColorMap(to_be_vis, cv2.COLORMAP_JET)
            to_be_vis = cv2.cvtColor(to_be_vis, cv2.COLOR_BGR2RGB)
        else:
            # to_be_vis = to_be_vis.reshape(1, len(to_be_vis))
            plt.figure()
            left = list(range(len(to_be_vis)))
            plt.bar(left, to_be_vis)
            plt.savefig('tmp.jpg')
            plt.close()
            to_be_vis = cv2.imread('tmp.jpg')
            to_be_vis = cv2.cvtColor(to_be_vis, cv2.COLOR_BGR2RGB)
        writer.add_image(k, to_be_vis, step, dataformats='HWC')


def update_model_optimizer(models, optimizers, idx):
    next_model = models[idx]
    next_optimizer = optimizers[idx]
    for i in range(len(models)):
        if i == idx:
            continue

        models[i].load_state_dict(next_model.state_dict())
        optimizers[i].load_state_dict(next_optimizer.state_dict())


def train_validate(interval, epoch, best_perf, lr_scheduler, semantic_augmentation_idxes, cfg, 
                    train_dataset, valid_loader, valid_dataset, model, criterion, optimizer, final_output_dir, 
                    tb_log_dir, writer_dict):
    
    for i in range(interval):

        if hasattr(train_dataset, 'aug_idxes'):
            train_dataset.aug_idxes = [t[i*len(train_dataset):(i+1)*len(train_dataset)] for t in semantic_augmentation_idxes]
        else:
            setattr(train_dataset, 'aug_idxes', [t[i*len(train_dataset):(i+1)*len(train_dataset)] for t in semantic_augmentation_idxes])

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
            shuffle=cfg.TRAIN.SHUFFLE,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )


        lr_scheduler.step()
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        
        # evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, writer_dict
        )

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        epoch += 1
    return model, optimizer, perf_indicator, best_perf
    # pass
    # return model, optimizer, random.randint(1,100), 92.0



if __name__ == '__main__':
    main()

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
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass
import time
import numpy as np

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

    def model_optimizer_lrSchedule_dataset(semantic_augmentation_param_pool, semantic_augmentation_core):
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=True
        )

        # model = torch.nn.DataParallel(model, device_ids=deveces).cuda()

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
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch
        )

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

        return model, optimizer, lr_scheduler, train_dataset, valid_dataset, valid_loader, begin_epoch, all_best_perf

    num_models = 2
    devices_per_model = [[0,1,2,3], [4,5,6,7],
                        [4,5,6,7], [4,5,6,7]]
    num_parts_per_augmentation = [0, 2, 4, 6, 8]
    vertical_translate = [0.0, 0.1, 0.2, 0.3]
    horizontal_translate = [0.0, 0.1, 0.2, 0.3]
    part_scale = np.arange(0.7, 1.4, 0.1)

    semantic_augmentation_core = SemanticAugmentation(0,0,0,0)
    semantic_augmentation_param_pool = []
    for s in part_scale:
        for v in vertical_translate:
            for h in horizontal_translate:
                for n in num_parts_per_augmentation:
                    semantic_augmentation_param_pool.append(
                        {
                        'scale_factor':s, 
                        'vertical_translate':v, 
                        'horizontal_translate':h, 
                        'num_augmentation':n
                        }
                    )
    # sample augmentation for different instances and models
    size_semantic_augmentation_pool = len(semantic_augmentation_param_pool)
    theta = nn.Parameter(torch.ones(size_semantic_augmentation_pool, dtype=torch.float32) / size_semantic_augmentation_pool, requires_grad=True)
    # initialize theta optimizer
    theta_optimizer = torch.optim.Adam([theta], lr=0.05, betas=(0.5, 0.999))
    # initialize network model
    model, optimizer, lr_scheduler, train_dataset, valid_dataset, valid_loader, begin_epoch, all_best_perf = model_optimizer_lrSchedule_dataset(semantic_augmentation_param_pool, semantic_augmentation_core)


    # optimizers = []
    # lr_schedulers = []
    # models = []
    # train_datasets = []
    # train_loaders = []
    # for i in num_models:
    #     model, optimizer, lr_scheduler, train_dataset, train_loader = model_optimizer_lrSchedule_dataset(devices_per_model[i], semantic_augmentation_pool, aug_prob_distribution)
    #     optimizers.append(optimizer)
    #     models.append(model)
    #     lr_schedulers.append(lr_scheduler)
    #     train_datasets.append(train_dataset)
    #     train_loaders.append(train_loader)

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

    

    interval = 1
    begin_interval = begin_epoch // interval
    end_interval = cfg.TRAIN.END_EPOCH // interval

    # initial multiprocess
    conn_pool = []
    process_pool = []
    for ith_Process in range(num_models):
        conn_pool.append(Pipe())
        args=(conn_pool[ith_Process][1], devices_per_model[ith_Process], interval, begin_epoch, all_best_perf, lr_scheduler, cfg, 
                train_dataset, valid_loader, valid_dataset, criterion, final_output_dir, 
                tb_log_dir, writer_dict)
        process_pool.append(Process(target=train_validate, args=args))
        process_pool[ith_Process].start()


    end = time.time()
    for ith_interval in range(begin_interval, end_interval):
        aug_prob_distribution = torch.nn.functional.softmax(theta)
        # send model, optimizer, dataset
        semantic_augmentation_idxes_pool = []
        for i in range(num_models):
            semantic_augmentation_idxes_pool.append(torch.torch.multinomial(aug_prob_distribution, len(train_dataset)*interval, replacement=True))
            conn_pool[i][0].send([model, optimizer, semantic_augmentation_idxes_pool[i], True])

        model_pool = []
        optimizer_pool = []
        perf_pool = []
        best_perf_pool = []
        for i in range(num_models):
            model, optimizer, perf_indicator, best_perf = conn_pool[i][0].recv()
            model_pool.append(model)
            optimizer_pool.append(optimizer)
            perf_pool.append(perf_indicator)
            best_perf_pool.append(best_perf)

        # calc loss to update Theta
        acc = torch.Tensor(perf_pool, dtype=torch.float32)
        acc = acc - acc.mean()
        P_tra_pool = torch.zeros(num_models, dtype=torch.float32, requires_grad=True)
        for i in range(num_models):
            idxes = semantic_augmentation_idxes_pool[i] 
            p = 1.
            for idx in idxes:
                p = p * aug_prob_distribution[idx]
            P_tra_pool[i] = p

        theta_loss = torch.log(P_tra_pool) * acc
        theta_loss = theta_loss.mean()

        # compute gradient and do update step
        theta_optimizer.zero_grad()
        theta_loss.backward()
        theta_optimizer.step()



        cur_best_perf = max(perf_pool)
        best_perf = max(best_perf_pool)
        model = model_pool[perf_pool.index(cur_best_perf)]
        optimizer = optimizer_pool[perf_pool.index(cur_best_perf)]

        if all_best_perf < cur_best_perf:
            all_best_perf = cur_best_perf
            best_model = True
        else:
            best_model = False
        
        # print
        msg = 'Outer loop: [{0}]\t' \
                  'Time {}s \t' \
                  'Loss {} \t' \
                  'current PCKh@0.5 {} \t' \
                  'best PCKh@0.5 {}'.format(
                      epoch, time.time()-end, theta_loss.numpy(), cur_best_perf, best_perf)
        logger.info(msg)


        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': ith_interval + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.state_dict(),
            'perf': cur_best_perf,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

def train_validate(conn, device_ids, interval, epoch, best_perf, lr_scheduler, cfg, 
                    train_dataset, valid_loader, valid_dataset, criterion, final_output_dir, 
                    tb_log_dir, writer_dict):
    while True:
        model, optimizer, semantic_augmentation_idxes, do_training = conn.recv()

        if not do_training:
            break

        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        
        for i in range(interval):

            if hasattr(train_dataset, 'aug_idxes'):
                train_dataset.aug_idxes = semantic_augmentation_idxes[i*len(train_dataset):(i+1)*len(train_dataset)]
            else:
                setattr(train_dataset, 'aug_idxes', semantic_augmentation_idxes[i*len(train_dataset):(i+1)*len(train_dataset)])

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

        conn.send([model.module(), optimizer, perf_indicator, best_perf])



if __name__ == '__main__':
    main()

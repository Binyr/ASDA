# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models

from core.evaluate import accuracy, calc_dists
from core.inference import get_final_preds, get_max_preds2
from utils.transforms import flip_back
from utils.vis import save_debug_images
from core.function import AverageMeter

import json
from tqdm import tqdm
import numpy as np


from utils.transforms import transform_preds
from core.function import _print_name_value

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
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)

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
    if args.model_file:
        cfg.defrost()
        cfg.TEST.MODEL_FILE = args.model_file
        cfg.freeze()

    if args.prevModelDir and args.modelDir:
        # copy pre models for philly
        copy_prev_models(args.prevModelDir, args.modelDir)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE)['state_dict'], strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file)['state_dict'])

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )


    scale_pyramid = np.arange(1.0, 1.1, 0.1)
    num_scale = len(scale_pyramid)
    num_stage = 1
    if not os.path.exists('all_preds.npy'):
        all_preds = []
        all_scales = []
        all_centers = []
        all_bboxes = []
        all_image_path = []
        for i in range(num_scale):
            valid_dataset.test_scale = scale_pyramid[i]

            valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
                shuffle=False,
                num_workers=cfg.WORKERS,
                pin_memory=True
            )
            preds, scales, centers, bboxes, image_path = calc_preds_before_warp(cfg, valid_loader, valid_dataset, model, criterion,
                     final_output_dir, tb_log_dir, num_stage=num_stage)
            all_preds.append(preds)
            all_scales.append(scales)
            all_centers.append(centers)
            all_bboxes.append(bboxes)
            all_image_path.append(image_path)
        all_preds = np.stack(all_preds, axis=0)
        all_scales = np.stack(all_scales, axis=0)
        all_centers = np.stack(all_centers, axis=0)
        all_bboxes = np.stack(all_bboxes, axis=0)

        np.save('all_preds.npy', all_preds)
        np.save('all_centers.npy', all_centers)
        np.save('all_scales.npy', all_scales)
        np.save('all_bboxes.npy', all_bboxes)
        np.save('all_image_path.npy', all_image_path)
    else:
        all_preds = np.load('all_preds.npy')
        all_scales = np.load('all_scales.npy')
        all_centers = np.load('all_centers.npy')
        all_bboxes = np.load('all_bboxes.npy')
        all_image_path = np.load('all_image_path.npy')

    num_scale, num_stage = all_preds.shape[:2]
    if not os.path.exists('best_txty.json'):
        # num_scale, num_scale = 1, 1
        result_to_print = []
        counter = 0
        for iscale in range(num_scale):
            for istage in range(num_stage):
                preds = all_preds[iscale, istage]
                scales = all_scales[iscale]
                centers = all_centers[iscale]
                #import pdb
                #pdb.set_trace()        
                # Transform back
                preds = preds[:,:,:2]

                coords_t = preds.copy()
                for i in range(coords_t.shape[0]):
                    coords_t[i] = transform_preds(
                        preds[i], centers[i], scales[i], [96, 96]
                    )
                name_values, perf_indicator = valid_dataset.evaluate(
                        cfg, coords_t, final_output_dir, all_bboxes, all_image_path[iscale])
                model_name = cfg.MODEL.NAME
                if isinstance(name_values, list):
                    for name_value in name_values:
                        _print_name_value(name_value, model_name)
                else:
                    _print_name_value(name_values, model_name)
                
                best_perf = perf_indicator
                best_name_values = name_values
                best_txty = np.zeros(2)
                result_to_print.append({'init_perf': float(round(perf_indicator, 3)),
                                        'scale': float(round(scale_pyramid[iscale], 3)),
                                        'stage': int(num_stage-istage)})
                
                for tx in np.arange(-0.5, 0.5, 0.05):
                    for ty in np.arange(-0.5, 0.5, 0.05):
                        counter += 1
                        if counter % 100 == 0:
                            print('{},{}'.format(tx, ty))
                        coords = preds[:,:,:2].copy()

                        coords += np.array([tx, ty])

                        coords_t  = coords.copy()

                        # Transform back
                        for i in range(coords.shape[0]):
                            coords_t[i] = transform_preds(
                                coords[i], centers[i], scales[i], [96, 96]
                            )
                        name_values, perf_indicator = valid_dataset.evaluate(
                        cfg, coords_t, '')



                        if best_perf < perf_indicator:
                            best_perf = perf_indicator
                            best_txty = np.array([tx, ty])
                            best_name_values = name_values
             
                            model_name = cfg.MODEL.NAME
                            if isinstance(name_values, list):
                                for name_value in name_values:
                                    _print_name_value(name_value, model_name)
                            else:
                                _print_name_value(name_values, model_name)
                result_to_print[-1].update({'best_perf': float(round(best_perf,3))})
                result_to_print[-1].update({'best_txty': np.round_(best_txty, 3).tolist()})

        with open('best_txty.json', 'w') as f:
            f.write(json.dumps(result_to_print))
    else:
        with open('best_txty.json', 'r') as f:
            result_to_print = json.load(f)

    idx = 0
    for i in range(num_scale):
        for j in range(num_stage):
            print('scale:{}, stage:{}, init_perf:{}, best_perf:{}, best_txty:{}'.format(result_to_print[idx]['scale'], 
                                                                        result_to_print[idx]['stage'],
                                                                        result_to_print[idx]['init_perf'],
                                                                        result_to_print[idx]['best_perf'],
                                                                        result_to_print[idx]['best_txty']))
            idx += 1

    
        
    # shift best tx ty
    idx = 0
    all_preds_after_warp = all_preds.copy()
    for iscale in range(num_scale):
        for istage in range(num_stage):
            preds = all_preds[iscale, istage]
            scales = all_scales[iscale]
            centers = all_centers[iscale]

            coords = preds[:,:,:2].copy()
            best_txty = np.array(result_to_print[idx]['best_txty'])
            coords += best_txty

            coords_t  = coords.copy()
            # Transform back
            for i in range(coords.shape[0]):
                coords_t[i] = transform_preds(
                    coords[i], centers[i], scales[i], [96, 96]
                )
            name_values, perf_indicator = valid_dataset.evaluate(
            cfg, coords_t, '')

            all_preds_after_warp[iscale, istage, :, :, :2] = coords_t
            idx += 1

    # merge preds
    all_preds_after_warp = all_preds_after_warp.transpose(0, 1, 4, 3, 2)
    num_scale, num_stage, d, J, N = all_preds_after_warp.shape
    # final_preds = all_preds_after_warp[0, 0]
    final_preds = merge_predicted_joints2(all_preds_after_warp.reshape(-1, d, J, N))

    final_preds = final_preds.transpose(2,1,0)

    # evaluate
    name_values, perf_indicator = valid_dataset.evaluate(
            cfg, final_preds, final_output_dir)
    model_name = cfg.MODEL.NAME
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(name_value, model_name)
    else:
        _print_name_value(name_values, model_name)




# -- merge the predicted joints in multiple scale
def merge_predicted_joints2(preJointStack):
    # count = 0
    numScale = len(preJointStack)
    _, numJoint, numImage = preJointStack[0].shape
    finalJoint_mean = np.zeros((3, numJoint, numImage))
    for ithImg in range(numImage):
        for ithJt in range(numJoint):
            maxvalHP_list = np.zeros((numScale))
            joint_list = np.zeros((numScale, 2))
            for ithScale in range(numScale):
                maxvalHP_list[ithScale] = preJointStack[ithScale][2, ithJt, ithImg]
                joint_list[ithScale, :] = preJointStack[ithScale][0:2, ithJt, ithImg]
            # -- find joint according to the mean
            # idx_vis = np.where(maxvalHP_list >= maxvalHP_list.mean(0, keepdims=False))[0]
            # maxvalHP_idx = maxvalHP_list[idx_vis]
            # joint_list_idx = joint_list[idx_vis, :]
            
            joints_mean = joint_list.mean(0, keepdims=False)
            joints_diff = np.linalg.norm(joint_list - joints_mean[np.newaxis, ...], axis=1, ord=1)
            # joints_diff = np.sum(np.abs(joint_list - np.tile(joints_mean, (len(joint_list), 1))), 1)
            # import pdb
            # pdb.set_trace()
            # count += 1
            # print(count)

            idx_diff = np.where(joints_diff <= joints_diff.mean(0, keepdims=False)*1.0)[0]
            if len(idx_diff)==0:
                idx_diff = np.argmin(joints_diff)[np.newaxis, ...]
            maxvalHP_idx = maxvalHP_list[idx_diff]
            joint_list_idx = joint_list[idx_diff, :]

            finalJoint_mean[2, ithJt, ithImg] = maxvalHP_idx.mean(0, keepdims=False)
            finalJoint_mean[0:2, ithJt, ithImg] = np.mean(joint_list_idx, 0)
    return finalJoint_mean

    
def calc_preds_before_warp(config, val_loader, val_dataset, model, criterion, output_dir,
         tb_log_dir, writer_dict=None, num_stage=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_stage, num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_joints = np.zeros(
        (num_stage, num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )

    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.float32)


    all_boxes = np.zeros((num_stage, num_samples, 6), dtype=np.float32)

    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        for i, (input, target, target_weight, meta) in enumerate(tqdm(val_loader)):
            # compute output
            outputs = [model(input)]
            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = [model(input_flipped)]


                outputs_flipped_np = []
                for o in outputs_flipped:
                    outputs_flipped_np.append(flip_back(o.cpu().numpy(), val_dataset.flip_pairs))
                

                outputs_flipped = []
                for o in outputs_flipped_np:
                    outputs_flipped.append(torch.from_numpy(o.copy()).cuda())

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    for o in outputs_flipped:
                        o[:, :, :, 1:] = \
                            o.clone()[:, :, :, 0:-1]
                #!!!!
                for io, o in enumerate(outputs_flipped):
                    outputs[io] = outputs[io] + outputs_flipped[io]
                # output = (output + output_flipped) * 0.5

            num_images = input.size(0)

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            for istage in range(num_stage):
                output = outputs[-istage-1]

                preds, maxvals = get_max_preds2(output.clone().cpu().numpy())
                # preds, maxvals = get_final_preds(
                #     config, output.clone().cpu().numpy(), c, s)

                all_preds[istage, idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[istage, idx:idx + num_images, :, 2:3] = maxvals

                # double check this all_boxes parts
                all_boxes[istage, idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[istage, idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[istage, idx:idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[istage, idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])


            scales[idx:idx + num_images] = s
            centers[idx:idx + num_images] = c

            idx += num_images

    return all_preds, scales, centers, all_boxes, image_path

def calc_dists_from_htm(output, target):
    pred_cord, _ = get_max_preds(output)
    target_cord, _ = get_max_preds(target)
    norm = np.ones((pred_cord.shape[0], 2))
    dists = calc_dists(pred_cord, target_cord, norm)
    return dists.transpose(1, 0)

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))
        loss = 0


        if self.use_target_weight:
            loss = 0.5 * self.criterion(
                heatmaps_pred.mul(target_weight),
                heatmaps_gt.mul(target_weight)
            )
        else:
            loss = 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        # import pdb
        # pdb.set_trace()
        return loss.mean(dim=2)

if __name__ == '__main__':
    main()
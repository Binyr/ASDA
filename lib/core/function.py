from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, train_loader, model, stn, criterion, optimizer, stn_optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, stn_first):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    stn_losses = AverageMeter()
    stn_acc = AverageMeter()
    # switch to train mode
    model.train()
    stn.train()
    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)


        def forward_net(input, target, target_weight, meta, stn, model, criterion):
            # stn
            part_imgs = meta['part_imgs']
            part_masks = meta['part_masks']
            init_thetas = meta['init_thetas']
            part_idxes = meta['part_idxes']
            input_D = stn(input, part_imgs, part_masks, part_idxes, init_thetas)

            # mean=[0.485, 0.456, 0.406]
            # std=[0.229, 0.224, 0.225]
            # III = input_D.detach().cpu().numpy()
            # for iI, I in enumerate(III):
            #     I = I.transpose(1,2,0)
            #     I = (I*np.array(std) + np.array(mean))*255
            #     I = I.astype(np.uint8)
            #     import cv2
            #     I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            #     cv2.imwrite('{}.jpg'.format(iI), I)
            #     if iI > 10:
            #         break
            # PPP = part_imgs.detach().cpu().numpy()[:, 0]
            # for iP, P in enumerate(PPP):
            #     P = P.transpose(1,2,0)
            #     P = (P*np.array(std) + np.array(mean))*255
            #     P = P.astype(np.uint8)
            #     import cv2
            #     P = cv2.cvtColor(P, cv2.COLOR_BGR2RGB)
            #     cv2.imwrite('part_{}.jpg'.format(iP), P)
            #     if iP > 10:
            #         break

            # assert 0
            # compute output
            outputs = model(input_D)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            if isinstance(outputs, list):
                loss = criterion(outputs[0], target, target_weight)
                for output in outputs[1:]:
                    loss += criterion(output, target, target_weight)
            else:
                output = outputs
                loss = criterion(output, target, target_weight)

            # prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            # save_debug_images(config, input_D, meta, target, meta['joints'][:,:,:2].cpu().numpy(), output,
            #                   prefix)


            return loss, output

        def optimize_D(loss, model, optimizer, output, target, losses, acc, input):
            # compute gradient and do update step
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)
        def optimize_G(loss, stn, stn_optimizer, output, target, stn_losses, stn_acc, input):
            # update stn
            stn.zero_grad()
            stn_optimizer.zero_grad()
            stn_loss = -loss
            stn_loss.backward()
            stn_optimizer.step()

            stn_losses.update(stn_loss.item(), input.size(0))

            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
            stn_acc.update(avg_acc, cnt)

        
        if stn_first:
            for kk in range(config.STN.NG):
                loss, output = forward_net(input, target, target_weight, meta, stn, model, criterion)
                optimize_G(loss, stn, stn_optimizer, output, target, stn_losses, stn_acc, input)
            for kk in range(config.STN.ND):
                loss, output = forward_net(input, target, target_weight, meta, stn, model, criterion)
                optimize_D(loss, model, optimizer, output, target, losses, acc, input)
        else:
            for kk in range(config.STN.ND):
                loss, output = forward_net(input, target, target_weight, meta, stn, model, criterion)
                optimize_D(loss, model, optimizer, output, target, losses, acc, input)
            for kk in range(config.STN.NG):
                loss, output = forward_net(input, target, target_weight, meta, stn, model, criterion)
                optimize_G(loss, stn, stn_optimizer, output, target, stn_losses, stn_acc, input)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})' \
                  'STN_Loss {stn_loss.val:.5f} ({stn_loss.avg:.5f})\t' \
                  'STN_Accuracy {stn_acc.val:.3f} ({stn_acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc, stn_loss=stn_losses, stn_acc=stn_acc)
            logger.info(msg)

        writer = writer_dict['writer']
        
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', losses.val, global_steps)
        writer.add_scalar('train_acc', acc.val, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1
    
        stn_global_steps = writer_dict['stn_train_global_steps']
        writer.add_scalar('stn_train_loss', stn_losses.val, stn_global_steps)
        writer.add_scalar('stn_train_acc', stn_acc.val, stn_global_steps)
        writer_dict['stn_train_global_steps'] = stn_global_steps + 1

        


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

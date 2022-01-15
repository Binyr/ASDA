# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torchvision

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class STN2(nn.Module):
    def __init__(self, num_parts):
        super(STN, self).__init__()
        self.num_parts = num_parts

        # Spatial transformer localization-network
        res50 = torchvision.models.resnet50(pretrained=True)
        layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1',
                 'layer2', 'layer3', 'layer4', 'avgpool']
        for layer in layers:
            setattr(self, layer, getattr(res50, layer))

        # Regressor for the 3 * 2 affine matrix
        self.fc_locs = nn.Sequential(
            nn.Conv2d(2048, 128*num_parts, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(128*num_parts, 3*2*num_parts, 1, 1, 0, groups=num_parts)
        ) 

        nn.init.constant_(self.fc_locs[2].weight, 0)
        nn.init.constant_(self.fc_locs[2].bias, 0)

    def forward(self, x, part_images, part_masks, part_idxes, init_thetas):
        # part_images (N, 8, 3, H, W)
        # x (N, 3, H, W)
        # part_idxes (N, 8), to select corresponding theta
        # joints_position (N, 8, 2, 3), init theta
        # part_masks (N, 8, 1, H, W)
        xs = self.conv1(x)
        xs = self.bn1(xs)
        xs = self.relu(xs)
        xs = self.maxpool(xs)
        xs = self.layer1(xs)
        xs = self.layer2(xs)
        xs = self.layer3(xs)
        xs = self.layer4(xs)
        xs = self.avgpool(xs)

        N, Np, Cp, Hp, Wp = part_images.size()
        # theta (N, num_parts, 6)
        thetas = self.fc_locs(xs)
        thetas = thetas.view(N, self.num_parts, 2, 3)
        t = []
        for n in range(part_idxes.size(0)):
            t.append(thetas[n, part_idxes[n]])
        thetas = torch.stack(t, dim=0)

        max_num_parts = part_idxes.size(1) 
        thetas = thetas.view(-1, 2, 3)

        init_thetas = init_thetas.view(-1, 2, 3).cuda()
        thetas = thetas + init_thetas

        
        part_images = part_images.view(-1, Cp, Hp, Wp)
        grid = nn.functional.affine_grid(thetas, torch.Size([N*Np, Cp, x.size(2), x.size(3)]))
        part_images = nn.functional.grid_sample(part_images, grid)

        part_masks = part_masks.view(-1, 1, Hp, Wp)
        grid_mask = nn.functional.affine_grid(thetas, torch.Size([N*Np, 1, x.size(2), x.size(3)]))
        part_masks = nn.functional.grid_sample(part_masks, grid_mask)

        part_images = part_images.view(N, Np, Cp, Hp, Wp)
        part_masks = part_masks.view(N, Np, 1 ,Hp, Wp)
        for i in range(part_images.size(1)):
            x = x * (1 - part_masks[:, i]) + part_images[:, i] * part_masks[:, i]
        return x

class STN(nn.Module):
    def __init__(self, num_parts):
        super(STN, self).__init__()
        self.num_parts = num_parts

        # Spatial transformer localization-network
        res50 = torchvision.models.resnet18(pretrained=True)
        layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1',
                 'layer2', 'layer3', 'layer4', 'avgpool']
        for layer in layers:
            setattr(self, layer, getattr(res50, layer))

        # Regressor for the 3 * 2 affine matrix
        self.fc_locs = nn.Sequential(
            nn.Conv2d(512, 128*num_parts, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(128*num_parts, 3*num_parts, 1, 1, 0, groups=num_parts),
            nn.Sigmoid()
        ) 

        nn.init.constant_(self.fc_locs[2].weight, 0)
        nn.init.constant_(self.fc_locs[2].bias, 0)
        self.pi = 3.1415926


    def forward(self, x, part_images, part_masks, part_idxes, init_thetas):
        # part_images (N, 8, 3, H, W)
        # x (N, 3, H, W)
        # part_idxes (N, 8), to select corresponding theta
        # joints_position (N, 8, 2, 3), init theta
        # part_masks (N, 8, 1, H, W)
        xs = self.conv1(x)
        xs = self.bn1(xs)
        xs = self.relu(xs)
        xs = self.maxpool(xs)
        xs = self.layer1(xs)
        xs = self.layer2(xs)
        xs = self.layer3(xs)
        xs = self.layer4(xs)
        xs = self.avgpool(xs)

        N, Np, Cp, Hp, Wp = part_images.size()
        # theta (N, num_parts, 6)
        rt = self.fc_locs(xs) - 0.5
        rt = rt.view(N, self.num_parts, 3)
        # thetas = torch.zeros(N, self.num_parts, 2, 3, dtype=torch.float32)
        # thetas = 
        # thetas = thetas.view(N, self.num_parts, 2, 3)
        t = []
        for n in range(part_idxes.size(0)):
            t.append(rt[n, part_idxes[n]])
        # rt (N, 8, 3)
        rt = torch.stack(t, dim=0)
        thetas = torch.zeros(N, Np, 2, 3, dtype=torch.float32).cuda()
        thetas[:,:,0,0] = torch.cos(rt[:,:,0]*self.pi)
        thetas[:,:,0,1] = torch.sin(rt[:,:,0]*self.pi)
        thetas[:,:,1,0] = -torch.sin(rt[:,:,0]*self.pi)
        thetas[:,:,1,1] = torch.cos(rt[:,:,0]*self.pi)
        thetas[:,:,0,2] = rt[:,:,1]*2.4
        thetas[:,:,1,2] = rt[:,:,2]*2.4
        thetas = thetas.reshape(-1, 2, 3)

        # init_thetas = init_thetas.view(-1, 2, 3).cuda()
        
        
        part_images = part_images.view(-1, Cp, Hp, Wp)
        grid = nn.functional.affine_grid(thetas, torch.Size([N*Np, Cp, x.size(2), x.size(3)]))
        part_images = nn.functional.grid_sample(part_images, grid)

        part_masks = part_masks.view(-1, 1, Hp, Wp)
        grid_mask = nn.functional.affine_grid(thetas, torch.Size([N*Np, 1, x.size(2), x.size(3)]))
        part_masks = nn.functional.grid_sample(part_masks, grid_mask)

        part_images = part_images.view(N, Np, Cp, Hp, Wp)
        part_masks = part_masks.view(N, Np, 1 ,Hp, Wp)
        for i in range(part_images.size(1)):
            x = x * (1 - part_masks[:, i]) + part_images[:, i] * part_masks[:, i]
        return x


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        super(PoseHighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x = self.final_layer(y_list[0])

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    if cfg.MODEL.FINETUNE:
        assert os.path.exists(cfg.MODEL.HTM_MODEL_PATH), \
        'HTM_MODEL_PATH: {} do not exist'.format(cfg.MODEL.HTM_MODEL_PATH)
        logger.info('=> loading model from {}'.format(cfg.MODEL.HTM_MODEL_PATH))
        pretrained_model = torch.load(cfg.MODEL.HTM_MODEL_PATH)
        if 'best_state_dict' in pretrained_model:
            model.load_state_dict(pretrained_model['best_state_dict'], strict=False)
        else:
            model.load_state_dict(pretrained_model, strict=False)

    return model


def get_stn_net(cfg, is_train, **kwargs):
    model = STN(num_parts=25)
    return model

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
## change in 2019.1.27
# compared to v2, just change config formate about fuse heatmap

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
import numpy as np

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
import copy

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

class Bottleneck_CAFFE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_CAFFE, self).__init__()
        # add stride to conv1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
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

class conv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, bn=True, relu=True ):
        super(conv, self).__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                 dilation=dilation, groups=groups, bias=bias)
        nn.init.normal_(self.c.weight, std=0.001)
        self.bn = None
        self.relu = None

        if bn:
            self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        if relu:
            self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.c(x)
        if not self.bn == None:
            x = self.bn(x)
        if not self.relu == None:
            x = self.relu(x)
        return  x

class deconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, bn=True, relu=True):
        super(deconv, self).__init__()
        self.dc = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                 output_padding=output_padding, groups=groups, bias=bias)
        nn.init.normal_(self.dc.weight, std=0.001)
        self.bn = None
        self.relu = None

        if bn:
            self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        if relu:
            self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dc(x)
        if not self.bn == None:
            x = self.bn(x)
        if not self.relu == None:
            x = self.relu(x)
        return x

def get_A1(adjacent_connection):
    N = len(adjacent_connection)
    A = np.eye(N)
    D = np.zeros((N,N))
    # import pdb
    # pdb.set_trace()
    for i in range(N):
        for adj in adjacent_connection[i]:
            A[i, adj] = 1
        D[i,i] = (A[i].sum())**-0.5
    # import pdb
    # pdb.set_trace()
    # D1 = D**-0.5
    A1 = np.matmul(np.matmul(D,A),D)
    # import pdb
    # pdb.set_trace()
    return A1

class PNonlocal(nn.Module):
    def __init__(self, node_depth, num_node, adjs, depth, embed=True,
                 maxpool=2, residual=False, er=True, ebr=True, wzr=True):
        super(PNonlocal, self).__init__()
        self.residual = residual
        self.node_depth = node_depth
        self.num_node = num_node
        self.A = torch.nn.Parameter(torch.FloatTensor(get_A1(adjs)), requires_grad=False)
        self.adjs = copy.deepcopy(adjs)
        for i in range(len(self.adjs)):
            self.adjs[i].append(i)
            self.adjs[i].sort()

        self.embed = embed
        self.maxpool = None
        self.upsample = None

        inchannel = node_depth * num_node
        outchannel = depth * num_node
        if embed == True:
            self.embed_a = conv(inchannel, outchannel, 3, stride=1, padding=1, groups=num_node,bn=ebr,relu=er)
            self.embed_b = conv(inchannel, outchannel, 3, stride=1, padding=1, groups=num_node,bn=ebr,relu=er)
            self.embed_c = conv(inchannel, outchannel, 3, stride=1, padding=1, groups=num_node,bn=ebr,relu=er)

        if embed == True:
            self.upembed = conv(outchannel, inchannel, 3, stride=1, padding=1, groups=num_node, relu=wzr)
        else:
            self.upembed = conv(inchannel, inchannel, 3, stride=1, padding=1, groups=num_node, relu=wzr)

        if maxpool is not False and maxpool > 1:
            self.maxpool = nn.MaxPool2d(maxpool, maxpool)
            self.upsample = nn.Upsample(scale_factor=maxpool, mode='bilinear')



    def forward(self, x):
        x1 = x
        x2 = x
        x3 = x

        if not self.maxpool == None:
            x1 = self.maxpool(x1)
            x2 = self.maxpool(x2)
            x3 = self.maxpool(x3)

        if self.embed:
            x1 = self.embed_a(x1)
            x2 = self.embed_b(x2)
            x3 = self.embed_c(x3)



        n, c, h, w = x1.size()

        a = x1.reshape(n, self.num_node, -1, h*w)
        b = x2.reshape(n, self.num_node, -1, h*w)
        g = x3.reshape(n, self.num_node, -1, h*w)

        k = torch.matmul(g, b.permute(0,1,3,2)) #(n, node, c, c)
        k = k / (h*w + 0.00001)

        ksize = k.size()
        k = k.view(ksize[0], ksize[1], -1)
        k = torch.matmul(self.A, k)
        k = k.view(*ksize)

        a = torch.matmul(k, a)
        a = a.reshape(n,c,h,w)

        if not self.upsample == None:
            a = self.upsample(a)

        a = self.upembed(a)
        if self.residual:
            out = x + a
        else:
            out = a
        return out


class PNonlocal_multihead(nn.Module):
    def __init__(self, node_depth, num_node, adjs, depth, embed=True,
                 maxpool=2, residual=False, num_head=1, er=True, ebr=True, wzr=True):
        super(PNonlocal_multihead, self).__init__()
        self.num_head = num_head
        self.num_node = num_node
        self.node_depth = node_depth
        inchannel = num_node * node_depth
        self.layers = conv(inchannel, inchannel, 3, stride=1, padding=1, groups=num_node)
        self.pnonlocal_layers = nn.ModuleList([PNonlocal(node_depth//num_head, num_node, adjs, depth,
                                          embed, maxpool, residual, er, ebr, wzr) for i in range(num_head)])
        print(er, ebr, wzr)
    def forward(self, x):
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        # x = self.layers(x)
        x = x.view(N, self.num_node, self.num_head, self.node_depth // self.num_head, H, W)
        ll = []
        for hhh in range(self.num_head):
            x_head = x[:, :, hhh]
            x_head = x_head.reshape(N, -1, H, W)
            x_head = self.pnonlocal_layers[hhh](x_head)
            ll.append(x_head.reshape(N, self.num_node, self.node_depth // self.num_head, H, W))
        x = torch.cat(ll, dim=2)
        x = x.reshape(N, C, H, W)
        return x


# class GAT_multihead_nonlocal(nn.Module):
#     def __init__(self, node_depth, num_node, num_layers, adjs, num_head, gat_act_func, nlconfig):
#         super(GAT_multihead_nonlocal, self).__init__()
#         self.node_depth = node_depth
#         self.num_node = num_node
#         self.num_layers = num_layers
#         self.num_head = num_head
#         # self.A = torch.nn.Parameter(torch.FloatTensor(get_A1(adjs)), requires_grad=False)
#         if gat_act_func == 'sigmoid':
#             activation_func = nn.Sigmoid
#         elif gat_act_func == 'leakyrelu':
#             activation_func = lambda x=0.2:nn.LeakyReLU(x)
#         elif gat_act_func == 'relu':
#             activation_func = lambda x=True:nn.ReLU(x)
#
#         self.adjs = copy.deepcopy(adjs)
#         for i in range(len(self.adjs)):
#             self.adjs[i].append(i)
#             self.adjs[i].sort()
#
#         self.num_edge = 0
#         for i in self.adjs:
#             self.num_edge += len(i)
#
#         self.att_weight = nn.ModuleList([nn.ModuleList([nn.Sequential(
#                                         nn.Conv2d(self.num_edge*node_depth*2//num_head, self.num_edge, 1 ,
#                                                   stride=1,padding=0, groups=self.num_edge),
#                                         activation_func()) for i in range(num_head)]) for ii in range(num_layers)])
#
#
#         in_channels = out_channels = node_depth*num_node
#         self.layers = nn.ModuleList([conv(in_channels, out_channels,
#                                                3, stride=1, padding=1,relu=False,
#                                                groups=num_node) for ii in range(num_layers)])
#         self.nonlocal_layers = nn.ModuleList([PNonlocal(node_depth, num_node, adjs,
#                                                         nlconfig.DEPTH, nlconfig.EMBED, nlconfig.POOL,nlconfig.RESIDUAL)
#                                               for s in range(num_layers)])
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward_gat(self, x, layer_index, head_index):
#         N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
#
#         # avg = x.mean(dim=(2, 3), keepdim=True)
#         x = x.view(N, self.num_node, self.node_depth//self.num_head, H, W)
#         avg = x
#
#         ll = []
#         for eee in range(len(self.adjs)):
#             for fff in self.adjs[eee]:
#                 ll.append(torch.cat((avg[:, eee], avg[:, fff]), dim=1))
#         all_pair = torch.cat(ll, dim=1)
#         att_weight = self.att_weight[layer_index][head_index](all_pair) #(N, num_edge, H, W)
#
#         # A = torch.zeros(N, self.num_node, self.num_node, H, W)
#         # if att_weight.device.type == 'cuda':
#         #     A = A.cuda(att_weight.device)
#
#
#         out = x.clone().detach()
#         count = 0
#         for eee in range(len(self.adjs)):
#             ll = []
#             tmp_f = 0
#             for fff in self.adjs[eee]:
#                 # import pdb
#                 # pdb.set_trace()
#                 tmp_f = tmp_f + x[:, fff] * att_weight[:, count].view(N,1,H,W)
#                 count += 1
#             out[:, eee] = tmp_f
#
#         # import pdb
#         # pdb.set_trace()
#         # A = A.view(N, self.num_node, self.num_node, 1, H, W).expand(N, self.num_node, self.num_node, self.node_depth//self.num_head, H, W)
#         # x = x.view(N, 1, self.num_node, self.node_depth//self.num_head, H, W)
#         # x = x.expand_as(A)
#         # x =  (A * x).mean(dim=2)
#         x = out.reshape(N,C,H,W)
#         return  x
#
#
#     def forward(self, x):
#         N,C,H,W = x.size(0), x.size(1), x.size(2), x.size(3)
#         for i in range(self.num_layers):
#             nonlocal_x = self.nonlocal_layers[i](x)
#             x = self.layers[i](x)
#             x = x.view(N, self.num_node, self.num_head, self.node_depth//self.num_head, H, W)
#             ll = []
#             # import pdb
#             # pdb.set_trace()
#             for hhh in range(self.num_head):
#                 x_head = x[:, :, hhh]
#                 x_head = x_head.reshape(N, -1, H, W)
#                 x_head = self.forward_gat(x_head, i, hhh)
#                 ll.append(x_head.reshape(N, self.num_node, self.node_depth//self.num_head, H, W))
#             x = torch.cat(ll, dim=2)
#             x = x.reshape(N,C,H,W)
#             x = x + nonlocal_x
#         return x

class GAT_multihead_nonlocal(nn.Module):
    def __init__(self, node_depth, num_node, num_layers, adjs, num_head, gat_act_func, nlconfig):
        super(GAT_multihead_nonlocal, self).__init__()
        self.node_depth = node_depth
        self.num_node = num_node
        self.num_layers = num_layers
        self.num_head = num_head
        self.fuse_manner = nlconfig.FUSE
        # self.A = torch.nn.Parameter(torch.FloatTensor(get_A1(adjs)), requires_grad=False)
        if gat_act_func == 'sigmoid':
            activation_func = nn.Sigmoid
        elif gat_act_func == 'leakyrelu':
            activation_func = lambda x=0.2:nn.LeakyReLU(x)
        elif gat_act_func == 'relu':
            activation_func = lambda x=True:nn.ReLU(x)

        max_num_edge = 0
        for i in range(len(adjs)):
            n = len(adjs[i])
            if max_num_edge < n:
                max_num_edge = n
        max_num_edge = 2*max_num_edge + 1
        self.max_num_edge = max_num_edge

        self.adjs = copy.deepcopy(adjs)
        for i in range(len(self.adjs)):
            self.adjs[i].append(i)
            self.adjs[i].sort()

        max_num_edge = 0
        for i in range(len(self.adjs)):
            n = len(self.adjs[i])
            if max_num_edge < n:
                max_num_edge = n
        max_num_edge = 2 * max_num_edge
        self.max_num_edge = max_num_edge

        self.num_edge = 0
        for i in self.adjs:
            self.num_edge += len(i)

        self.pre_att_weight = nn.ModuleList([nn.ModuleList([nn.Conv2d(num_node*node_depth//num_head, num_node*max_num_edge, 1 ,
                                                  stride=1,padding=0, groups=num_node) for i in range(num_head)]) for ii in range(num_layers)])
        self.att_activation_func = nn.ModuleList([nn.ModuleList([activation_func() for i in range(num_head)]) for ii in range(num_layers)])

        in_channels = out_channels = node_depth*num_node
        self.layers = nn.ModuleList([conv(in_channels, out_channels,
                                               3, stride=1, padding=1,relu=False,
                                               groups=num_node) for ii in range(num_layers)])
        self.relu = nn.ReLU(inplace=True)

        self.nonlocal_layers = nn.ModuleList([PNonlocal_multihead(node_depth, num_node, adjs,
                                                                  nlconfig.DEPTH, nlconfig.EMBED, nlconfig.POOL,
                                                                  nlconfig.RESIDUAL, nlconfig.NUM_HEAD,
                                                                  er=nlconfig.ER,
                                                                  ebr=nlconfig.EBR,
                                                                  wzr=nlconfig.WZR)
                                              for s in range(num_layers)])

        if nlconfig.FUSE == 'cat':
            self.fuse_layer = conv(in_channels*2, in_channels, 1, 1, 0, groups=num_node)
        elif nlconfig.FUSE == 'add':
            self.fuse_layer = None
        elif nlconfig.FUSE == 'att':
            self.fuse_layer = nn.Sequential(conv(in_channels*2, num_node, 1, 1, 0, groups=num_node,relu=False,bn=False),
                                                           nn.Sigmoid())

    def forward_gat(self, x, layer_index, head_index):
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)


        #generate attention
        att_weight = []
        pre_att_weight = self.pre_att_weight[layer_index][head_index](x) #(n, 16*9, h, w)
        pre_att_weight = pre_att_weight.view(N, self.num_node, self.max_num_edge, H, W)
        node_index = torch.zeros(self.num_node,dtype=torch.long)
        for eee in range(len(self.adjs)):
            for fff in self.adjs[eee]:
                if eee == fff:
                    t = pre_att_weight[:, eee, node_index[eee]] + pre_att_weight[:, fff, node_index[fff]+1]
                else:
                    t = pre_att_weight[:, eee, node_index[eee]] + pre_att_weight[:, fff, node_index[fff]]
                t = self.att_activation_func[layer_index][head_index](t)
                node_index[eee] = node_index[eee] + 1
                node_index[fff] = node_index[fff] + 1
                att_weight.append(t)


        # avg = x.mean(dim=(2, 3), keepdim=True)
        x = x.view(N, self.num_node, self.node_depth//self.num_head, H, W)

        out = x.clone().detach()
        count = 0
        for eee in range(len(self.adjs)):
            tmp_f = 0
            for fff in self.adjs[eee]:
                # import pdb
                # pdb.set_trace()
                tmp_f = tmp_f + x[:, fff] * att_weight[count].view(N,1,H,W)
                count += 1
            out[:, eee] = tmp_f

        x = out.reshape(N,C,H,W)
        # x = self.relu(x)
        return  x


    def forward(self, x):
        N,C,H,W = x.size(0), x.size(1), x.size(2), x.size(3)
        nonlocal_x = x
        for i in range(self.num_layers):
            nonlocal_x = self.nonlocal_layers[i](nonlocal_x)

            x = self.layers[i](x)
            x = x.view(N, self.num_node, self.num_head, self.node_depth//self.num_head, H, W)
            ll = []
            for hhh in range(self.num_head):
                x_head = x[:, :, hhh]
                x_head = x_head.reshape(N, -1, H, W)
                x_head = self.forward_gat(x_head, i, hhh)
                ll.append(x_head.reshape(N, self.num_node, self.node_depth//self.num_head, H, W))
            x = torch.cat(ll, dim=2)
            x = x.reshape(N,C,H,W)

        # if self.fuse_manner == 'cat':
        #     x = x.view(N, self.num_node, self.node_depth, H, W)
        #     nonlocal_x = nonlocal_x.view(N, self.num_node, self.node_depth, H, W)
        #     x = torch.cat((x, nonlocal_x), dim=2)
        #     x = x.reshape(N, 2*C, H, W)
        #     x = self.fuse_layer(x)
        # elif self.fuse_manner == 'add':
        #     x = x + nonlocal_x
        # elif self.fuse_manner == 'att':
        #     nonlocal_x = nonlocal_x.view(N, self.num_node, self.node_depth, H, W)
        #     # ll.append(nonlocal_x)
        #     local_x = x.view(N, self.num_node, self.node_depth, H, W)
        #     x = torch.cat((nonlocal_x, local_x), dim=2)
        #     x = x.reshape(N, 2 * C, H, W)
        #     att = self.fuse_layer(x)
        #     att = att.view(N, self.num_node, 1, H, W)
        #     # nonlocal_x = nonlocal_x.view(N, C, H, W)
        #     # local_x = local_x.view(N, C, H, W)
        #     x = local_x * att + (1 - att) * nonlocal_x
        #     x = x.reshape(N, C, H, W)
        return [x, nonlocal_x]

class Gcn(nn.Module):
    def __init__(self, node_depth, num_node, num_layers, adjs):
        super(Gcn, self).__init__()
        self.node_depth = node_depth
        self.num_node = num_node
        self.num_layers = num_layers
        self.A = torch.nn.Parameter(torch.FloatTensor(get_A1(adjs)), requires_grad=False)
        in_channels = out_channels = node_depth*num_node
        self.layers = nn.ModuleList([conv(in_channels, out_channels,
                                               3, stride=1, padding=1,relu=False,
                                               groups=num_node) for i in range(num_layers)])
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            size = x.size()
            x = x.view(size[0], self.num_node, -1)
            x = torch.matmul(self.A, x)
            x = self.relu(x)
            x = x.view(*size)
        # import pdb
        # pdb.set_trace()
        return x

class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.cfg = cfg
        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        laterals_inchannel = [4*i for i in [64, 128, 256, 512]]
        self.lateral_layers = nn.ModuleList([conv(in_channels=c,out_channels=256,
                                            kernel_size=1,stride=1,padding=0) for c in laterals_inchannel])

        # kernel, padding, output_padding = self._get_deconv_cfg(4, 0)
        hm_size = (np.array(cfg.MODEL.HEATMAP_SIZE) // 8).astype(np.int32).tolist()
        self.deconv_layers = nn.ModuleList(
            [nn.Sequential(nn.Upsample(size=(hm_size[1] * 2 ** (i + 1), hm_size[0] * 2 ** (i + 1))),
                           conv(in_channels=256, out_channels=256,
                                kernel_size=1, stride=1, padding=0,
                                bn=False, relu=False)
                           ) for i in range(3)])



        self.gen_node_fms = conv(
            in_channels=256,
            out_channels=cfg.MODEL.NUM_JOINTS * cfg.GCN.NODE_DEPTH,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.gcn_layer = GAT_multihead_nonlocal(cfg.GCN.NODE_DEPTH, cfg.GCN.NUM_NODE, cfg.GCN.NUM_LAYERS,
                                                cfg.GCN.A,cfg.GCN.NUM_HEAD, cfg.GCN.GAT_ACT_FUNC,
                                                cfg.NON_LOCAL)
        # self.final_layer = nn.Conv2d(
        #     in_channels=extra.NUM_DECONV_FILTERS[-1],
        #     out_channels=cfg.MODEL.NUM_JOINTS,
        #     kernel_size=extra.FINAL_CONV_KERNEL,
        #     stride=1,
        #     padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        # )
        self.fuse_manner = cfg.FUSE.MANNER
        print(self.fuse_manner)
        if self.fuse_manner == 'add':
            self.fuse_layer = None
        elif self.fuse_manner == 'cat':
            in_channels = cfg.GCN.NUM_NODE*2
            ksize = cfg.FUSE.CAT_KSIZE
            self.fuse_layer = conv(in_channels, cfg.GCN.NUM_NODE, ksize,stride=1,padding=(ksize-1)//2,
                                   groups=cfg.GCN.NUM_NODE,relu=False,bn=False)
        elif self.fuse_manner == 'add2':
            in_channels = cfg.GCN.NODE_DEPTH * cfg.GCN.NUM_NODE
            self.j2c1 = conv(cfg.GCN.NUM_NODE, in_channels, 1,stride=1,padding=0)
            self.j2c2 = conv(cfg.GCN.NUM_NODE, in_channels, 1, stride=1, padding=0)

            self.final_layer = conv(in_channels=in_channels, out_channels=cfg.MODEL.NUM_JOINTS,
                                    kernel_size=3, stride=1,
                                    padding=1,
                                    bn=False, relu=False)
        elif self.fuse_manner == 'add3':
            in_channels = cfg.GCN.NODE_DEPTH * cfg.GCN.NUM_NODE
            ksize = cfg.FUSE.ADD3_KSIZE
            self.j2c1 = conv(cfg.GCN.NUM_NODE, in_channels, ksize, stride=1, padding=(ksize-1)//2,groups=cfg.GCN.NUM_NODE)
            self.j2c2 = conv(cfg.GCN.NUM_NODE, in_channels, ksize, stride=1, padding=(ksize-1)//2,groups=cfg.GCN.NUM_NODE)

            self.final_layer = conv(in_channels=in_channels, out_channels=cfg.MODEL.NUM_JOINTS,
                                    kernel_size=3, stride=1,
                                    padding=1, groups=cfg.GCN.NUM_NODE,
                                    bn=False, relu=False)
        elif self.fuse_manner == 'att1':
            in_channels = cfg.GCN.NUM_NODE * 2
            self.fuse_layer = nn.Sequential(conv(in_channels, cfg.GCN.NUM_NODE, 1, stride=1, padding=0,
                                                 groups=cfg.GCN.NUM_NODE, relu=False, bn=False),
                                            nn.Sigmoid())
        elif self.fuse_manner == 'att2':
            in_channels = cfg.GCN.NUM_NODE * (2+cfg.GCN.NODE_DEPTH)
            mid_channels = cfg.GCN.NODE_DEPTH * cfg.GCN.NUM_NODE
            self.fuse_layer = nn.Sequential(conv(in_channels, mid_channels, 3, stride=1, padding=1,
                                                 groups=cfg.GCN.NUM_NODE),
                                            conv(mid_channels, mid_channels, 3, stride=1, padding=1,
                                                 groups=cfg.GCN.NUM_NODE),
                                            conv(mid_channels, cfg.GCN.NUM_NODE, 1, stride=1, padding=0,
                                                 groups=cfg.GCN.NUM_NODE,relu=False,bn=False),
                                            nn.Sigmoid()
                                            )


        in_channels = cfg.GCN.NODE_DEPTH * cfg.GCN.NUM_NODE
        self.final_layer1 = nn.Sequential(conv(in_channels=in_channels, out_channels=in_channels,
                                               kernel_size=extra.FINAL_CONV_KERNEL, stride=1, groups=cfg.GCN.NUM_NODE,
                                               padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0),
                                          conv(in_channels=in_channels, out_channels=cfg.MODEL.NUM_JOINTS,
                                               kernel_size=3, stride=1,
                                               padding=1, groups=cfg.GCN.NUM_NODE,
                                               bn=False, relu=False)
                                          )
        self.final_layer2 = nn.Sequential(conv(in_channels=in_channels, out_channels=in_channels,
                                               kernel_size=extra.FINAL_CONV_KERNEL, stride=1, groups=cfg.GCN.NUM_NODE,
                                               padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0),
                                          conv(in_channels=in_channels, out_channels=cfg.MODEL.NUM_JOINTS,
                                               kernel_size=3, stride=1,
                                               padding=1, groups=cfg.GCN.NUM_NODE,
                                               bn=False, relu=False)
                                          )


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        laterals = []
        x = self.layer1(x)
        laterals.append(self.lateral_layers[0](x))
        x = self.layer2(x)
        laterals.append(self.lateral_layers[1](x))
        x = self.layer3(x)
        laterals.append(self.lateral_layers[2](x))
        x = self.layer4(x)
        laterals.append(self.lateral_layers[3](x))

        last_fm = None
        for i, block in enumerate(reversed(laterals)):
            if i == 0:
                last_fm = block
                # fms.append(last_fm)
                # out = self.out_layers[i](last_fm)
                # outs.append(self.upsample_layers(out))

            else:
                upsample = self.deconv_layers[i - 1](last_fm)
                last_fm = block + upsample
                # fms.append(last_fm)
                # out = self.out_layers[i](last_fm)
                # outs.append(self.upsample_layers(out))

        node_fms = self.gen_node_fms(last_fm)
        N, C, H, W = node_fms.size()
        local_x, nonlocal_x = self.gcn_layer(node_fms)
        local_out = self.final_layer1(local_x)
        nonlocal_out = self.final_layer2(nonlocal_x)

        if self.fuse_manner == 'add':
            out = local_out + nonlocal_out
        elif self.fuse_manner == 'cat':
            # N, C, H, W = node_fms.size()
            local_out = local_out.view(N, self.cfg.GCN.NUM_NODE, -1, H, W)
            nonlocal_out = nonlocal_out.view(N, self.cfg.GCN.NUM_NODE, -1, H, W)
            x = torch.cat((local_out, nonlocal_out),dim=2)
            x = x.view(N, -1, H, W)
            out = self.fuse_layer(x)
        elif self.fuse_manner == 'att1':
            # N, C, H, W = node_fms.size()
            node_fms = node_fms.view(N, self.cfg.GCN.NUM_NODE, -1, H, W)
            local_out = local_out.view(N, self.cfg.GCN.NUM_NODE, -1, H, W)
            nonlocal_out = nonlocal_out.view(N, self.cfg.GCN.NUM_NODE, -1, H, W)
            x = torch.cat((local_out, nonlocal_out), dim=2)
            x = x.reshape(N, -1, H, W)
            att = self.fuse_layer(x)
            att = att.view(N, self.cfg.GCN.NUM_NODE, 1, H, W)
            out = local_out * att + (1 - att)*nonlocal_out
            out = out.view(N, -1, H, W)
        elif self.fuse_manner == 'att2':
            # N, C, H, W = node_fms.size()
            node_fms = node_fms.view(N, self.cfg.GCN.NUM_NODE, -1, H, W)
            local_out = local_out.view(N, self.cfg.GCN.NUM_NODE, -1, H, W)
            nonlocal_out = nonlocal_out.view(N, self.cfg.GCN.NUM_NODE, -1, H, W)
            x = torch.cat((node_fms, local_out, nonlocal_out), dim=2)
            x = x.reshape(N, -1, H, W)
            att = self.fuse_layer(x)
            att = att.view(N, self.cfg.GCN.NUM_NODE, 1, H, W)
            out = local_out * att + (1 - att) * nonlocal_out
            out = out.view(N, -1, H, W)
        elif self.fuse_manner == 'add2' or self.fuse_manner == 'add3':
            x = self.j2c1(local_out) + self.j2c2(nonlocal_out)
            out = self.final_layer(x)


        local_out = local_out.view(N, -1, H, W)
        nonlocal_out = nonlocal_out.view(N, -1, H, W)
        # out = self.final_layer(x)

        return [out, local_out, nonlocal_out]

    def init_weights(self, pretrained=''):
        # import pdb
        # pdb.set_trace()
        if os.path.isfile(pretrained):
        #     logger.info('=> init deconv weights from normal distribution')
        #     for deconv in self.deconv_layers:
        #         for name, m in deconv.named_modules():
        #             if isinstance(m, nn.ConvTranspose2d):
        #                 logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
        #                 logger.info('=> init {}.bias as 0'.format(name))
        #                 nn.init.normal_(m.weight, std=0.001)
        #                 if self.deconv_with_bias:
        #                     nn.init.constant_(m.bias, 0)
        #             elif isinstance(m, nn.BatchNorm2d):
        #                 logger.info('=> init {}.weight as 1'.format(name))
        #                 logger.info('=> init {}.bias as 0'.format(name))
        #                 nn.init.constant_(m.weight, 1)
        #                 nn.init.constant_(m.bias, 0)
        #     logger.info('=> init final conv weights from normal distribution')
        #     for m in self.final_layer.modules():
        #         if isinstance(m, nn.Conv2d):
        #             # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #             logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
        #             logger.info('=> init {}.bias as 0'.format(name))
        #             nn.init.normal_(m.weight, std=0.001)
        #             nn.init.constant_(m.bias, 0)

            # pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))
            # import pdb
            # pdb.set_trace()
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}




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

def get_stn_net(cfg, is_train, **kwargs):
    model = STN(num_parts=25)
    return model

def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    #style = cfg.MODEL.STYLE

    block_class, layers = resnet_spec[num_layers]

    #if style == 'caffe':
    #    block_class = Bottleneck_CAFFE

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
        pass

    return model
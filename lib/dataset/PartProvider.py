import numpy as np
import os
from tqdm import tqdm
import random
import math
import pdb
import cv2
import csv
import json
from PIL import Image, ImageDraw
from collections import defaultdict

from utils.transforms import get_affine_transform

class PartProvider():
    def __init__(self, cfg, scale_range, part_image_size=[256, 256], num_parts_per_augmentation = [0,2,4,6,8], num_joints_SA=-1):
        AUG_PART = ['Left-arm', 'Right-arm', 'Left-leg', 'Right-leg','Upper-body','Upper-body-with-head', 
                    'Lower_body', 'Left-leg-with-shoe', 'Right-leg-with-shoe', 'All']
        HEAD_PART = ['Hat', 'Hair', 'Sunglasses', 'face', 'Head']
        CLOTHES = ['UpperClothes', 'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Left-shoe', 'Right-shoe', ]
        self.candidate_parts = AUG_PART + HEAD_PART + CLOTHES
        self.part_to_joint_id = {
            'Left-arm': [14,15,16],
            'Right-arm': [11,12,13],
            'Left-leg': [4,5,6],
            'Right-leg': [1,2,3],
            'Hat': [9,10],
            'Hair': [9,10],
            'Sunglasses': [9,10],
            'face': [9,10],
            'UpperClothes': [7,8,9],
            'Dress': [7,8],
            'Coat': [7,8,9],
            'Socks': [1,2,5,6],
            'Pants': [2,3,7,4,5],
            'Jumpsuits': [1,2,3,4,5,6,7,8],
            'Scarf': [9,10],
            'Skirt': [3,7,4],
            'Left-shoe': [5,6],
            'Right-shoe': [1,2],
            'Head': [9, 10],
            'Upper-body': [7, 8, 9, 11, 12, 13, 14, 15, 16],
            'Upper-body-with-head': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            'Lower_body': [1,2,3,4,5,6],
            'Left-leg-with-shoe': [4,5,6],
            'Right-leg-with-shoe': [1,2,3],
            'All': list(range(1, 17))
        }

        self.part_scale = {
            'Left-arm': 0.3,
            'Right-arm': 0.3,
            'Left-leg': 0.3,
            'Right-leg': 0.3,
            'Hat': 0.2,
            'Hair': 0.2,
            'Sunglasses': 0.1,
            'face': 0.2,
            'UpperClothes': 0.3,
            'Dress': 0.3,
            'Coat': 0.3,
            'Socks': 0.1,
            'Pants': 0.4,
            'Jumpsuits': 0.7,
            'Scarf': 0.1,
            'Skirt': 0.1,
            'Left-shoe': 0.1,
            'Right-shoe': 0.1,
            'Head': 0.2,
            'Upper-body': 0.4,
            'Upper-body-with-head': 0.5,
            'Lower_body': 0.5,
            'Left-leg-with-shoe': 0.4,
            'Right-leg-with-shoe': 0.4,
            'All': 0.9,
        }

        part_annot_file = cfg.ASA.PART_ANN_FILE
        self.part_root_dir = cfg.ASA.PART_ROOT_DIR
        with open(part_annot_file, 'r') as f:
            self.part_anns = json.load(f)

        self.delete_invalid_part()

        self.erode_kernel = cfg.ASA.ERODE_KERNEL
        self.gaussian_kernel = cfg.ASA.GAUSSIAN_KERNEL

        # self.scale_factor =  scale_factor
        # self.num_augmentation = num_augmentation
        # self.vertical_translate = vertical_translate
        # self.horizontal_translate = horizontal_translate
        # self.
        self.num_imgs_cannot_aug = 0
        self.part_image_size = np.array(part_image_size) #np.array([256, 256])

        self.num_parts_per_augmentation = num_parts_per_augmentation
        self.scale_range =  scale_range
        self.num_joints_SA = num_joints_SA

    def delete_invalid_part(self):
        print('filter invalid anns')
        valid_part_anns = defaultdict(list)
        for k, v in self.part_anns.items():
            check_dir = os.path.join(self.part_root_dir, k)
            file_list = os.listdir(check_dir)
            for ann in v:
                if ann['file_name'] not in file_list:
                    continue
                valid_part_anns[k].append(ann)
            print('{} done'.format(k))
        self.part_anns = valid_part_anns

    def sample_anchor_joint(self, joints, candidate_anchor_joint_ids):
        valid_index = []
        for i in candidate_anchor_joint_ids:
            if joints[i, 2] > 0:
                valid_index.append(i)
        if len(valid_index) == 0:
            return None

        t = random.sample(valid_index, 1)[0]
        return t

    def adjust_scale_by_diagnal(self, part_img, part_joints, body_bbox, target_scale, scale_factor):
        part_H, part_W = part_img.shape[:2]
        part_dia = np.linalg.norm([part_H, part_W])

        body_H, body_W = body_bbox[3] - body_bbox[1], body_bbox[2] - body_bbox[0]
        body_dia = np.linalg.norm([body_H, body_W])

        s = body_dia * target_scale / part_dia
        # random scale
        s = s * scale_factor

        part_img = cv2.resize(part_img, (int(part_W*s), int(part_H*s)))
        part_joints = part_joints * s
        return part_img, part_joints

    def generate_part(self, ann, body_bbox, do_SA=True):
        max_num_augmentation = max(self.num_parts_per_augmentation)
        part_imgs = np.zeros((max_num_augmentation, self.part_image_size[1], self.part_image_size[0], 3), dtype=np.uint8)
        part_masks = np.zeros((max_num_augmentation, self.part_image_size[1], self.part_image_size[0], 1), dtype=np.float32)
        thetas = np.zeros((max_num_augmentation, 2, 3), dtype=np.float32)
        types = -np.ones(max_num_augmentation, dtype=np.int64)

        joints_vis  = np.array(ann['joints_3d_vis'])[:,0]
        if not do_SA or (self.num_joints_SA > 0 and np.sum(joints_vis) < self.num_joints_SA):
            return part_imgs, part_masks, thetas, types

        num_augmentation = random.sample(self.num_parts_per_augmentation, 1)[0]
        for i in range(num_augmentation):
            t_type_idx = random.randint(0, len(self.candidate_parts)-1)
            t_type = self.candidate_parts[t_type_idx]
            # candidate_anchor_joint_ids = np.array(self.part_to_joint_id[t_type]) - 1
            # joints = np.array(ann['joints_3d'])[:,:2].reshape(-1, 2)
            # joints_vis  = np.array(ann['joints_3d_vis'])[:,0].reshape(-1, 1)
            # joints = np.concatenate((joints, joints_vis), axis=1)
            # anchor_joint_id = self.sample_anchor_joint(joints, candidate_anchor_joint_ids)

            # sample part joint to align
            part_ann = random.sample(self.part_anns[t_type], 1)[0]
            part_img = cv2.imread(os.path.join(self.part_root_dir, t_type, part_ann['file_name']))
            part_img = cv2.cvtColor(part_img, cv2.COLOR_BGR2RGB)
            part_joints = np.array(part_ann['keypoints']).reshape(-1, 3)

            scale_factor = random.uniform(*self.scale_range)
            part_img, part_joints = self.adjust_scale_by_diagnal(part_img, part_joints, body_bbox, self.part_scale[t_type], scale_factor)


            # 消除artifects
            part_mask = (cv2.cvtColor(part_img, cv2.COLOR_RGB2GRAY) > 0).astype(np.float32)
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.erode_kernel)
            part_mask = cv2.erode(part_mask, erode_kernel, 1)
            part_mask = cv2.GaussianBlur(part_mask, self.gaussian_kernel, 0)
            #part_mask = cv2.GaussianBlur(part_mask, (15,15), 0)
            part_mask = part_mask[:, :, np.newaxis]

            Hp, Wp = part_mask.shape[:2]
            center = np.array([Hp//2, Wp//2])
            scale = self.part_image_size / 200
            trans = get_affine_transform(center, scale, 0.0, self.part_image_size)

            part_img = cv2.warpAffine(part_img, trans,
                (int(self.part_image_size[0]), int(self.part_image_size[1])),
                flags=cv2.INTER_LINEAR)
            part_mask = cv2.warpAffine(part_mask, trans,
                (int(self.part_image_size[0]), int(self.part_image_size[1])),
                flags=cv2.INTER_LINEAR)

            theta = np.array([1,0,0,0,1,0]).astype(np.float32).reshape(2, 3)

            part_imgs[i] = part_img
            part_masks[i] = part_mask[...,np.newaxis]
            types[i] = t_type_idx
            thetas[i] = theta
        return part_imgs, part_masks, thetas, types






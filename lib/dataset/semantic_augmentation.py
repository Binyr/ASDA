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

class SemanticAugmentation():
	def __init__(self, scale_factor, vertical_translate, horizontal_translate, num_augmentation):
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

		self.head_joints_id = np.array([9, 10]) - 1
		self.max_aspect_of_part_and_body = 16
		self.min_aspect_of_part_and_body = 2

		part_annot_file = './lip/parts_filter_done/part_anns.json'
		self.part_root_dir = './lip/parts_filter_done/'
		with open(part_annot_file, 'r') as f:
			self.part_anns = json.load(f)
		self.delete_invalid_part()

		# self.scale_factor =  scale_factor
		# self.num_augmentation = num_augmentation
		# self.vertical_translate = vertical_translate
		# self.horizontal_translate = horizontal_translate
		# self.
		self.num_imgs_cannot_aug = 0

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
		

	def get_3rd_point(self, a, b, inv=False):
		direct = a - b
		return b + np.array([-direct[1], direct[0]], dtype=np.float32)

	def get_affine_transform_2d(self, src, dst, inv=False):
		src_3d = np.zeros((3, 2))
		src_3d[:2, :] = src[:, :]
		src_3d[2, :] = get_3rd_point(src_3d[0, :], src_3d[1, :])
		dst_3d = np.zeros((3, 2))
		dst_3d[:2, :] = dst[:, :]
		dst_3d[2, :] = get_3rd_point(dst_3d[0, :], dst_3d[1, :])

		if inv:
			trans = cv2.getAffineTransform(np.float32(dst_3d), np.float32(src_3d))
		else:
			trans = cv2.getAffineTransform(np.float32(src_3d), np.float32(dst_3d))
		return trans

	def rotate_line(self, src, rot, alph):
		#alph = np.random.rand()
		center = src[0] * alph + src[1] * (1 - alph)

		dst = np.zeros((2, 2))
		rot_rad = rot * np.pi / 180
		sn, cs = np.sin(rot_rad), np.cos(rot_rad)
		dst[0, 0] = (src[0] - center)[0] * cs - (src[0] - center)[1] * sn
		dst[0, 1] = (src[0] - center)[0] * sn + (src[0] - center)[1] * cs
		dst[0, :] += center

		dst[1, 0] = (src[1] - center)[0] * cs - (src[1] - center)[1] * sn
		dst[1, 1] = (src[1] - center)[0] * sn + (src[1] - center)[1] * cs
		dst[1, :] += center

		return dst

	def shift_line(self, src, rot, alph, direct, beta):

		ee = (src[0] - src[1]) * beta * direct
		ff = np.zeros(2)
		ff[0], ff[1] = -ee[1], ee[0]

		dst = np.zeros((2, 2))

		dst[0] = src[0] + ff
		dst[1] = src[1] + ff

		dst = rotate_line(src, rot, alph)
		return dst
		
	def affine_transform(self, pt, t):
	    new_pt = np.array([pt[0], pt[1], 1.]).T
	    new_pt = np.dot(t, new_pt)
	    return new_pt[:2]

	def sample_anchor_joint(self, joints, candidate_anchor_joint_ids):
		valid_index = []
		for i in candidate_anchor_joint_ids:
			if joints[i, 2] > 0:
				valid_index.append(i)
		if len(valid_index) == 0:
			return None

		t = random.sample(valid_index, 1)[0]
		return t

	def adjust_scale(self, src, dst, src_head, dst_head):
		src_head_scale = np.linalg.norm(src_head)
		dst_head_scale = np.linalg.norm(dst_head)
		s1 = dst_head_scale / src_head_scale

		src_scale = np.linalg.norm(src)
		dst_scale = np.linalg.norm(dst)
		s2 = src_scale * s1 / dst_scale

		new_dst = np.zeros((2, 2))
		center = (dst[0]  + dst[1]) / 2
		new_dst[0] = (dst[0] - center) * s2 + center
		new_dst[1] = (dst[1] - center) * s2 + center
		return new_dst

	def adjust_scale_by_area(self, src, dst, src_area, dst_body_area):
		src_scale = np.linalg.norm(src)
		dst_scale = np.linalg.norm(dst)
		s1 = (dst_scale / src_scale) ** 2
		if src_area * s1 * max_aspect_of_part_and_body < dst_body_area:
			s2 = np.sqrt(dst_body_area / (src_area * s1 * max_aspect_of_part_and_body))
		elif src_area * s1 * min_aspect_of_part_and_body > dst_body_area:
			s2 = np.sqrt(dst_body_area / (src_area * s1 * min_aspect_of_part_and_body))
		else:
			s2 = 1.0

		new_dst = np.zeros((2, 2))
		center = (dst[0]  + dst[1]) / 2
		new_dst[0] = (dst[0] - center) * s2 + center
		new_dst[1] = (dst[1] - center) * s2 + center
		return new_dst

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

	def paste(self, ori_img, part_img, pos):
		img_H, img_W = ori_img.shape[:2]
		part_H, part_W = part_img.shape[:2]

		l = max(pos[0], 0)
		r = min(pos[0] + part_W, img_W)
		u = max(pos[1], 0)
		d = min(pos[1] + part_H, img_H)

		ll = max(-pos[0], 0)
		rr = min(img_W - pos[0], part_W)
		uu = max(-pos[1], 0)
		dd = min(img_H - pos[1], part_H)

		if max(r - l, 0) * max(d - u, 0) / (part_H * part_W) < 0.5:
			return None

		# place part
		part_mask = (cv2.cvtColor(part_img, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

		# 消除artifects
		erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
		part_mask = cv2.erode(part_mask, erode_kernel, 1)
		part_mask = cv2.GaussianBlur(part_mask, (15,15), 0)
		part_mask = cv2.GaussianBlur(part_mask, (15,15), 0)
		part_mask = part_mask[:, :, np.newaxis]

		
		# print(l, r, u, d)
		# print(ll, rr, uu, dd)
		# import pdb
		# pdb.set_trace()
		ori_img[u:d, l:r] = ori_img[u:d, l:r] * (1 - part_mask[uu:dd, ll:rr]) + part_img[uu:dd, ll:rr] * part_mask[uu:dd, ll:rr]
		return ori_img

	def get_minimun_bbox(self, joints):
		new_joints = joints[np.where(joints[:, 2] > 0.5)]
		x1, y1 = np.min(new_joints[:, :2], axis=0)
		x2, y2 = np.max(new_joints[:, :2], axis=0)
		return np.array([x1, y1, x2, y2])

	def calc_area(self, bbox):
		return (bbox[3] - bbox[1] + 1) * (bbox[2] - bbox[0] + 1)


	def random_sign(self):
		if random.uniform(0, 1) < 0.5:
			return 1
		else:
			return -1


	def get_valid_part_idxes(self, joints):
		valid_part_idxes = []
		for i, part in enumerate(self.candidate_parts):
			for j in self.part_to_joint_id[part]:
				if joints[j-1, 2] > 0.5:
					valid_part_idxes.append(i)
					break
		return valid_part_idxes
	
	def randomly_parts_occlusion_repeat(self, imgAug, ann, scale_factor, vertical_translate, horizontal_translate, num_augmentation, part_type):
		
		for i in range(num_augmentation):
			# sample joint to align
			do_sample = True
			do_sample_count = 0
			while do_sample:
				do_sample_count += 1
				if do_sample_count > 500:
					self.num_imgs_cannot_aug += 1
					break
				do_sample = False

				t_type = self.candidate_parts[part_type]
				# 1-based index to 0-based index
				candidate_anchor_joint_ids = np.array(self.part_to_joint_id[t_type]) - 1
				joints = np.array(ann['joints_3d'])[:,:2].reshape(-1, 2)
				joints_vis  = np.array(ann['joints_3d_vis'])[:,0].reshape(-1, 1)
				joints = np.concatenate((joints, joints_vis), axis=1)
				anchor_joint_id = self.sample_anchor_joint(joints, candidate_anchor_joint_ids)
				if anchor_joint_id is None:
					anchor_joint_id = self.sample_anchor_joint(joints, np.arange(0,16))
					anchor_joint = joints[anchor_joint_id]
					anchor_joint_id = random.sample(candidate_anchor_joint_ids.tolist(), 1)[0]
				else:
					anchor_joint = joints[anchor_joint_id]

				# sample part joint to align
				part_ann = random.sample(self.part_anns[t_type], 1)[0]
				part_joints = np.array(part_ann['keypoints']).reshape(-1, 3)
				part_joint_to_align = part_joints[anchor_joint_id]
				part_img = cv2.imread(os.path.join(self.part_root_dir, t_type, part_ann['file_name']))
				part_H, part_W = part_img.shape[:2]
				if part_joint_to_align[2] <= 0.5 or part_joint_to_align[0] < 0 or part_joint_to_align[1] < 0 or part_joint_to_align[0] >= part_W or part_joint_to_align[1] >= part_H:
					do_sample = True
					continue

				# adjust scale
				body_bbox = self.get_minimun_bbox(joints)
				# part_img = cv2.imread(os.path.join(part_root_dir, t_type, part_ann['file_name']))
				part_img, part_joints = self.adjust_scale_by_diagnal(part_img, part_joints, body_bbox, self.part_scale[t_type], scale_factor)
				part_joint_to_align = part_joints[anchor_joint_id]

				# random shift
				body_H, body_W = body_bbox[3] - body_bbox[1], body_bbox[2] - body_bbox[0]
		
				p_x = horizontal_translate * body_W * self.random_sign()
				p_y = vertical_translate * body_H * self.random_sign()
				place_position = (anchor_joint[:2] + np.array([p_x, p_y]) - part_joint_to_align[:2]).astype(np.int32)

				t = self.paste(imgAug, part_img, place_position)
				if t is None:
					do_sample = True
					continue

		return imgAug


def calc_person_without_head_joints_ann(part_anns):
	# not_labeled_count = 0
	# not_visible_count = 0
	# all_count = 0
	for k, v in part_anns.items():
		print(k)
		not_labeled_count = 0
		not_visible_count = 0
		all_count = 0
		for ann in tqdm(v):
			joints = np.array(ann['keypoints']).reshape(-1, 3)
			head_joints = joints[head_joints_id]
			if head_joints[0, 2] < 1 or head_joints[1, 2] < 1:
				not_labeled_count += 1
			if head_joints[0, 2] < 2 or head_joints[1, 2] < 2:
				not_visible_count += 1
			all_count += 1
		print('not_labeled_count:{}'.format(not_labeled_count))
		print('not_visible_count:{}'.format(not_visible_count - not_labeled_count))
		print('visible_count:{}'.format(all_count - not_labeled_count))
	

if __name__ == '__main__':
	random.seed(23333)
	np.random.seed(23333)
	mpii_image_dir = './data/mpii/images'
	mpii_train_annotation_file = './data/mpii/annot/trainval.json'

	save_image_dir = './data/mpii/mpii_train_images_augmented_by_lip_part_filter_done_without_rotate_tmp'

	if not os.path.exists(save_image_dir):
		os.makedirs(save_image_dir)

	with open(mpii_train_annotation_file, 'r') as f:
		mpii_anns = json.load(f)

	image_2_ann = defaultdict(list)
	for ann in tqdm(mpii_anns):
		image_2_ann[ann['image']].append(ann)


	# calc_person_without_head_joints_ann(part_anns)
	# import pdb
	# pdb.set_trace()
	semanticAugmentationCore = SemanticAugmentation(scale_factor=0.8, vertical_translate=0.2, horizontal_translate=0.2, num_augmentation=4)

	for image_name in tqdm(list(image_2_ann.keys())[:]):
		anns = image_2_ann[image_name]

		img = cv2.imread(os.path.join(mpii_image_dir, image_name))
		img_h, img_w = img.shape[:2]

		for ith in range(5):
			imgAug = img.copy()
			save_image_name = os.path.splitext(image_name)[0] + '_%04d'%ith + '.jpg'
			for ann in anns:
				imgAug = semanticAugmentationCore.randomly_parts_occlusion_repeat(imgAug, ann)

			
			cv2.imwrite(os.path.join(save_image_dir, save_image_name), imgAug)
	print(num_imgs_cannot_aug)
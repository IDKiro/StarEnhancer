import os
import random
import numpy as np
import cv2
from glob import glob

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img


def augment(imgs=[]):
	H, W, _ = imgs[0].shape

	if random.random() < 0.1:
		Hc, Wc = H, W
	elif random.randint(0, 1) == 1:
		randr = random.uniform(0.75, 1.0)
		Hc = int(H * randr)
		Wc = int(W * randr)
	else:
		Hc = int(H * random.uniform(0.75, 1.0))
		Wc = int(W * random.uniform(0.75, 1.0))

	Hs = random.randint(0, H-Hc)
	Ws = random.randint(0, W-Wc)
	
	for i in range(len(imgs)):
		imgs[i] = cv2.resize(imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :], 
							 (224, 224), interpolation=cv2.INTER_AREA)

	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)
	if random.randint(0, 1) == 1: 
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=0)
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.transpose(imgs[i], (1, 0, 2))
	return imgs


class fivek_style(Dataset):
	def __init__(self, sub_dir, is_train, args):
		self.is_train = is_train

		self.style_dirs = sorted(glob(os.path.join(args.data_dir, sub_dir, '*-*')))
		self.style_num = len(self.style_dirs)

		self.img_names = sorted(os.listdir(self.style_dirs[0]))
		self.img_num = len(self.img_names)

	def __len__(self):
		l = self.img_num * self.style_num
		return l

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		style_id = idx // self.img_num
		img_id = idx % self.img_num

		source_img = read_img(os.path.join(self.style_dirs[style_id], self.img_names[img_id]))
		
		if self.is_train:
			[source_img] = augment([source_img])
		else:
			source_img = cv2.resize(source_img, (224, 224), interpolation=cv2.INTER_AREA)

		return hwc_to_chw(source_img), style_id


class fivek_enhance(Dataset):
	def __init__(self, sub_dir, is_train, args=None):
		assert args != None

		self.is_train = is_train

		self.style_dirs = sorted(glob(os.path.join(args.data_dir, sub_dir, '*-*')))
		self.style_num = len(self.style_dirs)

		if not self.is_train:
			self.source_id = args.source_id - 1
			self.target_id = args.target_id - 1

		self.img_names = sorted(os.listdir(self.style_dirs[0]))
		self.img_num = len(self.img_names)
		self.img_list = list(range(self.img_num))

		self.in_num = args.in_num if args.in_num > 0 else self.img_num
		self.out_num = args.out_num if args.out_num > 0 else self.img_num

		self.embeddings = np.load(os.path.join(args.save_dir, 'embeddings.npy'))

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		source_id = random.randint(0, self.style_num - 1) if self.is_train else self.source_id
		target_id = random.randint(0, self.style_num - 1) if self.is_train else self.target_id
		source_dir = self.style_dirs[source_id]
		gt_dir = self.style_dirs[target_id]

		source_img = read_img(os.path.join(source_dir, self.img_names[idx]))
		target_img = read_img(os.path.join(gt_dir, self.img_names[idx]))

		if self.is_train:
			[source_img, target_img] = augment([source_img, target_img])
			if random.randint(0, 1) == 1:
				source_list = random.sample(self.img_list, random.randint(1, self.img_num))
				target_list = random.sample(self.img_list, random.randint(1, self.img_num))
			else:
				source_list = self.img_list
				target_list = self.img_list
		else:
			source_list = random.sample(self.img_list, self.in_num)
			target_list = random.sample(self.img_list, self.out_num)

		source_embeddings = self.embeddings[source_id, source_list, :]
		target_embeddings = self.embeddings[target_id, target_list, :]

		source_center = np.mean(source_embeddings, axis=0)
		target_center = np.mean(target_embeddings, axis=0)

		source_center = source_center / np.linalg.norm(source_center)
		target_center = target_center / np.linalg.norm(target_center)

		return hwc_to_chw(source_img), source_center, hwc_to_chw(target_img), target_center

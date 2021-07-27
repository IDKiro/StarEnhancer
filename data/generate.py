import os, shutil
import random, math
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
import cv2


if __name__ == '__main__':
	root_dir = './Full-Res/'

	train_dir = './train/'
	valid_dir = './valid/'
	test_dir = './test/'

	IDs = os.listdir(root_dir)

	img_size = 448

	filenames = [os.path.basename(f) for f in glob(os.path.join(root_dir, IDs[0], '*.jpg'))]
	filenames.sort()

	train_names = filenames[0:4500]
	valid_names = filenames[4500:5000]
	test_names = filenames[4500:5000]

	for filename in tqdm(train_names):
		file_id = filename.split('-')[0]
		for ID in IDs:
			in_img = cv2.imread(os.path.join(root_dir, ID, filename)) / 255.
			out_img = cv2.resize(in_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
			out_img = np.uint8(np.round(np.clip(out_img, 0, 1) * 255.))

			out_file_dir = os.path.join(train_dir, ID)

			if not os.path.isdir(out_file_dir):
				os.makedirs(out_file_dir)

			out_filename = os.path.join(out_file_dir, file_id+'.jpg')
			cv2.imwrite(out_filename, out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

	for filename in tqdm(valid_names):
		file_id = filename.split('-')[0]
		for ID in IDs:
			in_img = cv2.imread(os.path.join(root_dir, ID, filename)) / 255.
			out_img = cv2.resize(in_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
			out_img = np.uint8(np.round(np.clip(out_img, 0, 1) * 255.))

			out_file_dir = os.path.join(valid_dir, ID)

			if not os.path.isdir(out_file_dir):
				os.makedirs(out_file_dir)

			out_filename = os.path.join(out_file_dir, file_id+'.jpg')
			cv2.imwrite(out_filename, out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

	for filename in tqdm(test_names):
		file_id = filename.split('-')[0]
		for ID in IDs:
			source_file = os.path.join(root_dir, ID, filename)

			out_file_dir = os.path.join(test_dir, ID)
			if not os.path.isdir(out_file_dir):
				os.makedirs(out_file_dir)

			target_file = os.path.join(out_file_dir, file_id+'.jpg')

			shutil.copyfile(source_file, target_file)

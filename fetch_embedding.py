import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Stylish
from dataset.loader import fivek_style


parser = argparse.ArgumentParser()
parser.add_argument('--dims', default=512, type=int, help='embedding dimensions')
parser.add_argument('--style_num', default=10, type=int, help='number of styles in the train set')
parser.add_argument('--batch_size', default=100, type=int, help='mini batch size')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./save_model/', type=str, help='path to embeddings saving')
parser.add_argument('--data_dir', default='./data/'	, type=str, help='path to dataset')
args = parser.parse_args()


if __name__ == '__main__':
	stylish = Stylish(args.dims, args.style_num)
	stylish = nn.DataParallel(stylish).cuda()

	infer_dataset = fivek_style('train', False, args)
	infer_loader = DataLoader(infer_dataset,
							  batch_size=args.batch_size,
							  num_workers=args.num_workers)

	if os.path.exists(os.path.join(args.save_dir, 'stylish.pth.tar')):
		stylish.load_state_dict(torch.load(os.path.join(args.save_dir, 'stylish.pth.tar'))['state_dict'])
		style_encoder = stylish.module.encoder
		style_encoder.eval()

		embeddings = []

		for (source_img, _) in tqdm(infer_loader):
			source_img = source_img.cuda(non_blocking=True)
			embeddings.append(style_encoder(source_img).detach().cpu().numpy())

		embeddings = np.concatenate(embeddings, axis=0)
		embeddings_list = np.split(embeddings, args.style_num, axis=0)
		embeddings = np.stack(embeddings_list)

		np.save(os.path.join(args.save_dir, 'embeddings.npy'), embeddings)

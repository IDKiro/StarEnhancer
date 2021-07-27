import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from model import Mapping, Enhancer, SlimEnhancer
from utils import AverageMeter, write_img, chw_to_hwc
from dataset.loader import fivek_enhance


parser = argparse.ArgumentParser()
parser.add_argument('--dims', default=512, type=int, help='embedding dimensions')
parser.add_argument('--source_id', default=6, type=int, help='source style id for test (start from 1)')
parser.add_argument('--target_id', default=3, type=int, help='target style id for test (start from 1)')
parser.add_argument('--in_num', default=-1, type=int, help='source style embeddings number (-1 for all)')
parser.add_argument('--out_num', default=-1, type=int, help='target style embeddings number (-1 for all)')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./save_model/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/'	, type=str, help='path to dataset')
parser.add_argument('--result_dir', default='./result/', type=str, help='path to results saving')
parser.add_argument('--save_result', default=False, action='store_true', help='save the enhanced results')
parser.add_argument('--test_speed', default=False, action='store_true', help='test the fps')
args = parser.parse_args()


def single(name):
	state_dict = torch.load(os.path.join(args.save_dir, name))['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def test(test_loader, mapping, enhancer):
	loss_fn_alex = lpips.LPIPS(net='alex')
	loss_fn_alex = loss_fn_alex.cuda()
	
	PSNR = AverageMeter()
	SSIM = AverageMeter()
	LPIPS = AverageMeter()
	
	torch.cuda.empty_cache()

	mapping.eval()
	enhancer.eval()

	if args.test_speed:
		test_bs = 16
		test_h = 1080			# 1080p is more friendly for other GPUs
		test_w = 1920			# 4K may be OOM
		test_epoch = 10
		test_iter = 256

		with torch.no_grad():
			style = mapping(torch.zeros([1, 512]).cuda())
			rand_img = torch.ones([test_bs, 3, test_h, test_w]).cuda()

			enhancer = torch.jit.trace(enhancer, [rand_img, style, style])

			for i in range(test_epoch):
				t1 = time.time()
				for _ in range(test_iter):
					output = enhancer(rand_img, style, style)

				t2 = time.time()
				print('Epoch %d, finish %d images in %.4fs, speed = %.4f FPS' % (i, test_bs*test_iter, t2-t1, test_bs*test_iter/(t2-t1)))

	for ind, (source_img, source_center, target_img, target_center) in enumerate(test_loader):
		source_img = source_img.cuda(non_blocking=True)
		source_center = source_center.cuda(non_blocking=True)
		target_img = target_img.cuda(non_blocking=True)
		target_center = target_center.cuda(non_blocking=True)

		with torch.no_grad():
			if not ind:
				style_A = mapping(source_center[[0], ...])
				style_B = mapping(target_center[[0], ...])
			
			output = enhancer(source_img, style_A, style_B)
			output = (output.clamp_(0, 1) * 255).round_() / 255.

			mse_loss = F.mse_loss(output, target_img, reduction='none').mean((1, 2, 3))
			psnr_val = 10 * torch.log10(1 / mse_loss).mean().item()

			_, _, H, W = output.size()
			down_ratio = max(1, round(min(H, W) / 256))
			ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))), 
							F.adaptive_avg_pool2d(target_img, (int(H / down_ratio), int(W / down_ratio))), 
							data_range=1, size_average=False).item()				# Zhou Wang

			lpips_val = loss_fn_alex(output * 2 - 1, target_img * 2 - 1).item()		# Richard Zhang

		PSNR.update(psnr_val)
		SSIM.update(ssim_val)
		LPIPS.update(lpips_val)

		print('Test: [{0}-{1}-{2}]\t'
			  'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
			  'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})\t'
			  'LPIPS: {lpips.val:.03f} ({lpips.avg:.03f})\t'
			  .format(ind+4501, args.source_id, args.target_id, psnr=PSNR, ssim=SSIM, lpips=LPIPS))

		if args.save_result:
			os.makedirs(args.result_dir, exist_ok=True)
			out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
			out_filename = '%04d-%02d-%02d-%.02f.png'%(ind+4501, args.source_id, args.target_id, psnr_val)
			write_img(os.path.join(args.result_dir, out_filename), out_img)


if __name__ == '__main__':
	mapping = Mapping(args.dims).cuda()
	enhancer = SlimEnhancer().cuda()

	test_dataset = fivek_enhance('test', False, args)
	test_loader = DataLoader(test_dataset, 
                             num_workers=args.num_workers)

	mapping.load_state_dict(single('mapping.pth.tar'))
	enhancer.load_state_dict(single('enhancer.pth.tar'))
		
	test(test_loader, mapping, enhancer)

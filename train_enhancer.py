import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Mapping, Enhancer, SlimEnhancer, TotalLoss
from utils import AverageMeter, BalancedDataParallel
from dataset.loader import fivek_enhance


parser = argparse.ArgumentParser()
parser.add_argument('--dims', default=512, type=int, help='embedding dimensions')
parser.add_argument('--source_id', default=6, type=int, help='source style id for validation (start from 1)')
parser.add_argument('--target_id', default=3, type=int, help='target style id for validation (start from 1)')
parser.add_argument('--in_num', default=-1, type=int, help='source style embeddings number (-1 for all)')
parser.add_argument('--out_num', default=-1, type=int, help='target style embeddings number (-1 for all)')
parser.add_argument('--t_batch_size', default=384, type=int, help='mini batch size for training')
parser.add_argument('--v_batch_size', default=500, type=int, help='mini batch size for validation')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--epochs', default=2000, type=int, help='number of training epochs')
parser.add_argument('--eval_freq', default=20, type=int, help='frequency of validation')
parser.add_argument('--save_dir', default='./save_model/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/'	, type=str, help='path to dataset')
args = parser.parse_args()


def train(train_loader, mapping, enhancer, criterion, optimizer):
	losses = AverageMeter()

	mapping.train()
	enhancer.train()

	for (source_img, source_center, target_img, target_center) in train_loader:
		source_img = source_img.cuda(non_blocking=True)
		source_center = source_center.cuda(non_blocking=True)
		target_img = target_img.cuda(non_blocking=True)
		target_center = target_center.cuda(non_blocking=True)

		style_A = mapping(source_center)
		style_B = mapping(target_center)
		output = enhancer(source_img, style_A, style_B)

		loss = criterion(output, target_img)

		losses.update(loss.item(), args.t_batch_size)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	return losses.avg


def valid(val_loader, mapping, enhancer):
	PSNR = AverageMeter()
	
	torch.cuda.empty_cache()

	mapping.eval()
	enhancer.eval()

	for (source_img, source_center, target_img, target_center) in val_loader:
		source_img = source_img.cuda(non_blocking=True)
		source_center = source_center.cuda(non_blocking=True)
		target_img = target_img.cuda(non_blocking=True)
		target_center = target_center.cuda(non_blocking=True)

		with torch.no_grad():
			style_A = mapping(source_center)
			style_B = mapping(target_center)
			output = enhancer(source_img, style_A, style_B).clamp_(0, 1)

		mse_loss = F.mse_loss(output, target_img, reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()
		PSNR.update(psnr.item(), args.v_batch_size)

	return PSNR.avg


if __name__ == '__main__':
	mapping = Mapping(args.dims)
	mapping = BalancedDataParallel(0, mapping, dim=0).cuda()

	enhancer = SlimEnhancer()
	enhancer = BalancedDataParallel(0, enhancer, dim=0).cuda()

	criterion = TotalLoss()

	train_dataset = fivek_enhance('train', True, args)
	train_loader = DataLoader(train_dataset, 
                              batch_size=args.t_batch_size, 
                              shuffle=True, 
                              num_workers=args.num_workers, 
                              pin_memory=True, 
                              drop_last=True)

	val_dataset = fivek_enhance('valid', False, args)
	val_loader = DataLoader(val_dataset, 
                            batch_size=args.v_batch_size, 
                            num_workers=5, 
                            pin_memory=True)

	if not os.path.exists(os.path.join(args.save_dir, 'enhancer.pth.tar')):
		optimizer = torch.optim.Adam([
      		{'params': enhancer.parameters(), 'lr': args.lr},
			{'params': mapping.parameters(), 'lr': args.lr}
		])						  
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

		best_psnr = 0
		for epoch in range(args.epochs + 1):
			loss = train(train_loader, mapping, enhancer, criterion, optimizer)
			print('Train [{0}]\t'
			      'Loss: {loss:.4f}\t '
				  'Best Val PSNR: {psnr:.2f}'.format(epoch, loss=loss, psnr=best_psnr))

			scheduler.step()

			if epoch % args.eval_freq == 0:
				avg_psnr = valid(val_loader, mapping, enhancer)
				print('Valid: [{0}]\tPSNR: {psnr:.2f}'.format(epoch, psnr=avg_psnr))

				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					torch.save({'state_dict': mapping.state_dict()}, 
                			   os.path.join(args.save_dir, 'mapping.pth.tar'))
					torch.save({'state_dict': enhancer.state_dict()}, 
                			   os.path.join(args.save_dir, 'enhancer.pth.tar'))

	else:
		print('==> Existing trained model')
		exit(1)

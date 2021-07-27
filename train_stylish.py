import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Stylish
from utils import AverageMeter
from dataset.loader import fivek_style


parser = argparse.ArgumentParser()
parser.add_argument('--dims', default=512, type=int, help='embedding dimensions')
parser.add_argument('--style_num', default=10, type=int, help='number of styles in the train set')
parser.add_argument('--t_batch_size', default=256, type=int, help='mini batch size for training')
parser.add_argument('--v_batch_size', default=500, type=int, help='mini batch size for validation')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--eval_freq', default=10, type=int, help='frequency of validation')
parser.add_argument('--save_dir', default='./save_model/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
args = parser.parse_args()


def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def train(train_loader, stylish, optimizer, epoch):
	losses = AverageMeter()

	stylish.train()

	for (source_img, style_id) in tqdm(train_loader):
		source_img = source_img.cuda(non_blocking=True)
		style_id = style_id.cuda(non_blocking=True)

		simInd = stylish(source_img)
		
		loss = F.cross_entropy(simInd * 20, style_id)

		losses.update(loss.item(), args.t_batch_size)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	print('Train [{0}] \t Loss: {loss:.2f}'.format(epoch, loss=losses.avg))


def valid(val_loader, stylish, epoch):
	top1 = AverageMeter()

	torch.cuda.empty_cache()
	stylish.eval()
	
	for (source_img, style_id) in tqdm(val_loader):
		source_img = source_img.cuda(non_blocking=True)

		with torch.no_grad():
			simInd = stylish(source_img)

		prec1 = accuracy(simInd.data.cpu(), style_id)[0]

		top1.update(prec1[0].item(), style_id.size(0))

	print ('Valid [{0}] \t Recall@1: {top1:.2f}'.format(epoch, top1=top1.avg))

	return top1.avg


if __name__ == '__main__':
	stylish = Stylish(args.dims, args.style_num)
	stylish = nn.DataParallel(stylish).cuda()

	train_dataset = fivek_style('train', True, args)
	train_loader = DataLoader(train_dataset,
                              batch_size=args.t_batch_size, 
							  shuffle=True, 
							  num_workers=args.num_workers, 
							  pin_memory=True, 
							  drop_last=True)

	val_dataset = fivek_style('valid', False, args)
	val_loader = DataLoader(val_dataset, 
							batch_size=args.v_batch_size, 
							num_workers=args.style_num, 
							pin_memory=True)

	if not os.path.exists(os.path.join(args.save_dir, 'stylish.pth.tar')):
		os.makedirs(args.save_dir, exist_ok=True)
		optimizer = torch.optim.Adam([
      		{'params': stylish.module.encoder.parameters(), 'lr': args.lr},
            {'params': stylish.module.proxy.parameters(), 'lr': args.lr * 100}
        ])
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

		best_top = 0
		for epoch in range(args.epochs + 1):
			train(train_loader, stylish, optimizer, epoch)
			scheduler.step()

			if epoch % args.eval_freq == 0:
				top = valid(val_loader, stylish, epoch)

				if top > best_top:
					best_top = top
					torch.save({'state_dict': stylish.state_dict()},
        					   os.path.join(args.save_dir, 'stylish.pth.tar'))

			print('Best validation Recall@1: %.2f' % best_top)

	else:
		print('==> Existing style vectors')


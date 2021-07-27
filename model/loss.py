
import torch
import torch.nn as nn
import torch.nn.functional as F


class TotalLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def rgb2lab(self, rgb_image):
		rgb_to_xyz = torch.FloatTensor([  # X        Y          Z
										[0.412453, 0.212671, 0.019334],  # R
										[0.357580, 0.715160, 0.119193],  # G
										[0.180423, 0.072169, 0.950227],  # B
										]).cuda(non_blocking=True).t()

		fxfyfz_to_lab = torch.FloatTensor([
											[0.0,	500.0,	0.0],  		# fx
										   	[116.0,	-500.0,	200.0],  	# fy
										   	[0.0,	0.0,	-200.0],  	# fz 
											]).cuda(non_blocking=True).t()

		img = (rgb_image / 12.92) * rgb_image.le(0.04045).float() + \
			(((torch.clamp(rgb_image, min=0.0001) + 0.055) / 1.055) ** 2.4) * rgb_image.gt(0.04045).float()

		img = img.permute(1, 0, 2, 3).contiguous().view(3, -1)
		img = torch.matmul(rgb_to_xyz, img)
		img = torch.mul(img, torch.FloatTensor([1/0.950456, 1.0, 1/1.088754]).cuda(non_blocking=True).view(3, 1))

		epsilon = 6/29
		img = ((img / (3.0 * epsilon**2) + 4.0/29.0) * img.le(epsilon**3).float()) + \
			(torch.clamp(img, min=0.0001)**(1.0/3.0) * img.gt(epsilon**3).float())
		img = torch.matmul(fxfyfz_to_lab, img) + torch.FloatTensor([-16.0, 0.0, 0.0]).cuda(non_blocking=True).view(3, 1)

		img = img.view(3, rgb_image.size(0), rgb_image.size(2), rgb_image.size(3)).permute(1, 0, 2, 3)

		img[:, 0, :, :] = img[:, 0, :, :] / 100
		img[:, 1, :, :] = (img[:, 1, :, :] / 110 + 1) / 2
		img[:, 2, :, :] = (img[:, 2, :, :] / 110 + 1) / 2

		img[(img != img).detach()] = 0

		img = img.contiguous()

		return img	

	def forward(self, out_image, gt_image):
		out_lab = self.rgb2lab(out_image)
		gt_lab = self.rgb2lab(gt_image)

		loss = F.l1_loss(out_lab, gt_lab)

		return loss

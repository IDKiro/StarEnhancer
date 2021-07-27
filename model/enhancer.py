
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .basis import ResBlk, DualAdainResBlk, conv1x1, conv3x3


class CurveEncoder(nn.Module):
	def __init__(self, dims):
		super(CurveEncoder, self).__init__()
		self.layers = [2, 2, 2, 2]
		self.planes = [64, 128, 256, 512]
		self.dims = dims

		self.num_layers = sum(self.layers)
		self.inplanes = self.planes[0]

		self.conv1 = nn.Conv2d(3, self.planes[0], kernel_size=7, stride=2, padding=3, bias=False)
		self.bias1 = nn.Parameter(torch.zeros(1))
		self.actv = nn.PReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		
		self.layer1 = nn.Sequential(*self._make_layer(ResBlk, self.planes[0], self.layers[0]))
		self.layer2 = nn.Sequential(*self._make_layer(ResBlk, self.planes[1], self.layers[1], stride=2))
		self.layer3 = nn.Sequential(*self._make_layer(ResBlk, self.planes[2], self.layers[2], stride=2))
		self.layer4 = self._make_layer(DualAdainResBlk, self.planes[3], self.layers[3], stride=2)

		self.gap = nn.AdaptiveAvgPool2d(1)
		self.bias2 = nn.Parameter(torch.zeros(1))
		self.fc = nn.Linear(self.planes[3], self.dims)

		self._reset_params()
		
	def _reset_params(self):
		for m in self.modules():
			if isinstance(m, ResBlk) or isinstance(m, DualAdainResBlk):
				nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
				nn.init.constant_(m.conv2.weight, 0)
				if m.downsample is not None:
					nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
			elif isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes:
			downsample = conv1x1(self.inplanes, planes, stride)

		layers = nn.ModuleList()
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return layers

	def forward(self, x, sa, sb):
		x = self.conv1(x)
		x = self.actv(x + self.bias1)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)

		for i in range(self.layers[3]):
			x = self.layer4[i](x, sa[i], sb[i])

		x = self.gap(x).flatten(1)
		x = self.fc(x + self.bias2)

		return x


class Enhancer(nn.Module):
	def __init__(self):
		super(Enhancer, self).__init__()
		self.cd = 256

		self.cl = [86, 52, 52, 18, 18, 
				   52, 86, 52, 18, 18, 
				   52, 52, 86, 18, 18]

		self.encoder = CurveEncoder(sum(self.cl))

	def interp(self, param, length):
		return F.interpolate(
			param.unsqueeze(1).unsqueeze(2), (1, length), 
			mode='bicubic', align_corners=True
		).squeeze(2).squeeze(1)

	def curve(self, x, func, depth):
		x_ind = (torch.clamp(x, 0, 1) * (depth - 1))
		x_ind = x_ind.round_().long().flatten(1).detach()
		out = torch.gather(func, 1, x_ind)
		return out.reshape(x.size())

	def forward(self, x, sa, sb):
		_, _, H, W = x.size()

		# curves
		fl = self.encoder(
			F.adaptive_avg_pool2d(x, (224, 224)), sa, sb
		).split(self.cl, dim=1)

		# transform
		residual = torch.cat([
			self.curve(x[:, [0], ...], self.interp(fl[i*5+0], self.cd), self.cd) + \
			self.curve(x[:, [1], ...], self.interp(fl[i*5+1], self.cd), self.cd) + \
			self.curve(x[:, [2], ...], self.interp(fl[i*5+2], self.cd), self.cd) + \
			self.interp(fl[i*5+3], H).unsqueeze(1).unsqueeze(3).expand(-1, -1, -1, W) + \
			self.interp(fl[i*5+4], W).unsqueeze(1).unsqueeze(2).expand(-1, -1, H, -1) 
			for i in range(3)], dim=1)

		return x + residual


class SlimEnhancer(nn.Module):
	def __init__(self):
		super(SlimEnhancer, self).__init__()
		self.cd = 256
		self.cl = 64

		self.encoder = CurveEncoder(self.cl * 9)

	def interp(self, param, length):
		return F.interpolate(
			param.unsqueeze(1).unsqueeze(2), (1, length), 
			mode='bicubic', align_corners=True
		).squeeze(2).squeeze(1)

	def curve(self, x, func, depth):
		x_ind = x * (depth - 1)
		x_ind = x_ind.long().flatten(2).detach()
		out = torch.gather(func, 2, x_ind)
		return out.reshape(x.size())

	def forward(self, x, sa, sb):
		B, _, H, W = x.size()

		# curves
		params = self.encoder(
			F.adaptive_avg_pool2d(x, (224, 224)), sa, sb
		).view(B, 9, self.cl, 1)

		curves = F.interpolate(
			params, (self.cd, 1), 
			mode='bicubic', align_corners=True
		).squeeze(3)

		# transform
		residual = self.curve(
			x.repeat(1, 3, 1, 1), curves, self.cd
		).view(B, 3, 3, H, W).sum(dim=2)

		return x + residual

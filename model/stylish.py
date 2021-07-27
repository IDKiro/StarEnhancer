import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

from .basis import ResBlk, conv1x1, conv3x3


class StyleEncoder(nn.Module):
	def __init__(self, dim):
		super(StyleEncoder, self).__init__()
		self.layers = [4, 4, 4, 4]
		self.planes = [64, 128, 256, 512]

		self.num_layers = sum(self.layers)
		self.inplanes = self.planes[0]

		self.conv1 = nn.Conv2d(3, self.planes[0], kernel_size=7, stride=2, padding=3, bias=False)
		self.bias1 = nn.Parameter(torch.zeros(1))
		self.actv = nn.PReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(ResBlk, self.planes[0], self.layers[0])
		self.layer2 = self._make_layer(ResBlk, self.planes[1], self.layers[1], stride=2)
		self.layer3 = self._make_layer(ResBlk, self.planes[2], self.layers[2], stride=2)
		self.layer4 = self._make_layer(ResBlk, self.planes[3], self.layers[3], stride=2)
		self.gap = nn.AdaptiveAvgPool2d(1)
		self.gmp = nn.AdaptiveMaxPool2d(1)
		self.bias2 = nn.Parameter(torch.zeros(1))

		self.fc = nn.Linear(self.planes[3], dim)

		self._reset_params()

	def _reset_params(self):
		for m in self.modules():
			if isinstance(m, ResBlk):
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

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.actv(x + self.bias1)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		avg_x = self.gap(x)
		max_x = self.gmp(x)

		x = (max_x + avg_x).flatten(1)
		x = self.fc(x + self.bias2)

		x = F.normalize(x, p=2, dim=1)

		return x


class Proxy(nn.Module):
	def __init__(self, dim, cN):
		super(Proxy, self).__init__()
		self.fc = Parameter(torch.Tensor(dim, cN))
		torch.nn.init.xavier_normal_(self.fc)

	def forward(self, input):
		centers = F.normalize(self.fc, p=2, dim=0)
		simInd = input.matmul(centers)

		return simInd


class Stylish(nn.Module):
	def __init__(self, dim, cN):
		super(Stylish, self).__init__()
		self.encoder = StyleEncoder(dim)
		self.proxy = Proxy(dim, cN)

	def forward(self, x):
		x = self.encoder(F.adaptive_avg_pool2d(x, (224, 224)))
		x = self.proxy(x)

		return x


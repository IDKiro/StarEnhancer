import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlk(nn.Module):
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(ResBlk, self).__init__()
		self.bias1a = nn.Parameter(torch.zeros(1))
		self.actv1 = nn.PReLU()
		self.bias1b = nn.Parameter(torch.zeros(1))
		self.conv1 = conv3x3(inplanes, planes, stride)
		
		self.bias2a = nn.Parameter(torch.zeros(1))
		self.actv2 = nn.PReLU()
		self.bias2b = nn.Parameter(torch.zeros(1))
		self.conv2 = conv3x3(planes, planes)
		self.scale = nn.Parameter(torch.ones(1))

		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.actv1(x + self.bias1a)

		if self.downsample is not None:
			identity = self.downsample(out)

		out = self.conv1(out + self.bias1b)

		out = self.actv2(out + self.bias2a)
		out = self.conv2(out + self.bias2b)
		
		out = out * self.scale

		out += identity

		return out


class DualAdainResBlk(nn.Module):
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(DualAdainResBlk, self).__init__()
		self.bias1a = nn.Parameter(torch.zeros(1))
		self.actv1 = nn.PReLU()
		self.bias1b = nn.Parameter(torch.zeros(1))
		self.conv1 = conv3x3(inplanes, planes, stride)

		self.bias2a = nn.Parameter(torch.zeros(1))
		self.actv2 = nn.PReLU()
		self.bias2b = nn.Parameter(torch.zeros(1))
		self.conv2 = conv3x3(planes, planes)
		self.scale = nn.Parameter(torch.ones(1))

		self.downsample = downsample
		self.stride = stride

	def forward(self, x, sa, sb):
		identity = x

		out = self.actv1(x + self.bias1a)

		if self.downsample is not None:
			identity = self.downsample(out)

		out = self.conv1(out + self.bias1b)

		alpha = sb[0] / sa[0]
		beta = sb[1] -  sa[1] * alpha
		out = out * alpha + beta

		out = self.actv2(out + self.bias2a)
		out = self.conv2(out + self.bias2b)

		out = out * self.scale

		out += identity

		return out

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mapping(nn.Module):
	def __init__(self, in_dim):
		super(Mapping, self).__init__()
		self.layers = 2
		self.planes = 512

		self.mlp_in = nn.Sequential(
			nn.Linear(in_dim, 512),
			nn.PReLU(),
			nn.Linear(512, 512),
			nn.PReLU(),
			nn.Linear(512, 512),
			nn.PReLU(),
			nn.Linear(512, 512),
			nn.PReLU()
		)

		self.mlp_out = nn.ModuleList()
		for _ in range(self.layers):
			self.mlp_out.append(
				nn.Sequential(
					nn.Linear(512, 512),
					nn.PReLU(),
					nn.Linear(512, self.planes * 2),
					nn.Sigmoid()
				)
			)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.mlp_in(x)

		s_list = []
		for i in range(self.layers):
			out = self.mlp_out[i](x).view(x.size(0), -1, 1, 1)
			s_list.append(list(torch.chunk(out, chunks=2, dim=1)))

		return s_list

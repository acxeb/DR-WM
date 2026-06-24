import torch
import torch.nn as nn


class Identity(nn.Module):
	"""
	Identity-mapping noise layer. Does not change the image
	"""

	def __init__(self):
		super(Identity, self).__init__()
		self.is_malicious = False

	def forward(self, image):
		return image

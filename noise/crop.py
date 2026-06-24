import torch
import torch.nn as nn
import numpy as np



class Dropout(nn.Module):

	def __init__(self, prob=0.3):
		super(Dropout, self).__init__()
		self.prob = prob
		self.is_malicious = False

	def forward(self, image, cover_image):

		maskk = torch.Tensor(np.random.choice([0.0, 1.0], image.shape[2:], p=[self.prob, 1 - self.prob])).to(image.device)
		maskk = maskk.expand_as(image)
		output = image * maskk + cover_image * (1 - maskk)
		return output


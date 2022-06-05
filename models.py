import torch.nn as nn
import math
import torch
from torch.nn import functional as F

class myAutoEncoder(nn.Module):
	def __init__(self, classes=2):
		super(myAutoEncoder, self).__init__()    
		self.encoder = nn.Sequential(
			#input for the encoder : 3 28 28
			nn.Conv2d(3, 16, 3, stride=1, padding=1),  
			nn.Tanh(True),
			nn.MaxPool2d(2),
			nn.Conv2d(16, 4, 3, stride=1, padding=1), 
			nn.Tanh(True),
			nn.MaxPool2d(2)
			#output for encoder :  4 7 7
		)

		self.decoder = nn.Sequential(         
			#Input of decoder : 4 7 7 
			nn.ConvTranspose2d(4, 16, 2, stride=2),
			nn.Tanh(True),         
			nn.ConvTranspose2d(16, 3, 2, stride=2),
			nn.Sigmoid()
			#output of decoder : 3 28 28
		)
		
		self.classifier = nn.Sequential(
			#Input of classifier : 3 28 28 3 
			nn.Linear(3*28*28*3, 3*28*28*6),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(3*28*28*6, 3*28*28*3),
			nn.ReLU(inplace=True),
			nn.Linear(3*28*28*3, 28),
			nn.Linear(28, 2)
		)

	def forward(self, x, y, z):
		x = self.encoder(x)
		x = self.decoder(x)
		x = x.view(x.size(0), -1)

		y = self.encoder(y)
		y = self.decoder(y)
		y = y.view(y.size(0), -1)
		
		z = self.encoder(z)
		z = self.decoder(z)
		z = z.view(z.size(0), -1)

		output = torch.cat((x, y, z), 1)
		output = self.classifier(x)

		return output

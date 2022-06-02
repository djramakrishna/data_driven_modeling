import torch.nn as nn
import torch.nn.functional as F


class myAutoEncoder(nn.Module):
    def __init__(self):
        super(myAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  
            nn.Tanh(True),
            nn.MaxPool2d(2, stride=1),  
            
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1), 
            nn.Tanh(True),
            nn.MaxPool2d(2, stride=1),
            
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1), 
            nn.Tanh(True),
            nn.MaxPool2d(2, stride=1)

            #output of encoder :  4 4 8
        )

        self.decoder = nn.Sequential(
            
            #Input of decoder : 4 4 8

            # Conv transpose 2D  2*2  8*8*8
            nn.ConvTranspose2d(8, 8, 2, stride=2, padding=0),  # b, 16, 5, 5
            nn.Tanh(True),
                
            # Conv transpose 2D 2*2 16*16*8
            nn.ConvTranspose2d(8, 8, 2, stride=2, padding=0),  # b, 8, 15, 15
            nn.Tanh(True),

            # Conv transpose 2D 2*2 28*28*16
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh(),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return x

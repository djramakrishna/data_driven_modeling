import torch.nn as nn
import torch.nn.functional as F


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
            #Input of classifier : 3 28 28 
            nn.Linear(3*28*28, self.classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

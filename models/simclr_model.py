# SimCLR contrastive learning model
import torch
import torch.nn as nn


class SimCLRModel(nn.Module):
    def __init__(self, encoder, projection):
        """[summary]
        
        Arguments:
            encoder -- ResNet model to be used as encoder
            projection -- Projection head
        """
        super().__init__()
        
        self.encoder = encoder
        self.encoder.linear = nn.Identity()  # replace last linear layer with identity
        
        self.projection = projection
        
    def forward(self, x):
        x = self.encoder(x)
        out = self.projection(x)
        
        return out
    
    def encode(self, x):
        self.eval()
        with torch.no_grad():
            return self.encoder(x)
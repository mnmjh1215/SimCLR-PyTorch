# Normalized Temperature-scaled Cross Entropy from the paper

import torch.nn as nn
import torch
        
LARGE_NUM = 1e9
        
class NTCrossEntropyLoss(nn.Module):
    def __init__(self, temperature, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
        self.device = device
        
    def get_cosine_similarity_matrix(self, x, y):
        x_norm = x / x.norm(dim=-1, keepdim=True)  # (batch_size, D)
        y_norm = y / y.norm(dim=-1, keepdim=True)  # (batch_size, D)
        
        return torch.mm(x_norm, y_norm.T)  # (batch_size, batch_size)

    def forward(self, z_i, z_j):
        batch_size, D = z_i.size()  # (batch_size, representation_dim)
        z = torch.cat([z_i, z_j], dim=0)
        
        cosine_similarity = self.get_cosine_similarity_matrix(z, z)
        
        # subtract very large number from diagonal elements
        # following official implementation: https://github.com/google-research/simclr/blob/master/objective.py#L80
        cosine_similarity[range(2 * batch_size), range(2 * batch_size)] -= LARGE_NUM
        
        # apply temperature
        cosine_similarity /= self.temperature
        
        # set target
        targets = torch.LongTensor(list(range(batch_size, 2*batch_size)) + list(range(0, batch_size))).to(self.device)
        
        # Normalized Temperature-scaled Cross Entropy
        loss = self.cross_entropy(cosine_similarity, targets)
        
        pred = torch.argmax(cosine_similarity, dim=-1)
        
        return loss, pred, targets
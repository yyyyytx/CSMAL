import torch.nn as nn
import torch.nn.functional as F
import torch

def l2_norm(x):
    if len(x.shape):
        x = x.reshape((x.shape[0],-1))
    return F.normalize(x, p=2, dim=1)

def get_distance(x):
    _x = x.detach()
    sim = torch.matmul(_x, _x.t())
    dist = 2 - 2*sim
    dist += torch.eye(dist.shape[0]).to(dist.device)   # maybe dist += torch.eye(dist.shape[0]).to(dist.device)*1e-8
    dist = dist.sqrt()
    return dist

class MetricModule(nn.Module):
    def __init__(self, in_dim, embed_dim=2):
        super(MetricModule, self).__init__()
        self.fc = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
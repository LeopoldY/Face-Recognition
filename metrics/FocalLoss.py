import torch
import torch.nn as nn

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps # avoid NaN loss, this is not used in the original code
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, lables):
        logp = self.ce(input, lables)
        p = torch.exp(-logp)
        loss = (1 - p + self.eps) ** self.gamma * logp
        return loss.mean()
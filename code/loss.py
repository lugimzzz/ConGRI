import torch
import os
import random
import linecache
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.distance import PairwiseDistance


# Contrastive Loss
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)

    def forward(self, output1, output2, label):
        l2_dist = self.pdist.forward(output1, output2)
        l2_dist = l2_dist.float()
        label = label.float()
        loss_contrastive = torch.mean(label * torch.pow(l2_dist, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - l2_dist, min=0.0), 2))

        return loss_contrastive

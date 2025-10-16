import torch
import torch.nn as nn
import torch.nn.functional as F

class ESFPNetAdaptedClassifier(nn.Module):
    """
    Simplified structure of the Transformer-based classifier used
    for informative vs. uninformative frame classification.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, return_feature=False):
        feat = self.encoder(x).view(x.size(0), -1)
        out = self.fc(feat)
        if return_feature:
            return out, F.normalize(feat, dim=1)
        return out

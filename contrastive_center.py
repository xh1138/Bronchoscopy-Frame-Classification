import torch
import torch.nn.functional as F

class FeatureCenterManager:
    """
    Maintain and update class centers for contrastive learning.
    """
    def __init__(self, num_classes=2, momentum=0.9):
        self.centers = {i: None for i in range(num_classes)}
        self.momentum = momentum

    def update(self, features, labels, confidences):
        """
        Update class centers using confident samples.
        """
        for feat, label, conf in zip(features, labels, confidences):
            if conf == 1:
                label = label.item()
                feat = F.normalize(feat.detach(), dim=0)
                if self.centers[label] is None:
                    self.centers[label] = feat
                else:
                    self.centers[label] = (
                        self.momentum * self.centers[label] +
                        (1 - self.momentum) * feat
                    )

    def contrastive_loss(self, features, labels):
        """
        Compute simple contrastive distance to class centers.
        """
        total_loss, count = 0.0, 0
        for feat, label in zip(features, labels):
            label = label.item()
            pos_center = self.centers[label]
            neg_center = self.centers[1 - label]
            if pos_center is not None and neg_center is not None:
                pos_sim = F.cosine_similarity(feat, pos_center, dim=0)
                neg_sim = F.cosine_similarity(feat, neg_center, dim=0)
                total_loss += (1 - pos_sim) + neg_sim
                count += 1
        return total_loss / max(count, 1)

import torch
from torch.utils.data import Dataset

class PseudoLabeledDataset(Dataset):
    """
    Simple container for pseudo-labeled images.
    """
    def __init__(self, images, labels, confidences):
        self.images = images
        self.labels = labels
        self.confidences = confidences

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.confidences[idx]


def generate_pseudo_dataset(model, unlabeled_loader, device, threshold=0.9):
    """
    Generate pseudo-labels using model predictions on unlabeled data.
    """
    model.eval()
    pseudo_imgs, pseudo_labels, pseudo_confs = [], [], []

    with torch.no_grad():
        for imgs in unlabeled_loader:
            imgs = imgs.to(device)
            logits, _ = model(imgs, return_feature=True)
            probs = torch.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            for i in range(len(preds)):
                if max_probs[i].item() > threshold:
                    pseudo_imgs.append(imgs[i].cpu())
                    pseudo_labels.append(preds[i].cpu())
                    pseudo_confs.append(torch.tensor(1))
    return PseudoLabeledDataset(pseudo_imgs, pseudo_labels, pseudo_confs)

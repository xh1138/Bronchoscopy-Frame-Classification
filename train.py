import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

def train_epoch(model, dataloader, optimizer, criterion, center_mgr, device, epoch, contrastive_weight=0.1):
    """
    Core training loop integrating classification and contrastive loss.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels, confidences in dataloader:
        imgs, labels, confidences = imgs.to(device), labels.to(device), confidences.to(device)
        optimizer.zero_grad()
        logits, feats = model(imgs, return_feature=True)
        cls_loss = criterion(logits, labels)
        center_mgr.update(feats, labels, confidences)
        contrastive_loss = center_mgr.contrastive_loss(feats, labels)
        loss = cls_loss + contrastive_weight * contrastive_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch}: loss={total_loss:.4f}, acc={acc:.4f}")
    return total_loss / len(dataloader)

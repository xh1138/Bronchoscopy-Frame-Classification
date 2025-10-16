import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels, _ in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _ = model(imgs, return_feature=True)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return total_loss / len(dataloader), precision, recall, f1

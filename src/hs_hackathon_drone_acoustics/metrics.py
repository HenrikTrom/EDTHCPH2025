import torch
from texttable import Texttable
from torch import Tensor, nn

from hs_hackathon_drone_acoustics import CLASSES


def evaluate(all_probas: Tensor, all_targets: Tensor) -> tuple[float, float, Tensor]:
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    all_preds = []
    for proba, target in zip(all_probas, all_targets, strict=True):
        loss = criterion(proba, target)
        running_loss += loss.item()
        _, pred = proba.max(dim=0)
        correct += pred == target
        all_preds.append(pred)
    avg_loss = running_loss / len(all_targets)
    accuracy = correct / len(all_targets)
    cm = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int)
    with torch.no_grad():
        for pred, target in zip(all_preds, all_targets, strict=True):
            cm[target.item(), pred.item()] += 1
    return avg_loss, accuracy, cm


def get_confusion_matrix_str(cm: Tensor) -> str:
    table = Texttable()
    table.header([r"Actual\Pred"] + CLASSES)
    for i, row in enumerate(cm):
        table.add_row([CLASSES[i]] + list(row))
    return str(table.draw())

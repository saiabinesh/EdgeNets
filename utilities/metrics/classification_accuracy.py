#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

import torch

def accuracy(output, target, topk=(1,), num_classes=None):
    """Computes the precision@k for the specified values of k and per-class accuracy if requested."""
    maxk = max(topk)
    batch_size = target.size(0)

    # Get top-k predictions
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    # Per-class accuracy
    per_class_acc = None
    if num_classes is not None:
        per_class_correct = torch.zeros(num_classes, dtype=torch.int64, device=output.device)
        per_class_total = torch.zeros(num_classes, dtype=torch.int64, device=output.device)

        for label, prediction in zip(target, pred[0]):  # Only consider top-1 predictions
            per_class_total[label] += 1
            if label == prediction:
                per_class_correct[label] += 1

        per_class_acc = {
            i: (100.0 * per_class_correct[i].item() / per_class_total[i].item() if per_class_total[i] > 0 else 0.0)
            for i in range(num_classes)
        }

    return res, per_class_acc

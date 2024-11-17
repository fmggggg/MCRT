import torch
from torchmetrics import Metric


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )

        if len(target.shape) > 1 and target.shape[1] > 1:  # muilti-hot
            preds = torch.sigmoid(logits) > 0.5  
            correct = torch.sum((preds == target).all(dim=1)).float() # only all correct can be considered correct prediction
            total = target.size(0)
        else:
            if len(logits.shape) > 1:
                preds = logits.argmax(dim=-1)
            else:  # binary accuracy
                preds = (logits >= 0.5).float()

            preds = preds[target != -1]  # invalid samples = -1
            target = target[target != -1]

            if target.numel() == 0:
                return 1  

            correct = torch.sum(preds == target).float()
            total = target.numel()
            
        self.correct += correct
        self.total += total

    def compute(self):
        return self.correct / self.total if self.total > 0 else torch.tensor(0.0)


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total
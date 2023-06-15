import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, weight=None):
        super(Loss, self).__init__()
        self.mlsm_loss = nn.MultiLabelSoftMarginLoss(weight=weight)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, cls_logits, seg_logits, cls_label, seg_label):
        return (self.mlsm_loss(cls_logits, cls_label),
                self.ce_loss(seg_logits, seg_label))

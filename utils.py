import torch
import torch.nn as nn


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    def forward(self, label, logits):
        # label: [batch_size, conv_len]
        # logits: [batch_size, conv_len, num_class]
        batch_size, conv_len = label.shape
        label = torch.flatten(label)
        logits = logits.contiguous().view(batch_size*conv_len, -1)
        loss = self.loss_fn(logits, label)
        return loss


class MaskedBCELoss1(nn.Module):
    def __init__(self):
        super(MaskedBCELoss1, self).__init__()
        self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, label, logits, mask):
        # label: [batch_size, conv_len, conv_len]
        # logits: [batch_size, conv_len, conv_len]
        # mask: [batch_size, conv_len, conv_len]
        label = label.flatten()
        logits = logits.flatten()
        mask = mask.flatten()
        loss = self.loss_fn(logits, label)
        loss = (loss * mask).sum() / mask.sum()
        return loss


class MaskedBCELoss2(nn.Module):
    def __init__(self):
        super(MaskedBCELoss2, self).__init__()
        self.loss_fn = nn.BCELoss(reduction='mean')

    def forward(self, label, logits, mask):
        # label: [batch_size, conv_len, conv_len]
        # logits: [batch_size, conv_len, conv_len]
        # mask: [batch_size, conv_len, conv_len]
        mask_ = mask.eq(1)
        label = torch.masked_select(label.float(), mask_)
        logits = torch.masked_select(logits, mask_)
        loss = self.loss_fn(logits, label)
        return loss

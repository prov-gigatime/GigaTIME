import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # import pdb;pdb.set_trace()
        ## input.shape torch.Size([2, 23, 512, 512])
        ## target.shape torch.Size([2, 23, 512, 512]) values: 0 or 1
        bce = F.binary_cross_entropy_with_logits(input, target)
        ## bce - single tensor with grad
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0) ##just get batch size
        
        input = input.reshape(num, -1) ## torch.Size([2, 6029312])
        target = target.reshape(num, -1) ## torch.Size([2, 6029312])
        intersection = (input * target) ## torch.Size([2, 6029312])

        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth) ## torch.Size([2]) 
        dice = 1 - dice.sum() / num ## single value
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

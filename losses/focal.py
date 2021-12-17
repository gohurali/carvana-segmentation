import torch

class FocalLoss(torch.nn.Module):
    def __init__(self,use_rw=False,weights=None):
        super().__init__()
        if(use_rw):
            self.bce_loss = torch.nn.BCEWithLogitsLoss(
                reduction='none',
                pos_weight=torch.tensor(weights)
            )
        else:
            self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
            
    def forward(self,pred,target):
        bce_loss = self.bce_loss(pred,target)
        classes = 1-(2*target)
        p_t = torch.sigmoid(pred*classes)
        floss = (p_t * bce_loss).mean() / p_t.mean()
        return floss
    
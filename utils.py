import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def iou_metric(bbox_preds, bbox_targets):
    '''
    Function to calculate intersection over union metric for
    predicted bounding boxes. Takes tensor of target coordinates containing
    max and min X and Y values and tensor of target coordinates of the
    same format
    '''
    bbox_preds = bbox_preds.to(device)
    bbox_targets = bbox_targets.to(device)

    #Intersection coordinates
    x_1 = torch.max(bbox_preds[:,0], bbox_targets[:,0])
    y_1 = torch.max(bbox_preds[:,1], bbox_targets[:,1])
    x_2 = torch.min(bbox_preds[:,2], bbox_targets[:,2])
    y_2 = torch.min(bbox_preds[:,3], bbox_targets[:,3])

    zero_tensor = torch.zeros(x_1.shape[0]).to(device)

    #Area of intersection
    area_inter = torch.max(x_2 - x_1 + 1, zero_tensor) * torch.max(y_2 - y_1 + 1, zero_tensor)

    #Area of predicted and target bounding boxes
    preds_area = (bbox_preds[:,2] - bbox_preds[:,0] + 1) * (bbox_preds[:,3] - bbox_preds[:,1] + 1)
    targets_area = (bbox_targets[:,2] - bbox_targets[:,0] + 1) * (bbox_targets[:,3] - bbox_targets[:,1] + 1)

    total_area = (preds_area + targets_area - area_inter)

    iou = torch.mean(area_inter / total_area)
    
    return iou

#Dice loss class for model training
class DiceLoss(nn.Module):
    '''Class to record dice loss for image segmentation given tensors of target and predicted masks.'''
    def __init__(self):
        super().__init__()

    def forward(self, mask_preds, mask_targets, eps=1):
        
        mask_preds = torch.sigmoid(mask_preds)       
        
        #Flatten predictions and targets
        mask_preds = mask_preds.view(-1)
        mask_targets = mask_targets.view(-1)
        
        #Intersection
        inter = (mask_preds * mask_targets).sum()   

        #Dice coefficient including eps to avoid division by 0                         
        dice = ((inter * 2.0) + eps)/(mask_preds.sum() + mask_targets.sum() + eps) 

        dice_loss = 1 - dice

        return dice_loss

#Dice score function for image segmentation
def dice_score(mask_preds, mask_targets, eps=1):
    '''Function to calculate dice coefficient for image segmentation given tensors of target and predicted masks.'''
    #Dimension
    dim = mask_preds.size(0)

    mask_preds = mask_preds.view(dim, -1).float()
    mask_targets = mask_targets.view(dim, -1).float()
    
    #Intersection
    inter = (mask_preds * mask_targets).sum().float()

    dice_score = ((inter * 2.0) + eps)/(mask_preds.sum() + mask_targets.sum() + eps) 
    
    return dice_score

class UncertaintyLoss(nn.Module):
    '''Uncertainty weighted loss from https://arxiv.org/pdf/1705.07115.pdf'''
    def __init__(self):
        super().__init__()
        #Log variances
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, seg_loss, bin_loss, bbox_loss):
        prec_seg = torch.exp(-1 * self.log_vars[0])
        seg_loss = prec_seg * seg_loss + self.log_vars[0]

        prec_bin = torch.exp(-1 * self.log_vars[1])
        bin_loss = prec_bin * bin_loss + self.log_vars[1]

        prec_bbox = torch.exp(-1 * self.log_vars[2])
        bbox_loss = prec_bbox * bbox_loss + self.log_vars[2]

        total_loss = seg_loss + bin_loss + bbox_loss

        return total_loss

        

        



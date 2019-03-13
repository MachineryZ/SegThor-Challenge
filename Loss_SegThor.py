import torch

import torch.nn as nn
import numpy as np

from torch.autograd import Function
from itertools import repeat
from torch.autograd import Variable



def Dice_Error_Hard(result, target, target_class, total_classes):
    
    
    epsilon = 1e-6
    result = torch.max(result, 0)[1]
    # The "torch.max" function returns to a tuple:
    # (max, max_index)
    # Sift the chosen classes
    if total_classes == 2:
        # If it is the organs:
        if target_class != 0:
            result = (result == 1)
            target = (target == target_class)
        # If it is the background:
        elif target_class == 0:
            result = (result == 0)
            target = ((target != target_class) == 1)
    elif total_classes > 2:
        result = (result == target_class)
        target = (target == target_class)
    intersect = torch.sum((target * result).float())
    target_sum = torch.sum(target.float())
    result_sum = torch.sum(result.float())
    result_sum.requires_grad_(requires_grad = True)

    IoU = (intersect + epsilon) / (target_sum + result_sum + epsilon)
    IoU.requires_grad_(requires_grad = True)
    return 1.0 - 2.0 * IoU


class Multi_Soft_Dice_Loss(nn.Module):
    def __init__(self):
        super(Multi_Soft_Dice_Loss, self).__init__()

    def forward(self, result, target, target_class, total_classes = 5, beta = 0.75):
        loss = 0.0
        softmax = nn.Softmax(dim = 1)
        result = softmax(result)
        for i in range(result.size(0)):
            epsilon = 1e-6
            if total_classes == 2:
                result_sum = torch.sum(result[i, 1, :, :, :])
                target_sum = torch.sum(target[i, :, : ,:] == target_class)
                intersect = torch.sum(result[i, 1, :, :, :] * (target[i, :, :, :] == target_class).float())
                dice = (2 * intersect + epsilon) / (target_sum + result_sum + epsilon)
                print("The batch {}'s dice is {}".format(i, dice))
                loss += 1 - dice

            elif total_classes == 5:
                Loss = []
                weight = [2, 0.4, 0.9, 0.8]
                for j in range(1, total_classes):
                    result_sum = torch.sum(result[i, j, :, :, :])
                    target_sum = torch.sum(target[i, :, :, :] == j)
                    intersect = torch.sum(result[i, j, :, :, :] * (target[i, :, :, :] == j).float())
                    dice = (2 * intersect + epsilon) / (target_sum + result_sum + intersect + epsilon)
                    print("The {} batch's {} class's dice is {}".format(i, j, dice))
                    Loss.append(1 - dice)
                for i in range(4):
                    loss += Loss[i] * weight[i]


            elif total_classes == 4:
                weight = [2, 1.0, 0.6]
                result_sum = torch.sum(result[i, 1, :, :, :])
                target_sum = torch.sum(target[i, :, :, :] == 1)
                #print("target ecophagus's number is ", target_sum)
                intersect = torch.sum(result[i, 1, :, :, :] * ((target[i, :, :, :] == 1).float()))
                dice = (2 * intersect + epsilon) / (target_sum + result_sum + intersect + epsilon)
                print("The {} batch's Ecophagus's dice is {}".format(i, dice))
                loss += (1 - dice) * 2

                result_sum = torch.sum(result[i, 2, :, :, :])
                target_sum = torch.sum(target[i, :, :, :] == 3)
                #print("target airway's number is ", target_sum)
                intersect = torch.sum(result[i, 2, :, :, :] * ((target[i, :, :, :] == 3).float()))
                dice = (2 * intersect + epsilon) / (target_sum + result_sum + intersect + epsilon)
                print("The {} batch's Airway's dice is {}".format(i, dice))
                loss += (1 - dice) * 1.0

                result_sum = torch.sum(result[i, 3, :, :, :])
                target_sum = torch.sum(target[i, :, :, :] == 4)
                #print("target aorta's number is ", target_sum)
                intersect = torch.sum(result[i, 3, :, :, :] * ((target[i, :, :, :] == 4).float()))
                dice = (2 * intersect + epsilon) / (target_sum + result_sum + intersect + epsilon)
                print("The {} batch's Aorta's dice is {}".format(i, dice))
                loss += (1 - dice) * 0.6
        return loss/result.size(0)



class MyCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MyCrossEntropyLoss, self).__init__()

    def forward(self, result, target, target_class, total_classes):
        loss = 0.0
           
        if total_classes == 2:
            weights = torch.FloatTensor([0.5, 1.0]).cuda()
            cross_entropy_loss = nn.CrossEntropyLoss(weight = weights, reduce = False, reduction = 'mean')
            celoss = cross_entropy_loss(result, target)
            print("The Cross Entropy Loss is {}".format(celoss.mean()))
            loss += celoss.mean()

        elif total_classes == 5:
            weights = torch.FloatTensor([0.5, 1.0, 1.0, 1.0, 1.0]).cuda()
            cross_entropy_loss = nn.CrossEntropyLoss(weight = weights, reduce = False, reduction = 'mean')
            celoss = cross_entropy_loss(result, target)
            print("The Cross Entropy Loss is {}".format(celoss.mean()))
            loss += celoss.mean()

        return loss





class Multi_Dice_Loss(nn.Module):
    def __init__(self):
        super(Multi_Dice_Loss, self).__init__()
        
    def forward(self, result, target, target_class, total_classes = 5):
        # The result has not gone through the softmax layer:
        # result size: [num_classes, z, x, y]
        # target size: [z, x, y]
        if total_classes == 5:
            weights = [0.1, 1.0, 1.0, 1.0, 1.0]
        elif total_classes == 2:
            weights = [0.5, 1]

        # Compute each class's Dice Loss
        loss = 0.0
        if total_classes == 5:
            for i in range(0, total_classes):
                loss_i = Dice_Error_Hard(result, target, i, 5)
                loss += loss_i * weights[i]
                print("The Dice of class {} is: {}".format(i, (1-loss_i)/2))
            return loss

        elif total_classes == 2:
            loss_0 = Dice_Error_Hard(result, target, target_class = 0, total_classes = 2)
            #loss += loss_0 * weights[0]
            loss_target = Dice_Error_Hard(result, target, target_class = 2, total_classes = 2)
            loss += loss_target * weights[1]
            print("The Dice loss of class {} is: {}".format(0, loss_0))
            print("The Dice loss of class {} is: {}".format(target_class, loss_target))
            return loss


class Focal_Loss(nn.Module):
    def __init__(self):
        super(Focal_Loss, self).__init__()


    def forward(self, result, target, gamma = 3):
        # The focal loss function should be:
        # Focal_loss = - alpha_t * (1 - p_t)^gamma * log(pt)
        weights = torch.FloatTensor([0.1, 1.0, 1.0, 1.0, 1.0]).cuda()
        cross_entropy_loss = nn.CrossEntropyLoss(weight = weights)
        # Attention!: the nn.CrossEntropyLoss contains 2 parts:
        # 1.logsoftmax layer
        # 2.nll_loss

        cross_entropy_loss = cross_entropy_loss(result, target)
        focal_loss = cross_entropy_loss * ((1 - result) ** gamma)
        focal_loss = focal_loss.mean()

        print("The Focal loss is {}".format(focal_loss))

        return focal_loss



'''
Author: Zhizuo Zhang, Peize Zhao, Xinglong Liu, Ning Huang

'''
import os
import time
import argparse
import matplotlib
import torch
import math
import adabound
import Loss_SegThor
import ResVNet
import Evaluation_SegThor
import Model_SegThor

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
import SimpleITK as sitk
import DataLoader_SegThor
import torch.nn.init as init

from DataLoader_SegThor import SegThorDatasetTriplet
from DataLoader_SegThor import SegThorDatasetHeart
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Optimizer

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
parser = argparse.ArgumentParser(description = 'VNet for SegThor Dataset')
parser.add_argument('--beta1', type = float, default = 0.9)
parser.add_argument('--beta2', type = float, default = 0.999)
parser.add_argument('--crop-size-triplet', type = int, default = [128,128,128], nargs = '*',  help = 'Parameter of the crop-size for three organs')
parser.add_argument('--crop-size-heart', type = int, default = [128,128,128], nargs = '*',  help = 'Parameter of the crop-size for heart')

parser.add_argument('--cuda', type = bool, default = True, help = 'enables CUDA training (default: True)')
parser.add_argument('--data-folder', type = str, default = '/mnt/lustre/zhangzhizuo/Data/SegThor')#'/mnt/lustre/zhangzhizuo/Data/SegThor', metavar = 'str', help = 'Folder that holds the .npz file to train or test')
parser.add_argument('--gaussian', type = float, default = [0.0, 0.01], nargs = '*', help = 'Parameter of the uniform distribution to generate the sigma of Gaussian Noise')
parser.add_argument('--log-interval', type = int, default = 1, metavar = 'N', help = 'batchers to wait before logging training status')
parser.add_argument('--loss', default = 'soft_dice_loss', type = str, help = 'Manual choose the loss function', choices = ('focalloss','crossentropyloss','diceloss'))
parser.add_argument('--lr', type = float, default = 1e-03, metavar = 'LR', help = 'Learning Rate (default: 1e-03')
parser.add_argument('--max-epoch', type = int, default = 400, metavar = 'N', help = 'Maximum epochs to train the models (default: 10)')
parser.add_argument('--model-choose', default = 'ResVNet_Triplet', type = str, help = 'Manual choose of the model')
#parser.add_argument('--momentum', type = float, default = 0.95, metavar = 'float', help = 'The parameter of SGD optimizer (default: 0.95)')
parser.add_argument('--optimizer', type = str, default = 'Adam', metavar = 'str', choices = ('SGD', 'Adam', 'RMSprop', 'AdaBound'), help = 'Choose of Optimizer (default: SGD)')
parser.add_argument('--resume', default = '', type = str, metavar = 'PATH', help = 'Path to latest chechpoint (default: none)')
parser.add_argument('--test-batch-size', type = int, default = 1, metavar = 'N', help = 'input batch size for testing (default: 64)')
#parser.add_argument('--train', action = 'store_true', default = False, help = 'Argument to train model (default: False)')
parser.add_argument('--train-batch-size', type = int, default = 4, metavar = 'N', help = 'input batch size for training (default:64)')
parser.add_argument('--valid-batch-size', type = int, default = 1, metavar = 'N', help = 'input batch size for validing (default:4)')


# Default PyTorch Version is 1.0.0
print(torch.__version__)

"""

The Mask Value is:
0 : Background
1 : Ecophagus (食道)
2 : Heart
3 : Trachea (气管)
4 : Aorta (主动脉)

"""


# Init the weights in the net
def kaiming_normal(net, a=0, mode='fan_in', nonlinearity='relu'):
    for m in net.modules():
        if isinstance(m, (nn.modules.conv._ConvNd, nn.Linear)):
            init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            if m.bias is not None:
                init.constant_(m.bias, 0.)
        else:
            pass
    return net


# Building the Net
def Build_Net(args):
    print('Building the {}'.format(args.model_choose))
    time0 = time.time()
    if args.model_choose == 'ResVNet_ASPP_Heart':
        model = ResVNet.ResVNet_ASPP_Heart()
    elif args.model_choose == 'ResVNet_ASPP_Triplet':
        model = ResVNet.ResVNet_ASPP_Triplet()
    elif args.model_choose == 'ResVNet_Heart':
        model = ResVNet.ResVNet_Heart()
    elif args.model_choose == 'ResVNet_Triplet':
        model= Model_SegThor.ResVNet_Triplet(out_channel = 4, num_init_features = 16)

    if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()

    time1 = time.time()
    print('Spent {} time to build the net'.format(("%.4f") % (time1 - time0)))
    print('The net\'s parameters are {} '.format(sum(param.numel() for param in model.parameters())))
    return model

# Adjust the learning rate during training
def Adjust_Learning_Rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.optimizer == 'Adam':
    	# Suggested lr for Adam is args.lr = 1e-4
        warm_up_epoch = 20
        if epoch <= warm_up_epoch:
            lr = (args.lr / warm_up_epoch) * epoch
        elif epoch > warm_up_epoch:
            lr = args.lr * (0.4 ** (epoch // 50))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return

    elif args.optimizer == 'AdaBound':
    	# Suggested lr for adabound is args.lr = 1e-5
        warm_up_epoch = 15
        if epoch <= warm_up_epoch:
            lr = (args.lr / warm_up_epoch) * epoch
        elif epoch > warm_up_epoch:
            lr = args.lr * (0.4 ** (epoch // 50))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return




def Train_Triplet(args, epoch, model, optimizer, train_loader, total_classes, target_class):
    for batch_idx, (image, mask) in enumerate(train_loader):

        if args.cuda:
            image, mask = image.cuda(), mask.cuda()
        
        # The output size: [batchsize,channel = num_classes = 4, crop_z, crop_x, crop_y]
        # Attention! The output has not gone through the softmax layer!
        output = model(image)
        print(output.size())



        # original mask size: [batch_size, channel=1, crop_z, crop_x, crop_y]
        mask = mask[:, 0, :, :, :].long().cuda()
        # mask size: [batch_size, crop_z, crop_x, crop_y]
        # Because the crossentropyloss need the target to be 4D Long type Tensor, so 


        # The definition of loss
        Loss = 0.0

        if args.loss == 'cross_entropy_loss':
            print("The {} epoch's result: ".format(epoch))
            weights = torch.cuda.FloatTensor([0.1, 1.0, 1.0, 1.0, 1.0])
            cross_entropy_loss = nn.CrossEntropyLoss(weight = weights)
            for i in range(output.size(0)):
                temp_output = output[i, :, :, :, :]
                temp_mask = mask[i, :, :, :]
                Loss += cross_entropy_loss(temp_output, temp_mask)                
            Loss = Loss/output.size(0)
            

        elif args.loss == 'soft_dice_loss':
            soft_dice_loss = Loss_SegThor.Multi_Soft_Dice_Loss()
            Loss += soft_dice_loss(output, mask, total_classes = total_classes, target_class = target_class)

        elif args.loss == 'hard_dice_loss':
        	Loss = 0.0

        elif args.loss == 'focal_loss':
            loss = Loss_SegThor.Focal_Loss()
            temp_mask = (mask == 2).long()
            Loss += loss(output, temp_mask)
            Loss = Loss/output.size(0)

        Adjust_Learning_Rate(args, optimizer, epoch)
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage {}: {:6f}'
                  .format(epoch, (batch_idx + 1) * len(image), len(train_loader.dataset),
                         100. *(batch_idx + 1) / len(train_loader), args.loss, Loss.data))



def Valid_Triplet(args, model, valid_loader, target_class, total_classes = 4):
    for batch_idx, (image, mask, ID) in enumerate(valid_loader):
        ID = int(np.array(ID))
        print("\n")
        print("The No.{}'s validation result : ".format(ID))
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()

        mask = mask[0, 0, :, :, :]
        mask = mask.cpu().data.numpy()
        result = model(image)
        softmax = nn.Softmax(dim=1)
        result = softmax(result)
        result = torch.max(result, 1)[1]
        result = result[0,:,:,:]
        result = resul.cpu().data.numpy()



        
        #center_z, center_y, center_x = 0, 0, 0

        #Z, Y, X = mask.shape[0], mask.shape[1], mask.shape[2]
        '''
        for z in range(Z):
            if (mask[z, :, :] == 1).sum() != 0 or (mask[z, :, :] == 3).sum() != 0 \
                or (mask[z, :, :] == 4).sum() != 0:
                center_z += z
                break
        for z in reversed(range(Z)):
            if (mask[z, :, :] == 1).sum() != 0 or (mask[z, :, :] == 3).sum() != 0 \
                or (mask[z, :, :] == 4).sum() != 0:
                center_z += z
                break
        center_z = center_z//2

        for y in range(Y):
            if (mask[:, y, :] == 1).sum() != 0 or (mask[:, y, :] == 3).sum() != 0 \
                or (mask[:, y, :] == 4).sum() != 0:
                center_y += y
                break
        for y in reversed(range(Y)):
            if (mask[:, y, :] == 1).sum() != 0 or (mask[:, y, :] == 3).sum() != 0 \
                or (mask[:, y, :] == 4).sum() != 0:
                center_y += y
                break
        center_y = center_y//2

        for x in range(X):
            if (mask[:, :, x] == 1).sum() != 0 or (mask[:, :, x] == 3).sum() != 0 \
                or (mask[:, :, x] == 4).sum() != 0:
                center_x += x
                break
        for x in reversed(range(X)):
            if (mask[:, :, x] == 1).sum() != 0 or (mask[:, :, x] == 3).sum() != 0 \
                or (mask[:, :, x] == 4).sum() != 0:
                center_x += x
                break
        center_x = center_x//2
        '''
        """
        center_z = int(0.47 * Z)
        center_y = int(0.5 * Y)
        center_x = int(0.5 * Z)
        # Here we need crop 8 times to get the area of interest
        dir = [[0, 1, 1], [0, 1, -1], [0, -1, -1], [0, -1, 1],
                [2, 1, 1], [2, 1, -1], [2, -1, -1], [2, -1, 1],
                [-2, 1, 1], [-2, 1, -1], [-2, -1, -1], [-2, -1, 1]]
        result = np.zeros((Z, Y, X))
        for i in range(12):
            temp_center_z = center_z + dir[i][0] * (args.crop_size_triplet[0]//2)
            temp_center_y = center_y + dir[i][1] * (args.crop_size_triplet[1]//2)
            temp_center_x = center_x + dir[i][2] * (args.crop_size_triplet[2]//2)
            temp_center_z, temp_center_y, temp_center_x = int(temp_center_z), int(temp_center_y), int(temp_center_x)
            temp_input = image[:, :, (temp_center_z - args.crop_size_triplet[0]//2):(temp_center_z + args.crop_size_triplet[0]//2), (temp_center_y - args.crop_size_triplet[1]//2):(temp_center_y + args.crop_size_triplet[1]//2), (temp_center_x - args.crop_size_triplet[2]//2):(temp_center_x + args.crop_size_triplet[2]//2)]
            temp_output = model(temp_input)
            
            softmax = nn.Softmax(dim = 1)
            temp_output = softmax(temp_output)
            temp_output = torch.max(temp_output, 1)[1]
            temp_output = temp_output[0, :, :, :]
            temp_output = temp_output.cpu().data.numpy()

            result[(temp_center_z - args.crop_size_triplet[0]//2):(temp_center_z + args.crop_size_triplet[0]//2), (temp_center_y - args.crop_size_triplet[1]//2):(temp_center_y + args.crop_size_triplet[1]//2), (temp_center_x - args.crop_size_triplet[2]//2):(temp_center_x + args.crop_size_triplet[2]//2)] \
                = temp_output
                """

        print("Processing The Hausdorff Distance & Dice result. Saving the mhd file")
        Evaluation_SegThor.Hausdorff_Distance(result, mask, target_class = -1, total_classes = 4)
        
        sitk_output = sitk.GetImageFromArray(result, isVector = False)
        sitk_output.SetSpacing([1,1,1])
        sitk.WriteImage(sitk_output, '/mnt/lustrenew/zhangzhizuo/Output/output_triplet_{}_{}.mhd'.format(ID, epoch))

        sitk_mask = sitk.GetImageFromArray(mask, isVector = False)
        sitk_mask.SetSpacing([1,1,1])
        sitk.WriteImage(sitk_mask, '/mnt/lustrenew/zhangzhizuo/Output/mask_triplet_{}_{}.mhd'.format(ID, epoch))




args = parser.parse_args() # we cannot use this command in the ipython file
#args, _ = parser.parse_known_args() # in the ipython file




##############################    Ecophagus & Airway & Aorta    ##############################


args.model_choose = 'ResVNet_Triplet'
model_triplet = Build_Net(args)
model_triplet = kaiming_normal(model_triplet)
#model.apply(weights_init)
# Loading Data in the Dataset
print("Start to load the SegThor Trainning Data for Triplet!")

args.data_folder = "/mnt/lustre/zhangzhizuo/Data/SegThor2"
data_folder = args.data_folder
train_set = SegThorDatasetTriplet(dataset_folder = data_folder, phase = 'train', crop_size = args.crop_size_triplet)
train_loader = DataLoader(train_set, batch_size = args.train_batch_size, shuffle = True, num_workers = 1)
print("Training Data Numbers: ", len(train_loader.dataset))


print("Start to load the SegThor Validation Data")
valid_set = SegThorDatasetTriplet(data_folder, phase = 'valid', crop_size = args.crop_size_triplet)
valid_loader = DataLoader(valid_set, batch_size = args.valid_batch_size, shuffle = False, num_workers = 1)
print("Valid Data Numbers: ", len(valid_loader.dataset))

#Build the model for Heart training


# Define the Optimizer
if args.optimizer == 'SGD':
    optimizer = optim.SGD(model_triplet.parameters(), lr = args.lr, momentum = args.momentum)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model_triplet.parameters(), lr = args.lr, betas=(args.beta1, args.beta2), eps=1e-08, weight_decay = 0, amsgrad = False)
elif args.optimizer == 'RmsProp':
    optimizer = optim.RMSprop(model_triplet.parameters(), weight_decay = args.weight_decay)
elif args.optimizer == 'AdaBound':
    optimizer = adabound.AdaBound(params = model_triplet.parameters(), lr = args.lr, final_lr = 0.1)


model_triplet.train()

for epoch in (range(1, args.max_epoch + 1)):
    Train_Triplet(args, epoch, model_triplet, optimizer, train_loader, total_classes = 4, target_class = -1)
    if int(epoch%10) == 0:
        model_triplet.eval()
        Valid_Triplet(args, model_triplet, valid_loader, target_class = -1, total_classes = 4)
        model_triplet.train()
    if int(epoch % 50) == 0:
        torch.save(model_triplet, '/mnt/lustrenew/zhangzhizuo/Model/{}__{}__{}__{}.pkl'.format(args.model_choose, time.strftime('%Y.%m.%d.%H.%I.%M.%S',time.localtime(time.time())), epoch, 'Triplet'))

#########################################################################################






































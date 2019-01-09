'''
Author: Zhizuo Zhang, Peize Zhao, Xinglong Liu, Ning Huang

'''
import time
import argparse
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.io as sio
import torchvision.transforms as tr

from SegThor_DataLoader import XXX
from Models import VNet
from tqdm import tqdm
from Loss import DICELossMultiClass
import numpy as np


parser = argparse.ArgumentParser(description = 'VNet for SegThor Dataset')
parser.add_argument('--batch-size', type = int, default = 32, metavar = 'N', help = 'input batch size for training (default:64)')
parser.add_argument('--test-batch-size', type = int, default = 32, metavar = 'N', help = 'input batch size for testing (default: 64)')
parser.add_argument('--lr', type = float, default = 1e-03, metavar = 'LR', help = 'Learning Rate (default: 1e-03')
parser.add_argument('--momentum', type = float, default = 0.95, metavar = 'float', help = 'The parameter of SGD optimizer (default: 0.95)')
parser.add_argument('--train', action = 'store_true', default = False, help = 'Argument to train model (default: False)')
parser.add_argument('--crop-size', type = int, default = [64, 224, 224], nargs = '*',  help = 'Parameter of the crop-size')
parser.add_argument('--max-epochs', type = int, default = 10, metavar = 'N', help = 'Maximum epochs to train the models (default: 10)')
parser.add_argument('--cuda', action = 'action_true', default = False, help = 'enables CUDA training (default: False)')
parser.add_argument('--log-interval', type int, default = 1, metavar = 'N', help = 'batchers to wait before logging training status')
parser.add_argument('--optimizer', type = str, default = 'SGD', metavar = 'str', help = 'Choose of Optimizer (default: SGD)')
parser.add_argument('--beta1', type = float, default = 0.9)
parser.add_argument('--beta2', type = float, default = 0.9)
parser.add_argument('--data-folder', type = str, default = '/mnt/lustre/zhangzhizuo/Data/SegThorData/', metavar = 'str', help = 'Folder that holds the .npz file to train or test')

args = parser.parse_arge()
args.cuda = args.cuda and torch.cuda.is_available()

# Loading Data in the Dataset
PATH_DATA_FOLDER = args.PATH_DATA_FOLDER

set_train = SegThorDataset(PATH_DATA_FOLDER, train = True, )
train_loader = DataLoader(set_train, batch_size = args.batch_size, shuffle = Ture, num_workers = 1)
set_test = SegThorDataset(PATH_DATA_FOLDER)
test_loader = DataLoader(set_test, batch_size = args.test_batch_size, shuffle = False, num_workers = 1)

print("Training Data Numbers: ", len(train_laoder.dataset))
print("Test Data Numbers: ", len(test_loader.dataset))

# Loading the Vnet Models:
print('Building VNet')
time0 = time.time()
model = VNet()
time1 = time.time()
print('Spent {} time to build the net'.format(time1 - time0))
print('The net\'s parameters are {} '.format(sum(param.numel() for param in model.parameters())))

if args.cuda:
	model.cuda()

if args.optimizer == 'SGD':
	optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum)

if args.optimizer == 'ADAM':
	optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))

# Define Loss Function
criterion = DICELossMultiClass()

# Training
def train(epoch, loss_list):
	model.train()
	for batch_idx, (image, mask) in enumerate(train_loader):
		if args.cuda:
			image, mask = image.cuda(), mask.cuda()
		image, mask = Variable(image), Variable(mask)
		
		optimizer.zero_grad()

		output = model(image)

		loss = criterion(output, mask)
		loss_list.append(loss.data[0])

		loss.backward()
		optimizer.step()

		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%]\tAverage DICE Loss: {:6f}'.format(epoch, batch_idx * len(image), len(train_loader.dataset), 100. *batch_idx / len(train_loader), loss.data[0]))

# Testing
def test(train_accuracy = False, save_output = False):
	test_loss = 0

	if train_accuracy:
		loader = train_loader
	else:
		loader = test_loader

	for batch_idx, (image, mask) in tqdm(enumerate(loader)):
		if args.cuda:
			image, mask = image.cuda(), mask.cuda()

		image, mask = Variable(image, volatile = True), Variable(mask, volatile = True)

		output = model(image)

		maxes, out = torch.max(output, 1, keepdim = Ture)

		if save_output and (not train_accuracy):
			np.save('/mnt/luster/zhangzhizuo/Data/Output/{}-batch-{}-outs.npy'.format(args.save, batch_dix), out.data.byte().cpu.numpy())
			np.save('/mnt/luster/zhangzhizuo/Data/Output/{}-batch-{}-masks.npy'.format(args.save, batch_dix), mask.data.byte().cpu.numpy())
			np.save('/mnt/luster/zhangzhizuo/Data/Output/{}-batch-{}-images.npy'.format(args.save, batch_dix), image.data.byte().cpu.numpy())

		if save_output and train_accuracy:
			np.save('/mnt/luster/zhangzhizuo/Data/Output/{}-train-batch-{}-outs.npy'.format(args.save, batch_dix), out.data.byte().cpu.numpy())
			np.save('/mnt/luster/zhangzhizuo/Data/Output/{}-train-batch-{}-masks.npy'.format(args.save, batch_dix), mask.data.byte().cpu.numpy())
			np.save('/mnt/luster/zhangzhizuo/Data/Output/{}-train-batch-{}-images.npy'.format(args.save, batch_dix), image.data.byte().cpu.numpy())

		test_loss += criterion(output, mask).data[0]

	# Average Dice Coefficient
	test_loss /= len(loader)
	if train_accuracy:
		print('\nTraining Set: Average DICE Coefficient: {:.4f})\n'.format(test_loss))
	slse
		print('\nTest Set: Average DICE Coefficient: {:.4f})\n'.format(test_loss))

if args.train:
	loss_list = []
	for i in tqdm(range(args.epochs)):
		train(i, loss_list)
		test()

	plt.plot(loss_list)
	plt.title("VNet batch_size = {}, max epochs = {}, learning rate = {}".format(args.batch_size, args.max_epochs, args.lr))
	plt.xlabel("Number of iterations")
	plt.ylabel("Average DICE loss per batch")
	plt.savefig("/mnt/lustre/zhangzhizuo/Data/Output/{}-VNet_Loss_batch_size={}_max_epoch={}_learning_rate={}.npy".format(args.save, args.batch_size, args.epochs, args.lr), np.asarray(loss_list))
	torch.save(model.stat_dict(), 'VNet-final-{}-{}-{}'.format(args.batch_size, args.max_epochs, args.lr))

elif args.load is not None:
	model.load_state_dict(torch.load(args.load))
	test(save_output = True)
	test(train_accuracy = True)
















import torch
import os

import numpy as np
import scipy.io as sio
import torchvision.transforms as tr
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class SegThorDataset(Dataset):
	files = []
	images = []
	masks = []
	dataset_size = 0
	image_x = 0
	image_y = 0
	image_z = 0

	def __init__(self, dataset_folder, phase = 'train', transforms = None):
		assert(phase == 'train' or phase == 'valid' or phase == 'test')
		self.phase = phase
		self.images = []
		self.masks = []
		self.transform = transform
		folder = dataset_folder

		if phase == 'train':
			folder = dataset_folder + 'Train/'
		elif phase == 'valid':
			folder = dataset_folder + 'Valid/'
		elif phase == 'test':
			folder = dataset_folser + 'Test'

		for file in os.listdir(folder):
			# Load the npz files, the npz files are dictionary:
			# NPZ_file['origin']: The numpyorigin of the .nii file
			# NPZ_file['spacing_old']: The original voxel volume, e.g.: [2., 0.9, 0.9]
			# NPZ_file['spacing_old']: The resampled voxel volume, e.g.: [1., 1., 1.]
			# NPZ_file['image']: The data which has been 0~1 normalized
			# NPZ_file['mask']: The mask which contains only 0~5(background, or other organs)
			# NPZ_file['seriesUID']: The number of the data
			# NPZ_file['direction']: The direction of each axis
			# NPZ_file['pad']: The padding array of the image
			# NPZ_file['bbox_old']: The size of the original image
			# NPZ_file['bbox_new']: The size of resampled image
			#
			NPZ_file = np.load(file)
			# Append the files
			self.files.append(folder + 'files')
			# Get the Images
			self.images.append(NPZ_file['image'])
			# Get the Masks
			self.masks.append(NPZ_file['mask'])


		self.dataset_size = len(self.files)


	def __getitem__(self, index):

		image = self.images[index]
		mask = self.masks[index]

		start_time = time.time()

		if phase == 'train':
			image, mask = data_augment(image, mask)
		image, mask = crop(image, seg, crop_z, crop_x, crop_y)
		if transform is not None:
			image = self.transform(image)
			mask = self.transform(mask)

		end_time = time.time()
		print("Finished DataLoader in {} times".format(end_time - start_time))

		return image, mask


	def __len__(self):
		return len(self.images)


















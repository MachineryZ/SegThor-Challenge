import torch
import os
import time
import random
import scipy.ndimage
import torchvision.transforms 


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


class SegThorDatasetMulti(Dataset):
	files = []
	phase = None
	images = []
	masks = []
	phase = None
	dataset_folder = None
	gaussian = None
	crop_size = None

	def __init__(self, dataset_folder, phase, crop_size = [96, 96, 96], gaussian = [0, 0.01], transform = None):
		assert(phase == 'train' or phase == 'valid' or phase == 'test')
		self.phase = phase
		self.images = []
		self.masks = []
		self.transform = transform
		self.crop_size = crop_size
		self.gaussian = gaussian
		self.ID = []
		self.files = []
		folder = dataset_folder

		if self.phase == 'train':
			folder = dataset_folder + 'Train/'
		elif self.phase == 'valid':
			folder = dataset_folder + 'Valid/'
		elif self.phase == 'test':
			folder = dataset_folder + 'Test/'
		#print(folder)
		
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
			NPZ_file = np.load(folder + file)
			# Append the files
			self.files.append(folder + 'files')
			# Get the Images
			self.images.append(NPZ_file['image'])
			# Get the Masks
			self.masks.append(NPZ_file['mask'])
			# Get the number
			self.ID.append(NPZ_file['seriesUID'])
			print("Processing the {} file".format(folder + file))

		self.dataset_size = len(self.files)


	def __getitem__(self, index):

		image = self.images[index]
		mask = self.masks[index]
		ID = self.ID[index]

		if self.phase == 'train':
			image, mask = Central_Based_Crop(image, self.crop_size, parameter = [12, 16, 16]), Central_Based_Crop(mask, self.crop_size)
			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]
			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy())
			#image, mask = RandomResizedCrop(image, mask, self.crop_size)
			#image = Gaussian_Noise(image, self.gaussian)
		
		elif self.phase == 'test':
			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]
			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy())
		
		elif self.phase == 'valid':
			image = image[np.newasxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]
			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy()), torch.from_numpy(ID.astype(np.float32).copy())
	def __len__(self):
		return len(self.images)

class SegThorDatasetQuadraple(Dataset):
	def __init__(self, dataset_folder, phase = 'train', crop_size = [256, 256, 256]):
		assert(phase == 'train' or phase == 'valid' or phase == 'test')
		self.phase = phase
		self.images = []
		self.masks = []
		self.crop_size = crop_size
		self.ID = []
		self.files = []
		folder = dataset_folder

		if self.phase == 'train':
			folder = dataset_folder + 'Train/'
		elif self.phase == 'valid':
			folder = dataset_folder + 'Valid/'
		elif self.phase == 'test':
			folder = dataset_folder + 'Test/'
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
			NPZ_file = np.load(folder + file)
			# Append the files
			self.files.append(folder + 'files')
			# Get the Images
			self.images.append(NPZ_file['image'])
			# Get the Masks
			self.masks.append(NPZ_file['mask'])
			# Get the number
			self.ID.append(NPZ_file['seriesUID'])
			print("Processing the {} file".format(folder + file))
		self.dataset_size = len(self.files)


	def __getitem__(self, index):

		image = self.images[index]
		mask = self.masks[index]
		ID = self.ID[index]

		#The Heart statistically loc at central 100 / 500 on x-y, 80 / 400 on z
		#The location of Hearts is basically more stable
		if self.phase == 'train':
			Z, Y, X = image.shape[0], image.shape[1], image.shape[2]
			image = image[int(Z*0.05):int(Z*0.95), int(Y*0.15):int(Y*0.85), int(X*0.15):int(X*0.85)]
			mask = mask[int(Z*0.05):int(Z*0.95), int(Y*0.15):int(Y*0.85), int(X*0.15):int(X*0.85)]
			
			Z, Y, X = image.shape[0], image.shape[1], image.shape[2]
			image = (image * 10000)
			image = image.astype(np.int32)
			factor = [self.crop_size[0]/Z, self.crop_size[1]/Y, self.crop_size[2]/X]
			image = zoom(image, factor, order = 2, mode='constant')
			image = image.astype(np.float32)
			image = image/10000

			mask = zoom(mask, factor, order = 0, mode='constant')

			image, mask = RandomResizedCrop(image, mask, self.crop_size)
			# Random Rotation for z-axis, y-axis, x-axis
			if random.choice([True, False]):
				image, mask = Random_Rotation_3D(image, mask, (10, 5, 5))
			# Random Flip for for z-axis, y-axis, x-axis with 0.5 probability
			if random.choice([True, False]):
				image, mask = Random_Flip_3D(image, mask)
			# Random Elastic Distortion
			if random.choice([True, False]):
				image, mask = Elastic_Deformation(image, mask, image.shape[0] * 3,image.shape[0] * 0.05)

			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]
			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy())
		
		elif self.phase == 'valid':
			Z, Y, X = image.shape[0], image.shape[1], image.shape[2]
			image = image[int(Z*0.05):int(Z*0.95), int(Y*0.15):int(Y*0.85), int(X*0.15):int(X*0.85)]
			mask = mask[int(Z*0.05):int(Z*0.95), int(Y*0.15):int(Y*0.85), int(X*0.15):int(X*0.85)]
			
			Z, Y, X = image.shape[0], image.shape[1], image.shape[2]
			image = (image * 10000)
			image = image.astype(np.int32)
			factor = [self.crop_size[0]/Z, self.crop_size[1]/Y, self.crop_size[2]/X]
			image = zoom(image, factor, order = 2, mode='constant')
			image = image.astype(np.float32)
			image = image/10000

			mask = zoom(mask, factor, order = 0, mode='constant')

			image, mask = RandomResizedCrop(image, mask, self.crop_size)
			#mask = zoom(mask, factor, order = 0, mode='constant')

			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]

			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy()), torch.from_numpy(ID.astype(np.float32).copy())
		
		elif self.phase == 'test':
			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]
			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy()), torch.from_numpy(ID.astype(np.float32).copy())

	def __len__(self):
		return len(self.images)



class SegThorDatasetHeart(Dataset):
	def __init__(self, dataset_folder, phase = 'train', crop_size = [128, 128, 128]):
		assert(phase == 'train' or phase == 'valid' or phase == 'test')
		self.phase = phase
		self.images = []
		self.masks = []
		self.crop_size = crop_size
		self.ID = []
		self.files = []
		folder = dataset_folder

		if self.phase == 'train':
			folder = dataset_folder + 'Train/'
		elif self.phase == 'valid':
			folder = dataset_folder + 'Valid/'
		elif self.phase == 'test':
			folder = dataset_folder + 'Test/'
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
			NPZ_file = np.load(folder + file)
			# Append the files
			self.files.append(folder + 'files')
			# Get the Images
			self.images.append(NPZ_file['image'])
			# Get the Masks
			self.masks.append(NPZ_file['mask'])
			# Get the number
			self.ID.append(NPZ_file['seriesUID'])
			print("Processing the {} file".format(folder + file))
		self.dataset_size = len(self.files)


	def __getitem__(self, index):

		image = self.images[index]
		mask = self.masks[index]
		ID = self.ID[index]

		#The Heart statistically loc at central 100 / 500 on x-y, 80 / 400 on z
		#The location of Hearts is basically more stable
		if self.phase == 'train':
			# To Find an approximate center for heart
			center_z, center_y, center_x = Find_Heart_Center(mask)
			Z, Y, X = mask.shape[0], mask.shape[1], mask.shape[2]
			center_z = center_z + np.random.randint(-int(Z * 0.1), int(Z * 0.1))
			center_y = center_y + np.random.randint(-int(Y * 0.07), int(Y * 0.07))
			center_x = center_x + np.random.randint(-int(X * 0.07), int(X * 0.07))

			image = Normal_Crop(image, center_z, center_y, center_y, self.crop_size[0], self.crop_size[1], self.crop_size[2])
			mask = Normal_Crop(mask, center_z, center_y, center_y, self.crop_size[0], self.crop_size[1], self.crop_size[2])

			# Random Rotation for z-axis, y-axis, x-axis
			if random.choice([True, False]):
				image, mask = Random_Rotation_3D(image, mask, (10, 5, 5))
			# Random Flip for for z-axis, y-axis, x-axis with 0.5 probability
			if random.choice([True, False]):
				image, mask = Random_Flip_3D(image, mask)
			# Random Elastic Distortion
			if random.choice([True, False]):
				image, mask = Elastic_Deformation(image, mask, image.shape[0] * 3,image.shape[0] * 0.05)
			# Add Random Noise:
			"""
			if random.choice([True, False]):

			elif random.choice([True, False]):

			elif random.choice([True, False]):
			"""

			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]
			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy())
		
		elif self.phase == 'test':
			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]
			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy()), torch.from_numpy(ID.astype(np.float32).copy())
		
		elif self.phase == 'valid':
			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]
			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy()), torch.from_numpy(ID.astype(np.float32).copy())

	def __len__(self):
		return len(self.images)

class SegThorDatasetHeart2(Dataset):
	def __init__(self, dataset_folder, phase = 'train', crop_size = [128, 128, 128]):
		assert(phase == 'train' or phase == 'valid' or phase == 'test')
		self.phase = phase
		self.images = []
		self.masks = []
		self.crop_size = crop_size
		self.ID = []
		self.files = []
		folder = dataset_folder

		if self.phase == 'train':
			folder = dataset_folder + 'Train/'
		elif self.phase == 'valid':
			folder = dataset_folder + 'Valid/'
		elif self.phase == 'test':
			folder = dataset_folder + 'Test/'
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
			NPZ_file = np.load(folder + file)
			# Append the files
			self.files.append(folder + 'files')
			# Get the Images
			self.images.append(NPZ_file['image'])
			# Get the Masks
			self.masks.append(NPZ_file['mask'])
			# Get the number
			self.ID.append(NPZ_file['seriesUID'])
			print("Processing the {} file".format(folder + file))
		self.dataset_size = len(self.files)


	def __getitem__(self, index):

		image = self.images[index]
		mask = self.masks[index]
		ID = self.ID[index]

		#The Heart statistically loc at central 100 / 500 on x-y, 80 / 400 on z
		#The location of Hearts is basically more stable
		if self.phase == 'train':
			# To Find an approximate center for heart
			"""
			center_z, center_y, center_x = Find_Heart_Center(mask)
			Z, Y, X = mask.shape[0], mask.shape[1], mask.shape[2]
			center_z = center_z + np.random.randint(-int(Z * 0.1), int(Z * 0.1))
			center_y = center_y + np.random.randint(-int(Y * 0.07), int(Y * 0.07))
			center_x = center_x + np.random.randint(-int(X * 0.07), int(X * 0.07))

			image = Normal_Crop(image, center_z, center_y, center_y, self.crop_size[0], self.crop_size[1], self.crop_size[2])
			mask = Normal_Crop(mask, center_z, center_y, center_y, self.crop_size[0], self.crop_size[1], self.crop_size[2])

			#Random Rotation for z-axis, y-axis, x-axis
			if random.choice([True, False]):
				image, mask = Random_Rotation_3D(image, mask, (10, 5, 5))
			# Random Flip for for z-axis, y-axis, x-axis with 0.5 probability
			if random.choice([True, False]):
				image, mask = Random_Flip_3D(image, mask)
			# Random Elastic Distortion
			if random.choice([True, False]):
				image, mask = Elastic_Deformation(image, mask, image.shape[0] * 3,image.shape[0] * 0.05)
			# Add Random Noise:
			"""
			"""
			if random.choice([True, False]):

			elif random.choice([True, False]):

			elif random.choice([True, False]):
			"""
			Z, Y, X = image.shape[0], image.shape[1], image.shape[2]
			image = image[int(Z*0.05):int(Z*0.95), int(Y*0.15):int(Y*0.85), int(X*0.15):int(X*0.85)]
			mask = mask[int(Z*0.05):int(Z*0.95), int(Y*0.2):int(Y*0.85), int(X*0.2):int(X*0.85)]
			factor = [self.crop_size[0]/image.shape[0], self.crop_size[1]/image.shape[1], self.crop_size[2]/image.shape[2]]
			image = image * 10000
			image = image.astype(np.int32)
			image = zoom(image, factor, order = 3, mode = 'constant')
			image = image.astype(np.float32)
			image = image/10000
			mask = zoom(mask, factor, order = 0, mode = 'constant')
			if random.choice([True, False]):
				image, mask = Random_Rotation_3D(image, mask, (10, 5, 5))
			# Random Flip for for z-axis, y-axis, x-axis with 0.5 probability
			if random.choice([True, False]):
				image, mask = Random_Flip_3D(image, mask)
			# Random Elastic Distortion
			if random.choice([True, False]):
				image, mask = Elastic_Deformation(image, mask, image.shape[0] * 3,image.shape[0] * 0.05)
			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]
			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy())
		
		elif self.phase == 'test':

			image = image[int(Z*0.05):int(Z*0.95), int(Y*0.2):int(Y*0.85), int(X*0.2):int(X*0.85)]
			mask = mask[int(Z*0.05):int(Z*0.95), int(Y*0.2):int(Y*0.85), int(X*0.2):int(X*0.85)]
			factor = [self.crop_size[0]/image.shape[0], self.crop_size[1]/image.shape[1], self.crop_size[2]/image.shape[2]]
			image = image * 10000
			image = image.astype(np.int32)
			image = zoom(image, factor, order = 3, mode = 'constant')
			image = image.astype(np.float32)
			image = image/10000
			mask = zoom(mask, factor, order = 0, mode = 'constant')
			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]
			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy()), torch.from_numpy(ID.astype(np.float32).copy())
		
		elif self.phase == 'valid':
			Z, Y, X = image.shape[0], image.shape[1], image.shape[2]
			image = image[int(Z*0.05):int(Z*0.95), int(Y*0.2):int(Y*0.85), int(X*0.2):int(X*0.85)]
			mask = mask[int(Z*0.05):int(Z*0.95), int(Y*0.2):int(Y*0.85), int(X*0.2):int(X*0.85)]
			factor = [self.crop_size[0]/image.shape[0], self.crop_size[1]/image.shape[1], self.crop_size[2]/image.shape[2]]
			image = image * 10000
			image = image.astype(np.int32)
			image = zoom(image, factor, order = 3, mode = 'constant')
			image = image.astype(np.float32)
			image = image/10000
			mask = zoom(mask, factor, order = 0, mode = 'constant')
			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]
			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy()), torch.from_numpy(ID.astype(np.float32).copy())

	def __len__(self):
		return len(self.images)

class SegThorDatasetTriplet(Dataset):
	def __init__(self, dataset_folder, phase = 'train', crop_size = [128, 64, 64]):
		assert(phase == 'train' or phase == 'valid' or phase == 'test')
		self.phase = phase
		self.images = []
		self.masks = []
		self.crop_size = crop_size
		self.ID = []
		self.files = []
		folder = dataset_folder

		if self.phase == 'train':
			folder = dataset_folder + 'Train/'
		elif self.phase == 'valid':
			folder = dataset_folder + 'Valid/'
		elif self.phase == 'test':
			folder = dataset_folder + 'Test/'
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
			NPZ_file = np.load(folder + file)
			# Append the files
			self.files.append(folder + 'files')
			# Get the Images
			self.images.append(NPZ_file['image'])
			# Get the Masks
			self.masks.append(NPZ_file['mask'])
			# Get the number
			self.ID.append(NPZ_file['seriesUID'])
			print("Processing the {} file".format(folder + file))

		self.dataset_size = len(self.files)


	def __getitem__(self, index):

		image = self.images[index]
		mask = self.masks[index]
		ID = self.ID[index]

		#The Location of The Triplets are z:[0.1~0.9], y:[0.3~0.8], x:[0.3~0.8]
		#The location of Hearts is basically more stable
		if self.phase == 'train':
			# To Find an approximate center for Crop Center
			"""
			center_z, center_y, center_x = Find_Triplet_Center(mask)
			Z, Y, X = mask.shape[0], mask.shape[1], mask.shape[2]
			center_z = center_z + np.random.randint(-int(Z * 0.1), int(Z * 0.1))
			center_y = center_y + np.random.randint(-int(Y * 0.07), int(Y * 0.07))
			center_x = center_x + np.random.randint(-int(X * 0.07), int(X * 0.07))

			image = Normal_Crop(image, center_z, center_y, center_x, self.crop_size[0], self.crop_size[1], self.crop_size[2])
			mask = Normal_Crop(mask, center_z, center_y, center_x, self.crop_size[0], self.crop_size[1], self.crop_size[2])
			"""
			Z, Y, X = image.shape[0], image.shape[1], image.shape[2]
			image = image[int(Z*0.05):int(Z*0.95), int(Y*0.15):int(Y*0.85), int(X*0.15):int(X*0.85)]
			mask = mask[int(Z*0.05):int(Z*0.95), int(Y*0.15):int(Y*0.85), int(X*0.15):int(X*0.85)]
			
			Z, Y, X = image.shape[0], image.shape[1], image.shape[2]
			image = (image * 10000)
			image = image.astype(np.int32)
			factor = [1, 256/Y, 256/Y]
			image = zoom(image, factor, order = 2, mode='constant')
			image = image.astype(np.float32)
			image = image/10000

			mask = zoom(mask, factor, order = 0, mode='constant')

			image, mask = RandomResizedCrop(image, mask, self.crop_size)
			# Random Rotation for z-axis, y-axis, x-axis
			if random.choice([True, False]):
				image, mask = Random_Rotation_3D(image, mask, (10, 5, 5))
			# Random Flip for for z-axis, y-axis, x-axis with 0.5 probability
			if random.choice([True, False]):
				image, mask = Random_Flip_3D(image, mask)
			# Random Elastic Distortion
			if random.choice([True, False]):
				image, mask = Elastic_Deformation(image, mask, image.shape[0] * 3,image.shape[0] * 0.05)

			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]
			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy())
			# Add Random Noise:
			"""
			if random.choice([True, False]):

			elif random.choice([True, False]):

			elif random.choice([True, False]):
			"""

			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]
			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy())
		
		elif self.phase == 'test':
			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]
			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy())
		
		elif self.phase == 'valid':
			Z, Y, X = image.shape[0], image.shape[1], image.shape[2]
			image = image[int(Z*0.05):int(Z*0.95), int(Y*0.15):int(Y*0.85), int(X*0.15):int(X*0.85)]
			mask = mask[int(Z*0.05):int(Z*0.95), int(Y*0.15):int(Y*0.85), int(X*0.15):int(X*0.85)]
			
			Z, Y, X = image.shape[0], image.shape[1], image.shape[2]
			image = (image * 10000)
			image = image.astype(np.int32)
			factor = [1, 256/Y, 256/Y]
			image = zoom(image, factor, order = 2, mode='constant')
			image = image.astype(np.float32)
			image = image/10000

			mask = zoom(mask, factor, order = 0, mode='constant')

			image, mask = RandomResizedCrop(image, mask, self.crop_size)
			#mask = zoom(mask, factor, order = 0, mode='constant')

			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]

			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy()), torch.from_numpy(ID.astype(np.float32).copy())

	def __len__(self):
		return len(self.images)

class SegThorDatasetLocalization(Dataset):
	def __init__(self, dataset_folder, phase = 'train', zoom_size = [256, 256, 256]):
		assert(phase == 'train' or phase == 'valid')
		self.phase = phase
		self.images = []
		self.masks = []
		self.zoom_size = zoom_size
		self.ID = []
		self.files = []
		self.shape = []
		folder = dataset_folder

		if self.phase == 'train':
			folder = dataset_folder + 'Train/'
		elif self.phase == 'valid':
			folder = dataset_folder + 'Valid/'
		elif self.phase == 'test':
			folder = dataset_folder + 'Test/'
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
			NPZ_file = np.load(folder + file)
			# Append the files
			self.files.append(folder + 'files')
			# Get the Images
			self.images.append(NPZ_file['image'])
			# Get the Masks
			self.masks.append(NPZ_file['mask'])
			# Get the number
			self.ID.append(NPZ_file['seriesUID'])
			# Get Shape
			self.shape.append(NPZ_file['image'].shape)
			print("Processing the {} file".format(folder + file))
		self.dataset_size = len(self.files)


	def __getitem__(self, index):

		image = self.images[index]
		mask = self.masks[index]
		ID = self.ID[index]
		image_shape = self.shape[index]
		if self.phase == 'train':
			# To Find an approximate center for heart
			factor = [self.zoom_size[0]/image.shape[0], self.zoom_size[1]/image.shape[1], self.zoom_size[2]/image.shape[2]]
			image = image * 1800 - 1200
			image = image.astype(np.int32)
			image = zoom(image, factor, order = 2, mode = 'constant')
			image = image.astype(np.float32)
			image = (image + 1200)/1800
			mask = zoom(mask, factor, order = 0, mode = 'constant')
			# Set all mask value into 1
			mask = (mask != 0)
			# Random Rotation for z-axis, y-axis, x-axis
			if random.choice([True, False]):
				image, mask = Random_Rotation_3D(image, mask, (10, 5, 5))
			# Random Flip for for z-axis, y-axis, x-axis with 0.5 probability
			if random.choice([True, False]):
				image, mask = Random_Flip_3D(image, mask)
			# Random Elastic Distortion
			if random.choice([True, False]):
				image, mask = Elastic_Deformation(image, mask, image.shape[0] * 3,image.shape[0] * 0.05)
			

			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]
			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy())
		
		elif self.phase == 'valid':
			factor = [self.zoom_size[0]/image.shape[0], self.zoom_size[1]/image.shape[1], self.zoom_size[2]/image.shape[2]]
			image = image * 1800 - 1200
			image = image.astype(np.int32)
			image = zoom(image, factor, order = 2, mode = 'constant')
			image = image.astype(np.float32)
			image = (image + 1200)/1800
			mask = zoom(mask, factor, order = 0, mode = 'constant')
			mask = (mask != 0)
			image = image[np.newaxis, :, :, :]
			mask = mask[np.newaxis, :, :, :]
			# Set all mask value into 1
			mask = (mask != 0)
			return torch.from_numpy(image.astype(np.float32).copy()), torch.from_numpy(mask.astype(np.float32).copy()), torch.from_numpy(ID.astype(np.float32).copy())

	def __len__(self):
		return len(self.images)

def RandomResizedCrop(input, gt, crop_size = [128, 128, 128]):
	#crop_size(z, x, y)
	input_shape = input.shape
	padding = [[0, 0]]
	if crop_size[1] > input_shape[1]:
		padding.append([(crop_size[1] - input_shape[1]) / 2 + 2, (crop_size[1] - input_shape[1]) / 2 + 2])
		padding.append([(crop_size[2] - input_shape[2]) / 2 + 2, (crop_size[2] - input_shape[2]) / 2 + 2])
		input = np.pad(input, padding, 'constant', constant_value = 0)
		gt = np.pad(gt, padding, 'constant', constant_value = 0)
	after_padding_shape = input.shape
	z_center = random.randint(int(crop_size[0] / 2), after_padding_shape[0] - int(crop_size[0] / 2))
	x_center = random.randint(int(crop_size[1] / 2), after_padding_shape[1] - int(crop_size[1] / 2))
	y_center = random.randint(int(crop_size[2] / 2), after_padding_shape[2] - int(crop_size[2] / 2))
	output = input[z_center - int(crop_size[0] / 2): z_center + int(crop_size[0] / 2), x_center - int(crop_size[1] / 2) : x_center + int(crop_size[1] / 2), y_center - int(crop_size[2] / 2) : y_center + int(crop_size[2] / 2)]
	gt = gt[z_center - int(crop_size[0] / 2): z_center + int(crop_size[0] / 2), x_center - int(crop_size[1] / 2) : x_center + int(crop_size[1] / 2), y_center - int(crop_size[2] / 2) : y_center + int(crop_size[2] / 2)]
	return output, gt

def Gaussian_Noise(input, parameter = [0, 0.01]):
	if parameter[0] == 0 and parameter[1] == 0:
	# Don't add the gaussian noise
		return input
	else:
		for z in range(input.shape[0]):
			input[z] = input[z] + np.random.normal(0.0, np.random.uniform(parameter[0], parameter[1]), size = input[z].shape)
	return input

def RandomCentralBasedCropForHeart(input1, input2, crop_size, parameter = [6, 6, 6]):
	# Crop Center == Image Center +/- Random Bias
	center_z, center_x, center_y = input1.shape[0]//2, input1.shape[1]//2, input1.shape[2]//2
	a, b, c = input1.shape[0], input1.shape[1], input1.shape[2]
	a, b, c = int(a), int(b), int(c)
	z, x, y = np.random.normal(center_z, a//parameter[0]), np.random.normal(center_x, b//parameter[1]), np.random.normal(center_y, c//parameter[2])
	z, x, y = int(z), int(x), int(y)
	while(z + crop_size[0]//2 > a or z - crop_size[0]//2 < 0 or x + crop_size[1]//2 > b or x - crop_size[1]//2 < 0 or y + crop_size[2]//2 > c or y - crop_size[2]//2 < 0 or (input2[(z - crop_size[0]//2): (z + crop_size[0]//2), (x - crop_size[1]//2):(x + crop_size[1]//2), (y - crop_size[2]//2):(y + crop_size[2]//2)] == 2).sum() < (crop_size[0] * crop_size[1] * crop_size[2]) * 0.1):
		z, x, y = np.random.normal(center_z, a//parameter[0]), np.random.normal(center_x, b//parameter[1]), np.random.normal(center_y, c//parameter[2])
		z, x, y = int(z), int(x), int(y)
	return input1[(z - crop_size[0]//2): (z + crop_size[0]//2), (x - crop_size[1]//2):(x + crop_size[1]//2), (y - crop_size[2]//2):(y + crop_size[2]//2)], input2[(z - crop_size[0]//2): (z + crop_size[0]//2), (x - crop_size[1]//2):(x + crop_size[1]//2), (y - crop_size[2]//2):(y + crop_size[2]//2)]

def RandomCentralBasedCropForAorta(input1, input2, crop_size, parameter = [6, 6, 6]):
	# Crop Center == Image Center +/- Random Bias
	center_z, center_x, center_y = input1.shape[0]//2, input1.shape[1]//2, input1.shape[2]//2
	a, b, c = input1.shape[0], input1.shape[1], input1.shape[2]
	a, b, c = int(a), int(b), int(c)
	z, x, y = np.random.normal(center_z, a//parameter[0]), np.random.normal(center_x, b//parameter[1]), np.random.normal(center_y, c//parameter[2])
	z, x, y = int(z), int(x), int(y)
	while(z + crop_size[0]//2 > a or z - crop_size[0]//2 < 0 or x + crop_size[1]//2 > b or x - crop_size[1]//2 < 0 or y + crop_size[2]//2 > c or y - crop_size[2]//2 < 0 or (input2[(z - crop_size[0]//2): (z + crop_size[0]//2), (x - crop_size[1]//2):(x + crop_size[1]//2), (y - crop_size[2]//2):(y + crop_size[2]//2)] == 4).sum() < (crop_size[0] * crop_size[1] * crop_size[2]) * 0.08):
		z, x, y = np.random.normal(center_z, a//parameter[0]), np.random.normal(center_x, b//parameter[1]), np.random.normal(center_y, c//parameter[2])
		z, x, y = int(z), int(x), int(y)
	return input1[(z - crop_size[0]//2): (z + crop_size[0]//2), (x - crop_size[1]//2):(x + crop_size[1]//2), (y - crop_size[2]//2):(y + crop_size[2]//2)], input2[(z - crop_size[0]//2): (z + crop_size[0]//2), (x - crop_size[1]//2):(x + crop_size[1]//2), (y - crop_size[2]//2):(y + crop_size[2]//2)]

def RandomCentralBasedCropForEsophagus(input1, input2, crop_size, parameter = [1, 6, 6]):
	# Crop Center == Image Center +/- Random Bias
	center_z, center_x, center_y = input1.shape[0]//2, input1.shape[1]//2, input1.shape[2]//2
	a, b, c = input1.shape[0], input1.shape[1], input1.shape[2]
	a, b, c = int(a), int(b), int(c)
	z, x, y = np.random.normal(center_z, a//parameter[0]), np.random.normal(center_x, b//parameter[1]), np.random.normal(center_y, c//parameter[2])
	z, x, y = int(z), int(x), int(y)
	while(z + crop_size[0]//2 > a or z - crop_size[0]//2 < 0 or x + crop_size[1]//2 > b or x - crop_size[1]//2 < 0 or y + crop_size[2]//2 > c or y - crop_size[2]//2 < 0 or (input2[(z - crop_size[0]//2): (z + crop_size[0]//2), (x - crop_size[1]//2):(x + crop_size[1]//2), (y - crop_size[2]//2):(y + crop_size[2]//2)] == 2).sum() < (crop_size[0] * crop_size[1] * crop_size[2]) * 0.02):
		z, x, y = np.random.normal(center_z, a//parameter[0]), np.random.normal(center_x, b//parameter[1]), np.random.normal(center_y, c//parameter[2])
		z, x, y = int(z), int(x), int(y)
	return input1[(z - crop_size[0]//2): (z + crop_size[0]//2), (x - crop_size[1]//2):(x + crop_size[1]//2), (y - crop_size[2]//2):(y + crop_size[2]//2)], input2[(z - crop_size[0]//2): (z + crop_size[0]//2), (x - crop_size[1]//2):(x + crop_size[1]//2), (y - crop_size[2]//2):(y + crop_size[2]//2)]

def RandomCentralBasedCropForAirway(input1, input2, crop_size, parameter = [1, 6, 6]):
	# Crop Center == Image Center +/- Random Bias
	center_z, center_x, center_y = input1.shape[0]//2, input1.shape[1]//2, input1.shape[2]//2
	a, b, c = input1.shape[0], input1.shape[1], input1.shape[2]
	a, b, c = int(a), int(b), int(c)
	z, x, y = np.random.normal(center_z, a//parameter[0]), np.random.normal(center_x, b//parameter[1]), np.random.normal(center_y, c//parameter[2])
	z, x, y = int(z), int(x), int(y)
	while(z + crop_size[0]//2 > a or z - crop_size[0]//2 < 0 or x + crop_size[1]//2 > b or x - crop_size[1]//2 < 0 or y + crop_size[2]//2 > c or y - crop_size[2]//2 < 0 or (input2[(z - crop_size[0]//2): (z + crop_size[0]//2), (x - crop_size[1]//2):(x + crop_size[1]//2), (y - crop_size[2]//2):(y + crop_size[2]//2)] == 2).sum() < (crop_size[0] * crop_size[1] * crop_size[2]) * 0.1):
		z, x, y = np.random.normal(center_z, a//parameter[0]), np.random.normal(center_x, b//parameter[1]), np.random.normal(center_y, c//parameter[2])
		z, x, y = int(z), int(x), int(y)
	return input1[(z - crop_size[0]//2): (z + crop_size[0]//2), (x - crop_size[1]//2):(x + crop_size[1]//2), (y - crop_size[2]//2):(y + crop_size[2]//2)], input2[(z - crop_size[0]//2): (z + crop_size[0]//2), (x - crop_size[1]//2):(x + crop_size[1]//2), (y - crop_size[2]//2):(y + crop_size[2]//2)]

def FixedCentralBasedCrop(input1, input2, crop_size):
	# Crop Center == Image Center
	center_z, center_x, center_y = input1.shape[0]//2, input1.shape[1]//2, input1.shape[2]//2
	return input1[(center_z - crop_size[0]//2): (center_z + crop_size[0]//2), (center_x - crop_size[1]//2):(center_x + crop_size[1]//2), (center_y - crop_size[2]//2):(center_y + crop_size[2]//2)], input2[(center_z - crop_size[0]//2): (center_z + crop_size[0]//2), (center_x - crop_size[1]//2):(center_x + crop_size[1]//2), (center_y - crop_size[2]//2):(center_y + crop_size[2]//2)]

def Random_Rotation_3D(image, mask, max_angles = (10, 5, 5)):
	image = (image * 1800 - 1200)
	image = image.astype(np.int32)
	image1 = image
	mask1 = mask
	# rotate along z-axis
	angle = random.uniform(-max_angles[0], max_angles[0])
	image2 = rotate(image1, angle, order=2, mode='nearest', axes=(0, 1), reshape=False)
	mask2 = rotate(mask1, angle, order=0, mode='nearest', axes=(0, 1), reshape=False)

	# rotate along y-axis
	angle = random.uniform(-max_angles[1], max_angles[1])
	image3 = rotate(image2, angle, order=2, mode='nearest', axes=(0, 2), reshape=False)
	mask3 = rotate(mask2, angle, order=0, mode='nearest', axes=(0, 2), reshape=False)

	# rotate along x-axis
	angle = random.uniform(-max_angles[2], max_angles[2])
	image_rot = rotate(image3, angle, order=2, mode='nearest', axes=(1, 2), reshape=False)
	mask_rot = rotate(mask3, angle, order=0, mode='nearest', axes=(1, 2), reshape=False)

	image = image.astype(np.float32)
	image_rot = (image_rot + 1200)/1800

	return image_rot, mask_rot

def Normal_Crop(input, center_z, center_y, center_x, crop_z, crop_y, crop_x):
	return input[(center_z - crop_z//2):(center_z + crop_z//2), (center_y - crop_y//2): \
				(center_y + crop_y//2), (center_x - crop_x//2):(center_x + crop_x//2)]

def Random_Flip_3D(image, mask):
	if random.choice([True, False]):
		image = image[::-1, :, :].copy()  # here must use copy(), otherwise error occurs
		mask = mask[::-1, :, :].copy()
	if random.choice([True, False]):
		image = image[:, ::-1, :].copy()
		mask = mask[:, ::-1, :].copy()
	if random.choice([True, False]):
		image = image[:, :, ::-1].copy()
		mask = mask[:, :, ::-1].copy()

	return image, mask

def Find_Heart_Center(mask):
	z, y, x = (mask == 2).nonzero()

	midz = np.mean(z)
	midy = np.mean(y)
	midx = np.mean(x)

	return int(midz), int(midx),int(midy)

def Find_Triplet_Center(mask):
	Z, Y, X = mask.shape[0], mask.shape[1], mask.shape[2]
	center_x, center_y, center_z = 0.0, 0.0, 0.0

	for z in range(Z):
		if (mask[z, :, :] == 1).sum() != 0 or (mask[z, :, :] == 3).sum() != 0 or (mask[z, :, :] == 4).sum() != 0:
			center_z += z
			break
	for z in reversed(range(Z)):
		if (mask[z, :, :] == 1).sum() != 0 or (mask[z, :, :] == 3).sum() != 0 or (mask[z, :, :] == 4).sum() != 0:
			center_z += z
			break
	center_z = center_z//2

	for y in range(Y):
		if (mask[:, y, :] == 1).sum() != 0 or (mask[:, y, :] == 3).sum() != 0 or (mask[:, y, :] == 4).sum() != 0:
			center_y += y
			break
	for y in reversed(range(Y)):
		if (mask[:, y, :] == 1).sum() != 0 or (mask[:, y, :] == 3).sum() != 0 or (mask[:, y, :] == 4).sum() != 0:
			center_y += y
			break
	center_y = center_y//2

	for x in range(X):
		if (mask[:, :, x] == 1).sum() != 0 or (mask[:, :, x] == 3).sum() != 0 or (mask[:, :, x] == 4).sum() != 0:
			center_x += x
			break
	for x in reversed(range(X)):
		if (mask[:, :, x] == 1).sum() != 0 or (mask[:, :, x] == 3).sum() != 0 or (mask[:, :, x] == 4).sum() != 0:
			center_x += x
			break
	center_x = center_x//2
	center_z, center_y, center_x = int(center_z), int(center_y), int(center_x)
	#print("center_z = ", center_z, " center_y = ", center_y, " center_x = ", center_x)
	return int(center_z), int(center_y), int(center_x)

def Elastic_Deformation(image, mask, alpha, sigma):
	# Alpha controls the bias of the deformation, larger alpha leads to more deformation
	# Sigma controls the kernel size of the deformation, larger sigma leads to larger affect area of the kernel
	""" 
		Elastic deformation of images as described in [Simard2003]_.
		[Simard2003] Simard, Steinkraus and Platt, "Best Practices for
		Convolutional Neural Networks applied to Visual Document Analysis", in
		Proc. of the International Conference on Document Analysis and
		Recognition, 2003.
	"""
	image = (image * 10000)
	image = image.astype(np.int32)

	shape = image.shape

	dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode = 'constant', cval = 0) * alpha
	dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode = 'constant', cval = 0) * alpha
	dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode = 'constant', cval = 0) * alpha

	x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

	indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))

	image = map_coordinates(image, indices, order = 2).reshape(shape)
	mask = map_coordinates(mask, indices, order = 0).reshape(shape)

	image = image.astype(np.float32)
	image = image/10000

	return image, mask

def augment_salt_pepper_noise(data, SNR=0.9, choice=None):
    for sample_idx in range(data.shape[0]):  # for-loop in z axis
        sample = data[sample_idx]
        if choice is 'salt':
            # creat a matrix, whose size is x*y, where 90%(SNR=0.9) of this matrix is 0 and 10% is 1
            noise = np.random.choice((0, 1), size=sample.shape, p=[SNR, 1-SNR])
            sample[noise == 1] = 255
        elif choice is 'pepper':
            noise = np.random.choice((0, 1), size=sample.shape, p=[SNR, 1-SNR])
            sample[noise == 1] = 0
        elif choice is 'salt and pepper':
            # creat a matrix, whose size is x*y, where 90%(SNR=0.9) of this matrix is 0, 5% is 1 and 5% is 2
            noise = np.random.choice((0, 1, 2), size=sample.shape, p=[SNR, (1-SNR)/2.0, (1-SNR)/2.0])
            sample[noise == 1] = 0
            sample[noise == 2] = 255
        data[sample_idx] = sample

    return data

def augment_gaussian_noise(data, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])  # generate a random number between 0 and 0.1

    for sample_idx in range(data.shape[0]):
        data[sample_idx] = data[sample_idx] + np.random.normal(0.0, variance, size=data[sample_idx].shape)
    return data

def augment_rician_noise(data, noise_variance=(0, 0.1)):
    for sample_idx in range(data.shape[0]):
        sample = data[sample_idx]
        variance = random.uniform(noise_variance[0], noise_variance[1])
        sample = np.sqrt(
            (sample + np.random.normal(0.0, variance, size=sample.shape)) ** 2 + np.random.normal(0.0, variance, size=sample.shape) ** 2)
        data[sample_idx] = sample
    return data

def Random_Crop_Triplet(image, mask, crop_size):
	Z, Y, X = mask.shape[0], mask.shape[1], mask.shape[2]
	min_z, max_z, min_y, max_y, min_x, max_x = 0, 0, 0, 0, 0, 0
	center_x, center_y, center_z = 0.0, 0.0, 0.0

	for z in range(Z):
		if (mask[z, :, :] == 1).sum() != 0 or (mask[z, :, :] == 3).sum() != 0 or (mask[z, :, :] == 4).sum() != 0:
			min_z
			center_z += z
			break
	for z in reversed(range(Z)):
		if (mask[z, :, :] == 1).sum() != 0 or (mask[z, :, :] == 3).sum() != 0 or (mask[z, :, :] == 4).sum() != 0:
			max_z = z
			center_z += z
			break
	center_z = center_z//2

	for y in range(Y):
		if (mask[:, y, :] == 1).sum() != 0 or (mask[:, y, :] == 3).sum() != 0 or (mask[:, y, :] == 4).sum() != 0:
			min_y = y
			center_y += y
			break
	for y in reversed(range(Y)):
		if (mask[:, y, :] == 1).sum() != 0 or (mask[:, y, :] == 3).sum() != 0 or (mask[:, y, :] == 4).sum() != 0:
			max_y = y
			center_y += y
			break
	center_y = center_y//2

	for x in range(X):
		if (mask[:, :, x] == 1).sum() != 0 or (mask[:, :, x] == 3).sum() != 0 or (mask[:, :, x] == 4).sum() != 0:
			min_x = x
			center_x += x
			break
	for x in reversed(range(X)):
		if (mask[:, :, x] == 1).sum() != 0 or (mask[:, :, x] == 3).sum() != 0 or (mask[:, :, x] == 4).sum() != 0:
			max_x = x
			center_x += x
			break
	center_x = center_x//2
	center_z = np.random.randint(min_z, max_z)
	center_y = np.random.randint(min_y, max_y)
	center_x = np.random.randint(min_x, max_x)

	while(center_z + crop_size[0]//2 >= Z or center_z - crop_size[0]//2 <= 0
		or center_y + crop_size[1]//2 >= Y or center_y - crop_size[1]//2 <= 0
		or center_x + crop_size[2]//2 >= Z or center_z - crop_size[2]//2 <= 0):
		center_z = np.random.randint(min_z, max_z)
		center_y = np.random.randint(min_y, max_y)
		center_x = np.random.randint(min_x, max_x)

	
	return Normal_Crop(image, center_z, center_y, center_x, crop_size[0], crop_size[1], crop_size[2]), Normal_Crop(mask, center_z, center_y, center_x, crop_size[0], crop_size[1], crop_size[2])
















import os
import numpy as np
from scipy.ndimage.interpolation import zoom
import SimpleITK as sitk
from multiprocessing import Pool
from functools import partial
import time

# prepare should return the original and mask image, resampled, in npz format

def resample(imgs, spacing, new_spacing, order=2):
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode='constant', order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')

def lumTrans(img):
    lungwin = np.array([-1200., 600.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    # newimg = (newimg*255).astype('uint8')
    return newimg


def load_itk_image(filename):
    '''
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False
    '''
    isflip = False
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing, isflip

def savenpy_segthor_my(name, gt_path, data_path, save_path):
    #for data pre processing for segthor 
    print('start processing: ', name)
    resolution = np.array([1.0, 1.0, 1.0])

    #print(type(name))
    #print(name)
    #print(name.split('.nii').)    
    seriesUID = os.path.basename(name).split('.nii')[0].split('Patient_')[1] # seriesUID = XX

    gt_name = gt_path + 'GT_' + seriesUID + '.nii' # '/mnt/lustre/zhangzhizuo/Data/train/GT/GT_seriesUID.nii'
    ori_img, origin, spacing, isflip = load_itk_image(name) # ori_img : original img
    if isflip:
        ori_img = ori_img[:, ::-1, ::-1]
        print('This image needs to be flipped')
    isMask = True
    if isMask:
        mask_img, origin, spacing, isflip = load_itk_image(gt_name)
        if isflip:
            mask_img = mask_img[:, ::-1, ::-1]


    newshape = np.round(np.array(ori_img.shape) * spacing / resolution).astype('int')

    ori_img = lumTrans(ori_img)

    re_img, _ = resample(ori_img, spacing, resolution, order = 2)

    if mask_img is not None:
        mask_img, _ = resample(mask_img, spacing, resolution, order = 0)

    originalshape = ori_img.shape
    shape = re_img.shape
    pad = [[0, 0]]
    
    seg = None
    # The label need to be int, like 0, 1, 2, 3
    if isMask:
        seg = mask_img.astype(np.uint8)

    np.savez_compressed(os.path.join(save_path, seriesUID + '.npz'), origin=origin, spacing_old=spacing,
                        spacing_new=resolution,
                        image=re_img.astype(np.float16), mask=seg, seriesUID=seriesUID, direction=isflip, pad=pad,
                        bbox_old=originalshape, bbox_new=re_img.shape)
    print(seriesUID, shape)


def preprocess_segthor():
    gt_path = '/mnt/lustre/zhangzhizuo/Data/train/GT/'
    data_path = '/mnt/lustre/zhangzhizuo/Data/train/Patient'
    save_path = '/mnt/lustre/zhangzhizuo/Data/'
    print('Starting PreProcessing data')
    time0 = time.time()

    filelist = []
    dirrr = os.listdir(data_path)
    for fileee in dirrr:
        file = os.path.join(data_path, fileee)
        # file = '/mnt/lustre/zhangzhizuo/Data/train/Patient/Patient_XX.nii'
        filelist.append(file)
    for i in range(len(filelist)):
        #print(type(filelist[i]))
        #print(filelist[i])
        savenpy_segthor_my(filelist[i], gt_path, data_path, save_path)
    time1 = time.time()
    print('End Preprocessing SegThor, with time %3.2f' % (time1 - time0))

if __name__ == '__main__':
    preprocess_segthor()


























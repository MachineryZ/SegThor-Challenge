#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Data Augmentation Utility functions
"""
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.ndimage

def computeQualityMeasures(lP,lT):
    quality=dict()
    labelPred=sitk.GetImageFromArray(lP, isVector=False)
    labelTrue=sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["avgHausdorff"]=hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance()

    dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["dice"]=dicecomputer.GetDiceCoefficient()

    return quality

def getNumpyData(self,dat,method=sitk.sitkLinear):
    ret=dict()
    for key in dat:
        ret[key] = np.zeros([self.params['VolSize'][0], self.params['VolSize'][1], self.params['VolSize'][2]], dtype=np.float32)

        img=dat[key]

        #we rotate the image according to its transformation using the direction and according to the final spacing we want
        factor = np.asarray(img.GetSpacing()) / [self.params['dstRes'][0], self.params['dstRes'][1],
                                                 self.params['dstRes'][2]]

        factorSize = np.asarray(img.GetSize() * factor, dtype=float)

        newSize = np.max([factorSize, self.params['VolSize']], axis=0)

        newSize = newSize.astype(dtype=int)

        T=sitk.AffineTransform(3)
        T.SetMatrix(img.GetDirection())

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)
        resampler.SetOutputSpacing([self.params['dstRes'][0], self.params['dstRes'][1], self.params['dstRes'][2]])
        resampler.SetSize(newSize)
        resampler.SetInterpolator(method)
        if self.params['normDir']:
            resampler.SetTransform(T.GetInverse())

        imgResampled = resampler.Execute(img)


        imgCentroid = np.asarray(newSize, dtype=float) / 2.0

        imgStartPx = (imgCentroid - self.params['VolSize'] / 2.0).astype(dtype=int)

        regionExtractor = sitk.RegionOfInterestImageFilter()
        regionExtractor.SetSize(list(self.params['VolSize'].astype(dtype=int)))
        regionExtractor.SetIndex(list(imgStartPx))

        imgResampledCropped = regionExtractor.Execute(imgResampled)

        ret[key] = np.transpose(sitk.GetArrayFromImage(imgResampledCropped).astype(dtype=float), [2, 1, 0])

    return ret
def resample(image, transform):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = sitk.sitkCosineWindowedSinc
    default_value = 100.0
    return sitk.Resample(image, reference_image, transform, interpolator, default_value)

# random transform at the time of loading, avoid computer knowing the pattern in advance
#it is better to use a small patch, as 1. it is faster, 2. specific to the nodule patch, 3.
def augmentImage(itk_img, center, augmentImageSize):
    #1 identiy transform
    augmentedImage = []
    augmentedImage.append(itk_img)
    #2 random scaling transform, same in XY, different in Z, range .8~1.2
    #scale transforms * 5
    minScale = 0.7  #want more zoom out, to see the big picture
    maxScale = 1.08 #don't want too small ROI
    for i in range(0,15):
        scale = sitk.ScaleTransform(3)
        s = np.random.uniform(minScale, maxScale, 2)
        scale.SetParameters((s[0], s[0], s[1]))
        scale.SetCenter(center)
        augmentedImage.append(resample(itk_img, scale))

    #versor rotation transform * 10
    for i in range(0, 20):
        rotation_center = center
        theta_x = np.pi*np.random.uniform(-1., 1.)  #(-pi, pi)
        theta_y = np.pi*np.random.uniform(-1., 1.)  #(-pi, pi)
        theta_z = np.pi*np.random.uniform(-.5, .5)  #(-pi/2, pi/2)
        t = np.random.uniform(-4., 4., 3)
        translation = (t[0], t[1], t[2])            #(-4, 4)
        transform = sitk.Euler3DTransform(rotation_center, theta_x, theta_y, theta_z, translation)
        augmentedImage.append(resample(itk_img, transform))

    #similarity * 5*3
    for i in range(0, 5):
        rotation_center = center
        axis = (0,0,1)
        angle= np.pi*np.random.uniform(-.5, .5)  #(-pi/2, pi/2)
        t = np.random.uniform(-4., 4., 3)
        translation = (t[0], t[1], t[2])            #(-4, 4)
        scale_factor = np.random.uniform(minScale, maxScale)
        transform = sitk.Similarity3DTransform(scale_factor, axis, angle, translation, rotation_center)
        augmentedImage.append(resample(itk_img, transform))

        #axis 2
        axis = (0, 1, 0)
        angle = np.pi * np.random.uniform(-.5, .5)  # (-pi/2, pi/2)
        t = np.random.uniform(-4., 4., 3)
        translation = (t[0], t[1], t[2])  # (-4, 4)
        scale_factor = np.random.uniform(minScale, maxScale)
        transform = sitk.Similarity3DTransform(scale_factor, axis, angle, translation, rotation_center)
        augmentedImage.append(resample(itk_img, transform))
        #axis 3
        axis = (1, 0, 0)
        angle = np.pi * np.random.uniform(-.5, .5)  # (-pi/2, pi/2)
        t = np.random.uniform(-4., 4., 3)
        translation = (t[0], t[1], t[2])  # (-4, 4)
        scale_factor = np.random.uniform(minScale, maxScale)
        transform = sitk.Similarity3DTransform(scale_factor, axis, angle, translation, rotation_center)
        augmentedImage.append(resample(itk_img, transform))
    #scale versor * 5*3
    for i in range(0, 5):
        #check the order of translation, scale and rotation,
        #should the center be the one after or before translation???
        axis = (0, 0, 1)
        s = np.random.uniform(minScale, maxScale, 2)
        scales = (s[0], s[0], s[1])
        t = np.random.uniform(-4., 4., 3)
        translation = (t[0], t[1], t[2])  # (-4, 4)
        angle = np.pi * np.random.uniform(-.5, .5)  # (-pi/2, pi/2)
        transform = sitk.ScaleVersor3DTransform(scales, axis, angle, translation)
        transform.SetCenter(center)
        augmentedImage.append(resample(itk_img, transform))
        #axis 2
        axis = (0, 1, 0)
        s = np.random.uniform(minScale, maxScale, 2)
        scales = (s[0], s[0], s[1])
        t = np.random.uniform(-4., 4., 3)
        translation = (t[0], t[1], t[2])  # (-4, 4)
        angle = np.pi * np.random.uniform(-.5, .5)  # (-pi/2, pi/2)
        transform = sitk.ScaleVersor3DTransform(scales, axis, angle, translation)
        transform.SetCenter(center)
        augmentedImage.append(resample(itk_img, transform))
        #axis 3
        axis = (1, 0, 0)
        s = np.random.uniform(minScale, maxScale, 2)
        scales = (s[0], s[0], s[1])
        t = np.random.uniform(-4., 4., 3)
        translation = (t[0], t[1], t[2])  # (-4, 4)
        angle = np.pi * np.random.uniform(-.5, .5)  # (-pi/2, pi/2)
        transform = sitk.ScaleVersor3DTransform(scales, axis, angle, translation)
        transform.SetCenter(center)
        augmentedImage.append(resample(itk_img, transform))
    # #scale skew versor, 5*4
    #  scale skew arteface is too obvious, not good for traning
    # for i in range(0, 5):
    #     versor = (0, 0, 0, 1.0)
    #     s = np.random.uniform(.8, 1.2, 2)
    #     scale = (s[0], s[0], s[1])
    #     skew = np.random.uniform(0, 1, 6)   #np.linspace(start=0.0, stop=1.0, num=6)  # six eqaully spaced values in[0,1], an arbitrary choice
    #     t = np.random.uniform(-4., 4., 3)
    #     translation = (t[0], t[1], t[2])  # (-4, 4)
    #     transform = sitk.ScaleSkewVersor3DTransform(scale, skew, versor, translation)
    #     transform.SetCenter(center)
    #     augmentedImage.append(resample(itk_img, transform))
    #
    #     versor = (1., 0, 0, 0)
    #     s = np.random.uniform(.8, 1.2, 2)
    #     scale = (s[0], s[0], s[1])
    #     skew = np.random.uniform(0, 1,6)
    #     t = np.random.uniform(-4., 4., 3)
    #     translation = (t[0], t[1], t[2])  # (-4, 4)
    #     transform = sitk.ScaleSkewVersor3DTransform(scale, skew, versor, translation)
    #     transform.SetCenter(center)
    #     augmentedImage.append(resample(itk_img, transform))
    #
    #     versor = (0, 1.0, 0, 0)
    #     s = np.random.uniform(.8, 1.2, 2)
    #     scale = (s[0], s[0], s[1])
    #     skew = np.random.uniform(0, 1, 6)
    #     t = np.random.uniform(-4., 4., 3)
    #     translation = (t[0], t[1], t[2])  # (-4, 4)
    #     transform = sitk.ScaleSkewVersor3DTransform(scale, skew, versor, translation)
    #     transform.SetCenter(center)
    #     augmentedImage.append(resample(itk_img, transform))
    #
    #     versor = (0, 0, 1.0, 0)
    #     s = np.random.uniform(.8, 1.2, 2)
    #     scale = (s[0], s[0], s[1])
    #     skew = np.random.uniform(0, 1, 6)
    #     t = np.random.uniform(-4., 4., 3)
    #     translation = (t[0], t[1], t[2])  # (-4, 4)
    #     transform = sitk.ScaleSkewVersor3DTransform(scale, skew, versor, translation)
    #     transform.SetCenter(center)
    #     augmentedImage.append(resample(itk_img, transform))

    return augmentedImage
def produceRandomlyDeformedImage(image, label, numcontrolpoints, stdDef):
    sitkImage=sitk.GetImageFromArray(image, isVector=False)
    sitklabel=sitk.GetImageFromArray(label, isVector=False)

    transfromDomainMeshSize=[numcontrolpoints]*sitkImage.GetDimension()

    tx = sitk.BSplineTransformInitializer(sitkImage,transfromDomainMeshSize)


    params = tx.GetParameters()

    paramsNp=np.asarray(params,dtype=float)
    paramsNp = paramsNp + np.random.randn(paramsNp.shape[0])*stdDef

    paramsNp[0:int(len(params)/3)]=0 #remove z deformations! The resolution in z is too bad

    params=tuple(paramsNp)
    tx.SetParameters(params)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitkImage)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)

    resampler.SetDefaultPixelValue(0)
    outimgsitk = resampler.Execute(sitkImage)
    outlabsitk = resampler.Execute(sitklabel)

    outimg = sitk.GetArrayFromImage(outimgsitk)
    outimg = outimg.astype(dtype=np.float32)

    outlbl = sitk.GetArrayFromImage(outlabsitk)
    outlbl = (outlbl>0.5).astype(dtype=np.float32)

    return outimg,outlbl


def produceRandomlyTranslatedImage(image, label, target):
    shape = image.shape
    sitkImage = sitk.GetImageFromArray(image, isVector=False)
    sitklabel = sitk.GetImageFromArray(label.astype(float), isVector=False)

    margin = 5
    x, y, z, d = target
    R = d / 2.0

    rotation_center = [z, y, x]

    scaleX = np.random.uniform(0.9, 1.1)
    scale = sitk.ScaleTransform(3, (1, scaleX, scaleX))
    scale.SetCenter(rotation_center)

    theta_x = np.random.uniform(0, np.pi / 2)
    theta_y = 0.0  # np.random.uniform(0, np.pi)
    theta_z = 0.0  # np.pi/2   #np.random.uniform(0, np.pi/5)
    rigid_euler = sitk.Euler3DTransform(rotation_center, theta_x, theta_y, theta_z, (0, 0, 0))

    # Using the composite transformation we just add them in (stack based, first in - last applied).
    # s eq: rotate, scale and translate

    # first check transformed point, then affine translation
    transformed_target = scale.GetInverse().TransformPoint(rigid_euler.GetInverse().TransformPoint([z, y, x]))
    Z = transformed_target[0]
    Y = transformed_target[1]
    X = transformed_target[2]

    zmin = -(shape[2] - (Z + R + margin)) / 2
    zmax = (Z - R - margin) / 2
    ymin = -(shape[1] - (Y + R + margin)) / 2
    ymax = (Y - R - margin) / 2
    xmin = -(shape[0] - (X + R + margin)) / 2
    xmax = (X - R - margin) / 2
    randTrans = (np.random.randint(zmin, zmax), np.random.randint(ymin, ymax), np.random.randint(xmin, xmax))

    translation = sitk.TranslationTransform(3, randTrans)

    composite_transform = sitk.Transform(rigid_euler)
    composite_transform.AddTransform(scale)
    composite_transform.AddTransform(translation)

    transformed_target = composite_transform.GetInverse().TransformPoint([z, y, x])
    # revserse to numpy order
    transformed_target = transformed_target[::-1]

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitkImage)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(composite_transform)

    outimgsitk = resampler.Execute(sitkImage)
    outlabsitk = resampler.Execute(sitklabel)

    outimg = sitk.GetArrayFromImage(outimgsitk)
    outimg = outimg.astype(dtype=float)

    outlbl = sitk.GetArrayFromImage(outlabsitk) > 0.25
    outlbl = outlbl.astype(dtype=float)
    transformed_target = transformed_target + (d,)

    if transformed_target[0]>shape[0] or transformed_target[0]>shape[0] or transformed_target[0]>shape[0]:
        print transformed_target, ' out of bounds of ', shape, ' , original: ', target
        return image, label, target

    return outimg, outlbl, np.array(transformed_target)

def deformImage(image, label, target):
    image, label, target = produceRandomlyTranslatedImage(image, label, target)
    return image, label, target

if __name__ == '__main__':
    test_name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.286422846896797433168187085942'
    
    # Set up Figure
    fig, ax = plt.subplots(1,1)
    
    #test_read_mhd(test_name+'.mhd', ax)
    #test_lung_seg(test_name+'.mhd', ax)
    #test_rle(test_name+'.mhd', test_name+'.rle', ax)
    test_annotation(test_name, test_name+'.mhd', 'annotations.csv', ax)

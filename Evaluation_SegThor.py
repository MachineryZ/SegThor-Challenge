import numpy as np
import SimpleITK as sitk
import os
import torch.nn as nn
import torch

def Hausdorff_Distance(result, target, target_class, total_classes = 5):
    
    res = result
    tar = target




    # For Binary Classifier
    if total_classes == 2:

        result = sitk.GetImageFromArray(result, isVector = False)
        target = sitk.GetImageFromArray(target, isVector = False)

        dice = sitk.LabelOverlapMeasuresImageFilter()
        dice.Execute(result > 0.5, target == target_class)
        Dice = dice.GetDiceCoefficient()
        
        print("The Dice for Heart is : {}".format(Dice))
        
        if (res > 0.5).sum() == 0 or (tar == target_class).sum() == 0:
            print("Sorry The No.{} class has not been classified...".format(target_class))
            return
            
        hausdorff = sitk.HausdorffDistanceImageFilter()
        hausdorff.Execute(result > 0.5, target == target_class)
        distance = hausdorff.GetHausdorffDistance()
        print("The Hausdorff Distance for No.{} class is : {}".format(target_class, distance))

        return
    if total_classes == 4:

        result = sitk.GetImageFromArray(result, isVector = False)
        target = sitk.GetImageFromArray(target, isVector = False)

        if (target == 1).sum() == 0 or (target == 3).sum() == 0 or (target == 4):
            break
        dice = sitk.LabelOverlapMeasuresImageFilter()
        dice.Execute(result == 1, target == 1)
        Dice = dice.GetDiceCoefficient()
        print("The Dice for Esophagus is : {}".format(Dice))
        hausdorff = sitk.HausdorffDistanceImageFilter()
        hausdorff.Execute(result == 1, target == 1)
        distance = hausdorff.GetHausdorffDistance()
        print("The Hausdorff Distance for No.{} class is : {}".format(1, distance))

        dice = sitk.LabelOverlapMeasuresImageFilter()
        dice.Execute(result == 2, target == 3)
        Dice = dice.GetDiceCoefficient()
        print("The Dice for Airway is : {}".format(Dice))

        hausdorff = sitk.HausdorffDistanceImageFilter()
        hausdorff.Execute(result == 2, target == 3)
        distance = hausdorff.GetHausdorffDistance()
        print("The Hausdorff Distance for No.{} class is : {}".format(3, distance))

        dice = sitk.LabelOverlapMeasuresImageFilter()
        dice.Execute(result == 3, target == 4)
        Dice = dice.GetDiceCoefficient()
        print("The Dice for Aorta is : {}".format(Dice))
        hausdorff = sitk.HausdorffDistanceImageFilter()
        hausdorff.Execute(result == 3, target == 4)
        distance = hausdorff.GetHausdorffDistance()
        print("The Hausdorff Distance for No.{} class is : {}".format(4, distance))



    # For Multiple Classifier
    if total_classes == 5:

        for i in range(1, total_classes):
            if (res == i).sum() == 0 or (tar == i).sum() == 0:
                print("The No.{} class has not been classified...".format(i))
                continue
            hausdorff = sitk.HausdorffDistanceImageFilter()
            hausdorff.Execute(result == i, target == i)
            distance = hausdorff.GetHausdorffDistance()
            print("The Hausdorff Distance for No.{} class is : {}".format(i, distance))

        for i in range(1, total_classes):
            if (res == i).sum() == 0 or (tar == i).sum() == 0:
                print("The No.{} class has not been classified...".format(i))
                continue
            dice = sitk.LabelOverlapMeasuresImageFilter()
            dice.Execute(result == i, target == i)
            num = dice.GetDiceCoefficient()
            print("The Dice for No.{} class is : {}".format(i, num))

        return;







import os
import re
import glob
import argparse
import pydicom
import numpy as np
import logging
from tqdm import tqdm
import SimpleITK as sitk
import cv2
from dicompylercore import dicomparser


def load_dicom_series(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def drawslince(image, msk, contor):
    ctor = []
    for point in contor:
        # print(point)
        # point = ["3.7","96.0","-554"]
        # print(f"CT size: {image.GetSize()}")
        # print(f"CT spacing: {image.GetSpacing()}")
        temppoint = image.TransformPhysicalPointToIndex(point)
        # print(temppoint)
        ctor.append((temppoint[0], temppoint[1]))
    ctor = np.array(ctor).reshape(-1, 1, 2)
    msk = cv2.drawContours(msk, [ctor], -1, 1, thickness=cv2.FILLED)
    return msk, temppoint[2]

def mask2nii(image, mskpath):
    rtss = dicomparser.DicomParser(mskpath)
    rois = rtss.GetStructures()
    roikey = [roi_id for roi_id in rois.keys() if rois[roi_id]['name'].upper()=="PTV"]
    roinames = [rois[roi_id]['name'].upper() for roi_id in roikey]
    print(roinames)
    if len(roikey) == 0:
        return None
    size = image.GetSize()
    if len(size) == 4:
        # Convert SimpleITK image to NumPy array
        image_array = sitk.GetArrayFromImage(image)
        # Remove the last dimension if it is of size 1
        # image_array = np.squeeze(image_array, axis=-1)

        image_array = image_array.reshape(size[:-1])
        # Convert back to SimpleITK image
        new_image = sitk.GetImageFromArray(image_array)
        new_image.SetSpacing(image.GetSpacing()[:-1])  # Update spacing for 3D
        print(f"New image spacing: {new_image.GetSpacing()}")
        image = new_image
        size = image.GetSize()
    roi = rtss.GetStructureCoordinates(roikey[0])
    mask = np.zeros(size, dtype=np.uint8)
    for slicer_i in roi.keys():
        msk = np.zeros((size[0], size[1]))
        maskcount = len(roi[slicer_i])
        for i in range(maskcount):
            tempcontor = roi[slicer_i][i]['data']
            print(f"CT size: {size}")
            print(f"msk size: {msk.shape}")
            msk, slicenum = drawslince(image, msk, tempcontor)
        mask[:, :, slicenum] = msk
    mask = mask.transpose((2, 0, 1))
    mask = sitk.GetImageFromArray(mask)
    mask.SetOrigin(image.GetOrigin())
    mask.SetDirection(image.GetDirection())
    mask.SetSpacing(image.GetSpacing())
    print(mask)
    return mask


# A
# folder_path = "/home/zhaosheng/Documents/LiuHuan/files/liuhuan_lung_plan_6000/liuhuan_lung_plan_6000/2022_04_6000/two_ptv/RT220776/"
# struce_path = "/home/zhaosheng/Documents/LiuHuan/files/liuhuan_lung_plan_6000/liuhuan_lung_plan_6000/2022_04_6000/two_ptv/RT220776/1788461_StrctrSets.dcm"

# # B
folder_path = "/home/zhaosheng/Documents/LiuHuan/files/liuhuan_lung_plan_6000/liuhuan_lung_plan_6000/2023_03_6000/one_ptv/RT230480"
struce_path = "/home/zhaosheng/Documents/LiuHuan/files/liuhuan_lung_plan_6000/liuhuan_lung_plan_6000/2023_03_6000/one_ptv/RT230480/RS.RT230480.STRCTRLABEL.dcm"

ct_image = load_dicom_series(folder_path)
mask2nii(ct_image,struce_path)
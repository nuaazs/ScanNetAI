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

A = "/home/zhaosheng/Documents/LiuHuan/files/liuhuan_lung_plan_6000/liuhuan_lung_plan_6000/2023_03_6000/two_ptv/RT230393/RD.RT230393.LungFinal.dcm"
dose_image = sitk.ReadImage(A)
print(dose_image)
print(dose_image.GetSize())
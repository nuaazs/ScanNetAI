# coding = utf-8
# @Time    : 2023-12-26  12:46:53
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Prepare dataset (niis).

import os
import re
import glob
import argparse
import pydicom
import numpy as np
import logging
from tqdm import tqdm
import SimpleITK as sitk

# Configure logging to output to both file and console
log_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
log_file = 'data_prepare.log'

file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


# read dicom series
def load_dicom_series(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

# resample image
def resample_image(image, new_spacing=[1.0, 1.0, 1.0]):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    
    return resample.Execute(image)

# save nifti image
def save_nifti(image, output_filename):
    sitk.WriteImage(image, output_filename, True)  # True for .nii.gz


def find_rt_folders(root_path):
    """
    Find all folders under root_path following the pattern RT+6 digits.
    """
    rt_pattern = re.compile(r'RT\d{6}')
    for dirpath, dirnames, files in os.walk(root_path):
        for dirname in dirnames:
            if rt_pattern.match(dirname):
                yield os.path.join(dirpath, dirname)

def safe_crop_image_to_match(reference_image, target_image):
    """
    Safely crop the target_image to match the size and position of the reference_image.
    Adjusts the ROI to ensure it is within the bounds of the target image.
    """
    reference_origin = reference_image.GetOrigin()
    target_origin = target_image.GetOrigin()
    reference_size = reference_image.GetSize()
    target_size = target_image.GetSize()

    # Calculate the difference between the origins in index space
    origin_diff_index_space = target_image.TransformPhysicalPointToIndex(reference_origin)

    # Calculate the end index of the ROI in the target image's index space
    end_index = [start + size for start, size in zip(origin_diff_index_space, reference_size)]

    # Adjust the size of the ROI to ensure it's within the target image bounds
    adjusted_size = [min(end, targ_sz) - start for end, start, targ_sz in zip(end_index, origin_diff_index_space, target_size)]
    
    # Create the ROI filter
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetSize(adjusted_size)
    roi_filter.SetIndex(origin_diff_index_space)
    
    # Apply the ROI filter
    cropped_target_image = roi_filter.Execute(target_image)
    
    return cropped_target_image


def process_ct_dose(folder, output_path, resample_spacing):
    """
    Process CT and Dose DICOM files in the given folder, resample them, 
    and save as NII.GZ files in the corresponding output folder.
    """

    # Find CT and Dose files
    print(os.path.join(folder, '*_[cC][tT]*.[dD][cC][mM]'))
    ct_files = sorted(glob.glob(os.path.join(folder, '*_[cC][tT]*.[dD][cC][mM]')))
    dose_files = sorted(glob.glob(os.path.join(folder, '*_[dD][oO][sS][eE]*.[dD][cC][mM]')))
    
    if (len(ct_files) <= 10) or (len(dose_files) <= 0):
        logging.warning(f'CT files: {len(ct_files)}, Dose files: {len(dose_files)}')
        return None,None
    # Process CT files
    ct_image = load_dicom_series(folder)
    
    # Read dose
    dose_dcm_path = dose_files[0]
    dose_image = sitk.ReadImage(dose_dcm_path)

    ct_image = resample_image(ct_image, resample_spacing)
    dose_image = resample_image(dose_image, resample_spacing)
    dose_size = dose_image.GetSize()
    dose_origin = dose_image.GetOrigin()


    # 计算两个图像的重叠区域
    ct_origin = ct_image.GetOrigin()
    dose_origin = dose_image.GetOrigin()

    ct_size = ct_image.GetSize()
    dose_size = dose_image.GetSize()

    ct_spacing = ct_image.GetSpacing()
    dose_spacing = dose_image.GetSpacing()

    # 计算重叠区域的物理坐标
    overlap_start = [max(ct_origin[i], dose_origin[i]) for i in range(3)]
    overlap_end = [min(ct_origin[i] + ct_spacing[i]*ct_size[i], dose_origin[i] + dose_spacing[i]*dose_size[i]) for i in range(3)]

    # 计算重叠区域在 CT 图像中的索引
    ct_index_start = [int(round((overlap_start[i] - ct_origin[i]) / ct_spacing[i])) for i in range(3)]
    ct_index_end = [int(round((overlap_end[i] - ct_origin[i]) / ct_spacing[i])) for i in range(3)]

    # 裁剪 CT 图像
    cropped_ct_size = [ct_index_end[i] - ct_index_start[i] for i in range(3)]
    cropped_ct = sitk.RegionOfInterest(ct_image, cropped_ct_size, ct_index_start)

    # 裁剪剂量图像
    dose_index_start = [int(round((overlap_start[i] - dose_origin[i]) / dose_spacing[i])) for i in range(3)]
    dose_index_end = [int(round((overlap_end[i] - dose_origin[i]) / dose_spacing[i])) for i in range(3)]
    cropped_dose_size = [dose_index_end[i] - dose_index_start[i] for i in range(3)]
    cropped_dose = sitk.RegionOfInterest(dose_image, cropped_dose_size, dose_index_start)

    logging.info(f'CT image shape: {cropped_ct.GetSize()}')
    logging.info(f'Dose image shape: {cropped_dose.GetSize()}')
    logging.info(f'CT image spacing: {cropped_ct.GetSpacing()}')
    logging.info(f'Dose image spacing: {cropped_dose.GetSpacing()}')

    if cropped_ct.GetSize() != cropped_dose.GetSize():
        logging.warning('CT and Dose image sizes do not match!')
        return None,None
    if cropped_ct.GetSpacing() != cropped_dose.GetSpacing():
        logging.warning('CT and Dose image spacings do not match!')
        return None,None
    dose_output_filename = os.path.join(output_path, 'dose.nii.gz')
    save_nifti(cropped_dose, dose_output_filename)
    ct_output_filename = os.path.join(output_path, 'ct.nii.gz')
    save_nifti(cropped_ct, ct_output_filename)
    return cropped_ct, cropped_dose

def main(root_path, output_path, resample_spacing):
    """
    Main function to process all RT folders under the root path.
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    rt_folders = list(find_rt_folders(root_path))
    logging.info(f'Found {len(rt_folders)} RT folders to process.')

    success_count = 0
    failure_count = 0

    # Process each RT folder with a progress bar
    with tqdm(total=len(rt_folders), desc='Processing RT folders') as progress_bar:
        for folder in rt_folders:
            rt_number = os.path.basename(folder)
            rt_output_path = os.path.join(output_path, rt_number)
            os.makedirs(rt_output_path, exist_ok=True)
            logging.info(f'Processing RT folder {folder}')
            cropped_ct, cropped_dose = process_ct_dose(folder, rt_output_path, resample_spacing)
            if cropped_ct is not None and cropped_dose is not None:
                success_count += 1
            else:
                failure_count += 1

            progress_bar.update(1)
            progress_bar.set_postfix_str(f'Success: {success_count}, Failure: {failure_count}, Success Rate: {success_count/(success_count+failure_count):.2%}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DICOM to NII.GZ Conversion Script')
    parser.add_argument('--root_path', type=str, default="/home/zhaosheng/Documents/LiuHuan", help='Root path containing RT folders')
    parser.add_argument('--output_path', type=str, default="./niis", help='Output path for NII.GZ files')
    parser.add_argument('--resample_spacing', type=float, nargs=3, metavar=('X', 'Y', 'Z'), default=[3.0, 3.0, 3.0], help='Resampling spacing in X, Y, and Z dimensions')
    args = parser.parse_args()
    
    main(args.root_path, args.output_path, args.resample_spacing)
    # LOG: 557/557 [05:05<00:00,  1.82it/s, Success: 402, Failure: 155, Success Rate: 72.17%]
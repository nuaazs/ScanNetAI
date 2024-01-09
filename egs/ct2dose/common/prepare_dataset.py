# coding = utf-8
# @Time    : 2023-12-26  12:46:53
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Prepare dataset (niis).

import cv2
import os
import re
import glob
import logging
import warnings
import argparse
import pydicom
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from dicompylercore import dicomparser

# Turnoff WARNING
warnings.filterwarnings("ignore")
sitk.ProcessObject_SetGlobalWarningDisplay(False)

# Configure logging to output to both file and console
log_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
file_handler = logging.FileHandler('data_prepare.log')
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

def squeeze_image(image):
    """Squeeze the image to remove dimensions of size 1."""
    size = image.GetSize()
    spacing = image.GetSpacing()
    if len(size) == 3:
        return image
    new_size, new_spacing = [s for s in size if s != 1], [sp for s, sp in zip(size, spacing) if s != 1]
    image_array = sitk.GetArrayFromImage(image).reshape(new_size)
    new_image = sitk.GetImageFromArray(image_array)
    new_image.SetSpacing(new_spacing)
    # set origin
    old_origin = image.GetOrigin()
    new_origin = [old_origin[i] for i in range(3) if size[i] != 1]
    new_image.SetOrigin(new_origin)
    # set direction
    old_direction = image.GetDirection()
    new_direction = [old_direction[i] for i in range(9) if size[i] != 1]
    new_image.SetDirection(new_direction)

    return new_image

def load_dicom_series(directory,ct_re="*[cC][tT]*.[dD][cC][mM]"):
    """Load DICOM series from the given directory."""
    dicom_files = [f for f in os.listdir(directory) if f.endswith('.dcm') or f.endswith('.DCM')]
    dicom_files = sorted([f for f in dicom_files if not f.startswith('R') and ("_CT" in f.upper() or f.startswith("CT"))])
    dicom_files = [os.path.join(directory, f) for f in dicom_files]
    if dicom_files:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_files)
        return reader.Execute()
    else:
        return None

def resample_image(image, new_spacing=[1.0, 1.0, 1.0]):
    """Resample image to new spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    return resample.Execute(image)

def save_nifti(image, output_filename):
    """Save the image as a NIFTI file."""
    sitk.WriteImage(image, output_filename, True)  # True for .nii.gz

def find_rt_folders(root_path):
    """Find all RT folders under the given root path."""
    rt_pattern = re.compile(r'RT\d{6}')
    for dirpath, dirnames, _ in os.walk(root_path):
        for dirname in dirnames:
            if rt_pattern.match(dirname):
                yield os.path.join(dirpath, dirname)

def drawslince(image, msk, contor):
    """Draw a slice of contour on the mask."""
    # logging.info(f"# drawslince -> CT IMAGE: {image.GetSize()}")
    # logging.info(f"# drawslince -> CT IMAGE Sp: {image.GetSpacing()}")
    ctor = []
    for point in contor:
        temppoint = image.TransformPhysicalPointToIndex(point)
        ctor.append((temppoint[0], temppoint[1]))
    ctor = np.array(ctor).reshape(-1, 1, 2)
    return cv2.drawContours(msk, [ctor], -1, 1, thickness=cv2.FILLED), temppoint[2]

def glob_files(root_path,re_string_list):
    """find files by re list"""
    for re_string in re_string_list:
        r = sorted(glob.glob(os.path.join(root_path, re_string)))
        if len(r)>0:
            return r
    return []

def process_ct_dose(folder, output_path, resample_spacing):
    """Process CT and Dose DICOM files for conversion to NII.GZ."""
    ct_files = glob_files(folder, ['*_[cC][tT]*.[dD][cC][mM]','*[cC][tT]*.[dD][cC][mM]'])
    dose_files = glob_files(folder, ['*_[dD][oO][sS][eE]*.[dD][cC][mM]','RD*.[dD][cC][mM]'])
    struct_files = glob_files(folder, ['*_StrctrSets*.[dD][cC][mM]','RS*.[dD][cC][mM]'])

    
    if not ct_files or not dose_files or not struct_files:
        logging.warning(f'Missing files in {folder}')
        logging.warning(f'*'*30)
        logging.warning(f'=ERROR=ERROR='*3)
        logging.warning(f"Dose file: {dose_files}")
        logging.warning(f"CT file: {ct_files[0]} ... ")
        logging.warning(f"RT file: {struct_files}")
        logging.warning(f'*'*30)
        return None, None, None

    logging.warning(f'*'*30)
    logging.warning(f"Dose file: {dose_files}")
    logging.warning(f"CT file: {ct_files[0]} ... ")
    logging.warning(f"RT file: {struct_files}")
    logging.warning(f'*'*30)

    dose_image = resample_image(squeeze_image(sitk.ReadImage(dose_files[0])), resample_spacing)

    # Processing structure file
    rtss = dicomparser.DicomParser(struct_files[0])
    rois = rtss.GetStructures()
    roikey = [roi_id for roi_id in rois if "PTV" in rois[roi_id]['name'].upper()]
    if not roikey:
        logging.warning(f"No ROI found in {folder}")
        return None, None, None

    ct_image = squeeze_image(load_dicom_series(folder))
    ct_size = ct_image.GetSize()

    mask = np.zeros(ct_size, dtype=np.uint8)
    roi = rtss.GetStructureCoordinates(roikey[0])
    for slicer_i in roi:
        msk = np.zeros((ct_size[0], ct_size[1]))
        for contor in roi[slicer_i]:
            msk, slicenum = drawslince(ct_image, msk, contor['data'])
            mask[:, :, slicenum] = msk
    mask = mask.transpose((2, 0, 1))
    mask = sitk.GetImageFromArray(mask)
    mask.SetOrigin(ct_image.GetOrigin())
    mask.SetDirection(ct_image.GetDirection())
    mask.SetSpacing(ct_image.GetSpacing())
    mask_image = squeeze_image(resample_image(mask, resample_spacing))
    mask_size = mask_image.GetSize()

    ct_image = resample_image(ct_image, resample_spacing)
    dose_origin = dose_image.GetOrigin()
    new_origin = [ct_image.GetOrigin()[0], ct_image.GetOrigin()[1], dose_origin[2]]
    ct_image.SetOrigin(new_origin)
    mask_image.SetOrigin(new_origin)


    ct_size = ct_image.GetSize()

    dose_size = dose_image.GetSize()
    

    ct_origin = ct_image.GetOrigin()
    dose_origin = dose_image.GetOrigin()
    ct_size = ct_image.GetSize()
    dose_size = dose_image.GetSize()
    ct_spacing = ct_image.GetSpacing()
    dose_spacing = dose_image.GetSpacing()

    
    print(f"CT: {ct_size}\nDose: {dose_size}\nMASK: {mask_size}")
    print(f"CT: {ct_spacing}\nDose: {dose_spacing}\nMASK: {mask_image.GetSpacing()}")
    print(f"CT: {ct_origin}\nDose: {dose_origin}\nMASK: {mask_image.GetOrigin()}")
    

    overlap_start = [max(ct_origin[i], dose_origin[i]) for i in range(3)]
    overlap_end = [min(ct_origin[i] + ct_spacing[i]*ct_size[i], dose_origin[i] + dose_spacing[i]*dose_size[i]) for i in range(3)]
    ct_index_start = [int(round((overlap_start[i] - ct_origin[i]) / ct_spacing[i])) for i in range(3)]
    ct_index_end = [int(round((overlap_end[i] - ct_origin[i]) / ct_spacing[i])) for i in range(3)]

    print(ct_index_start)
    print(ct_index_end)

    cropped_ct_size = [ct_index_end[i] - ct_index_start[i] for i in range(3)]

    cropped_ct = sitk.RegionOfInterest(ct_image, cropped_ct_size, ct_index_start)
    cropped_mask_image = sitk.RegionOfInterest(mask_image, cropped_ct_size, ct_index_start)

    dose_index_start = [int(round((overlap_start[i] - dose_origin[i]) / dose_spacing[i])) for i in range(3)]
    dose_index_end = [int(round((overlap_end[i] - dose_origin[i]) / dose_spacing[i])) for i in range(3)]
    cropped_dose_size = [dose_index_end[i] - dose_index_start[i] for i in range(3)]
    cropped_dose = sitk.RegionOfInterest(dose_image, cropped_dose_size, dose_index_start)
    
    logging.info(f"CT IMAGE: {cropped_ct.GetSize()}")
    logging.info(f"CT IMAGE Sp: {cropped_ct.GetSpacing()}")
    logging.info(f"DOSE IMAGE: {cropped_dose.GetSize()}")
    logging.info(f"DOSE IMAGE Sp: {cropped_dose.GetSpacing()}")
    logging.info(f"MASK IMAGE: {cropped_mask_image.GetSize()}")
    logging.info(f"MASK IMAGE Sp: {cropped_mask_image.GetSpacing()}")
    
    save_nifti(cropped_ct, os.path.join(output_path, 'ct.nii.gz'))
    save_nifti(cropped_dose, os.path.join(output_path, 'dose.nii.gz'))
    save_nifti(cropped_mask_image, os.path.join(output_path, 'struct.nii.gz'))

    return ct_image, dose_image, mask_image

def main(root_path, output_path, resample_spacing):
    """Main function to process and convert DICOM to NII.GZ."""
    os.makedirs(output_path, exist_ok=True)
    rt_folders = list(find_rt_folders(root_path))
    logging.info(f'Found {len(rt_folders)} RT folders to process.')
    success_count,failure_count = 0,0
    with tqdm(total=len(rt_folders), desc='Processing RT folders') as progress_bar:
        for folder in rt_folders:
            # if "RT221803" not in folder:
            #     continue
            tid = os.path.basename(folder)
            if os.path.exists(os.path.join(output_path, tid, 'ct.nii.gz')):
                logging.info(f'Skipping {folder}')
                continue
            print(f"Processing {folder}")
            rt_number = os.path.basename(folder)
            rt_output_path = os.path.join(output_path, rt_number)
            os.makedirs(rt_output_path, exist_ok=True)
            try:
                ct_image, dose_image, mask_image = process_ct_dose(folder, rt_output_path, resample_spacing)
            except Exception as e:
                ct_image, dose_image, mask_image = None,None,None
            if ct_image is not None:
                success_count += 1
            else:
                failure_count += 1
            progress_bar.update(1)
            progress_bar.set_postfix_str(f'Success: {success_count}, Failure: {failure_count}, Success Rate: {success_count/(success_count+failure_count):.2%}')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DICOM to NII.GZ Conversion Script')
    parser.add_argument('--root_path', type=str, default="/home/zhaosheng/Documents/LiuHuan/files/liuhuan_lung_plan_6000", help='Root path containing RT folders')
    parser.add_argument('--output_path', type=str, default="./niis_output", help='Output path for NII.GZ files')
    parser.add_argument('--resample_spacing', type=float, nargs=3, metavar=('X', 'Y', 'Z'), default=[3.0, 3.0, 3.0], help='Resampling spacing in X, Y, and Z dimensions')
    args = parser.parse_args()
    main(args.root_path, args.output_path, args.resample_spacing)
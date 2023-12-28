# DICOM to NII.GZ Conversion Script

## Overview

This Python script is designed to process and convert DICOM files from radiotherapy (RT) folders into NIfTI format (NII.GZ). It is particularly useful in medical image processing, where analyses often require the data to be in a uniform format like NIfTI.

## Features

- **DICOM Series Processing**: Converts DICOM series from RT folders into NIfTI format.
- **Image Resampling**: Resamples images to a specified spacing, improving uniformity across different scans.
- **Structure Processing**: Processes structure set files (RTSS) from DICOM and extracts regions of interest (ROI).
- **Mask Generation**: Generates masks based on the ROI from RTSS files.
- **Error Logging**: Records warnings and errors during processing, aiding in troubleshooting.

## Steps

1. **Load DICOM Series**: The script loads all DICOM series from specified directories.
2. **Image Resampling**: Each loaded image is resampled to a given spatial resolution.
3. **ROI Processing**: Extracts ROI from structure set files and generates corresponding masks.
4. **NIfTI Conversion**: Converts the processed DICOM series and masks into NIfTI files.
5. **Error and Progress Logging**: Logs progress and errors, providing detailed information on the process and potential issues.

## Input and Output

- **Input**: The script takes DICOM files located in specified RT folders. These folders should contain CT, Dose, and Structure Set DICOM files.
- **Output**: Outputs are NIfTI files (`.nii.gz`) for the CT scan, Dose distribution, and ROI masks.

## Usage

### Requirements

- Python 3.x
- Libraries: `SimpleITK`, `cv2` (OpenCV), `numpy`, `tqdm`, `dicompylercore`

### Command Line Arguments

- `--root_path`: Path to the root directory containing RT folders.
- `--output_path`: Output directory for NIfTI files.
- `--resample_spacing`: Resampling spacing in X, Y, and Z dimensions (default: `[3.0, 3.0, 3.0]`).

### Example

```bash
python script_name.py --root_path "/path/to/RT/folders" --output_path "./output_niis" --resample_spacing 3.0 3.0 3.0
```

Replace `script_name.py` with the actual name of the script.

## Important Notes

- Ensure that all required libraries are installed before running the script.
- The script assumes a certain structure in the RT folders; make sure your data is organized accordingly.
- The script logs information in `data_prepare.log`, which can be reviewed for troubleshooting.

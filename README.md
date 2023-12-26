

<font size=4> README: EN | <a href="./README.zh.md">中文</a>  </font>

<div align="center">
    <h1>ScanNetAI: Self-Supervised Deep Learning Model for CT Image Analysis</h1>
    <img src="scannetai.png" alt="ScanNetAI Logo" style="width:20%;">
</div>


<div align="center">
    <a href="./README.zh.md"><img src="https://img.shields.io/badge/README-中文版本-red"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
</div>
<br>

### GitHub Project Description

ScanNetAI is an innovative self-supervised deep learning model meticulously designed for processing and analyzing CT images. Developed with cutting-edge machine learning techniques, ScanNetAI excels in handling large-scale CT data, providing unparalleled insights into medical image analysis. Pre-trained on a diverse range of CT imaging data, ScanNetAI offers exceptional performance in critical downstream tasks, including image segmentation, medical image conversion, and dosage prediction. The model's robustness and versatility make it an indispensable tool for researchers and professionals in medical imaging, offering a new frontier in diagnostics and treatment planning.

## Acknowledgements

Special thanks to the Jiangsu Provincial People's Hospital for their invaluable contributions and support in the development of this project. Their expertise and dedication to medical innovation have been instrumental in enhancing the capabilities of ScanNetAI.

## Features

- **State-of-the-Art Self-Supervised Learning**: Utilizes a vast and diverse collection of unlabeled CT images for robust pre-training.
- **Multifaceted Applications**: Expertly handles tasks such as image segmentation, medical image conversion, and precise dosage prediction.
- **High Precision and Accuracy**: Employs advanced algorithms to achieve superior prediction accuracy and detailed image analysis.
- **Seamless Integration**: Optimized for easy adoption into existing medical imaging workflows, enhancing diagnostic accuracy and efficiency.

## Installation

To use ScanNetAI, first clone the repository:

```
git clone https://github.com/your-username/ScanNetAI.git
```

Then install the required dependencies:

```
pip install -r requirements.txt
```

## Quick Start

Begin using ScanNetAI with these simple steps:

```python
from scannet import ScanNetAI

# Load the model
model = ScanNetAI.load_pretrained('path/to/model')

# Process a CT image
embedding = model.process('path/to/ct_image')
```

## Use Cases

- **Image Segmentation**: Enhanced ability to segment CT images, revealing critical details and features.
- **Medical Image Conversion**: Transforms CT images into various formats, facilitating in-depth analysis.
- **Dosage Prediction**: Utilizes AI to predict optimal treatment dosages, personalizing patient care.

## Contributing

We encourage a wide range of contributions, from feature enhancements to code optimization, and documentation improvements. For more details, refer to the [Contribution Guide](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).

## References

- [ViT-V-Net for 3D Image Registration Pytorch](https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch)
- [TransUnet](https://github.com/Beckschen/TransUNet)
- [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
- [VoxelMorph](https://github.com/voxelmorph/voxelmorph)

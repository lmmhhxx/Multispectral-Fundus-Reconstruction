# Multispectral Fundus Reconstruction

This repository provides an implementation of multispectral fundus image reconstruction using a pretrained MST++ (Multi-Stage Spectral-wise Transformer) model. Instead of relying on standard RGB fundus images, the model takes three representative spectral-band images as input and efficiently reconstructs high-spectral-resolution fundus data. This approach is particularly suitable for medical imaging applications where non-invasive spectral acquisition is required.

##  Model Description

This project uses the [MST++](https://github.com/caiyuanhao1998/MST-plus-plus) 

We provide a pretrained model file `model_best.pth` for direct inference.

## Dataset

The dataset used in this project has been uploaded to Google Drive:
 [Download the dataset](https://drive.google.com/drive/folders/1-ugdfKb4m0JIjpfyJ5sPKO33MK6B7aMR?usp=sharing)


The dataset contains `.bmp` images
Accept 3-channel grayscale images as input     7：572nm  4：650nm  1：710nm
Output 7-channel grayscale images           7：572nm   6：600nm  5：617nm  4：650nm  3：662nm  2：685nm  1：710nm
Output 9-channel grayscale images          13：598nm 12：630nm 11：637nm 10：680nm 9：716nm 8：724nm 7：572nm 4：652nm 1：711nm
License
This code is released for academic research only. For commercial use, please contact the author.

To reconstruct 7-band multispectral images from 3-channel grayscale inputs using the pretrained model, follow the steps below:

1. Prepare Input Images
Place your 3-channel grayscale .bmp images in the test_input/ directory.

The images should be shaped , with each channel representing a specific spectral band . These serve as input to the model for spectral recovery.

2. Run Inference
Use the following command to start the reconstruction:
python inference.py \
  --model_path model_best.pth \
  --input_dir test_input \
  --output_dir test_output \
  --select_model model_best \
  --gpu_id 0

--model_path: Path to the pretrained model (e.g., model_best.pth)

--input_dir: Directory containing the input 3-band grayscale .bmp images

--output_dir: Directory to save the reconstructed 7-band .bmp images

--select_model: Selects which model to use 

--gpu_id: Index of GPU to use (e.g., 0; omit this to use CPU)

3. Output Format
The reconstructed images are saved in .bmp format in the test_output/ directory.

Each output contains 7 grayscale channels, corresponding to the predicted multispectral bands.
Reconstructing a 9-band multispectral image from a 3-channel grayscale input using a pre-trained model is the same as reconstructing a 7-band multispectral image from a 3-channel grayscale input, but it should be noted that the model and code in the 3to9 folder should be used.



# B-CvT: A Style Transfer Framework for Content Preservation and Fairness in Skin Lesion Imaging
This repository is the official implementation of B-CvT, a style transfer framework designed to preserve content and promote fairness in skin lesion imaging.

## Requireents
All dependencies are listed in ```environment.yml```. To install via Conda:
```
conda env create -f environment.yml -n <your_env_name>
```

## Pre-trained Models and Datasets

Download and place the pre-trained weights in  ```checkpoints/```:

B-CvT : [Download Link](https://drive.google.com/file/d/155VXRYsIaJjJVefdx_6TvxY-QWD6uUsl/view?usp=drive_link)

VGG : [Download Link](https://drive.google.com/file/d/1E2Qcq8F1a-5yB7PsoMRqKzVBkfAfKiLH/view?usp=drive_link)

Download the datasets and organize them as follows:

MS_COCO : [Download Link](https://cocodataset.org/#download)
Unzip to ```data/content/```.

WikiArt : [Download Link](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)
Unzip to ```data/style/```.

## Training

To start training the style transfer model put the content images and the style images under  ```data/content/``` and ```data/style/``` respectively, and then run:

```
python train.py --content_folder ./data/content --style_folder ./data/style 
```

Please refer to ```train.py``` for other paramaters.

## Test

To stylize single images or entire directories, run:
```
python test.py --content <content_image_or_dir> --style <style_image_or_dir>
```

## Citation

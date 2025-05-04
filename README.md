# B-CvT: A Style Transfer Framework for Content Preservation and Fairness in Skin Lesion Imaging
This is an official implementation of 

## Requireents
You can find all dependencies in environment.yml file. If you use conda, you can create environment and install all libraries by using following command. 

 <code>```conda env create -f environment.yml -n your_env_name```</code>

Please install geomloss library when to train.

<code>```pip install geomloss==0.2.6```</code>

## Pre-trained Models and Datasets

Download the pre-trained model weights from the following links and place them under  checkpoints/:

B-CvT : [Download Link](https://drive.google.com/file/d/155VXRYsIaJjJVefdx_6TvxY-QWD6uUsl/view?usp=drive_link)
VGG : [Download Link](https://drive.google.com/file/d/1E2Qcq8F1a-5yB7PsoMRqKzVBkfAfKiLH/view?usp=drive_link)

Download the dataset from Dataset Website below and unzip MS_COCO and WikiArt under ./data/content and ./data/style respectively.

MS_COCO : [Download Link](https://cocodataset.org/#download)
WikiArt : [Download Link](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)

## Training

## Test


License

Specify the license under which the code is released, e.g.: MIT License.

Citation

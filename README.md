# B-CvT: A Style Transfer Framework for Content Preservation and Fairness in Skin Lesion Imaging
This is an official implementation of 

## Requireents
You can find all dependencies in environment.yml file. If you use conda, you can create environment and install all libraries by using following command. 

conda env create -f environment.yml -n your_env_name

Please install geomloss library when to train.

pip install geomloss==0.2.6

Pre-trained Models and Weights

Download the pre-trained model weights from the following links and place them under checkpoints/:

Model A: Download Link

Model B: Download Link

Dataset

Link to the dataset and instructions on how to prepare it:

Download the dataset from Dataset Website.

Unzip and arrange the files as follows:

data/
├── train/
│   ├── images/
│   └── annotations/
└── val/
    ├── images/
    └── annotations/

Training

Provide commands and key hyperparameters to train the model:

python train.py \
    --config config/train.yaml \
    --data_root data/ \
    --output_dir outputs/ \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --epochs 100

Explain any optional flags or configurations.

Evaluation

Describe how to run evaluation or inference:

python evaluate.py \
    --checkpoint checkpoints/model_best.pth \
    --data_root data/val/ \
    --output_dir results/

Include metrics calculation scripts if applicable.

License

Specify the license under which the code is released, e.g.: MIT License.

Citation

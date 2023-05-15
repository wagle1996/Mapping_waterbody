# Watermap

The goal of our project is to identify the water bodies of water using semantic segmentation of satellite images. 

## Build instrcutions

Create a virtual environment and install the packages listed in requirements.txt file. If using GPUs you will need to install `cudnn/8.2.0,cuda/11.1.1,gcc/10.2`


## How It Works

1. `preprocessing.py` - converts the satellite images to a format that can be used by the model.
2. `hyperparameters.py` - the ideal hyperparameters for the model.
3. `main.py` - contains the actual model, training and testing implementation, and visualisation.

## Usage
Our entire dataset `TestImages, TestLabels, TrainingImages, TrainingLabels` and best `checkpoint.hdf5` available via this [Google Drive Link](https://drive.google.com/drive/folders/1yHc78Q7Y65jW_IRwhNYgOzK8M9Ub143X?usp=sharing).


```js
The `main.py` accepts the following commandline arguments:
optional arguments:
  -h, --help            show this help message and exit
  --skip_train          If true, skips training. (default: False)
  --augment_data        If true, uses data augmentation during training.
                        (default: False)
  --load_checkpoint LOAD_CHECKPOINT
                        Path to model checkpoint (.hdf5 file) (default: None)
  --show_example        If true, shows example output in comparison to
                        expected output. (default: False)
  --save_results        If true, saves trained model outputs for images in
                        training/test set. (default: False)

```





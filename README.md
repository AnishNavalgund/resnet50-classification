# Flower Classification using ResNet-50

This project performs image classification on the Oxford 102 Flower Dataset using a pre-trained ResNet-50 model in PyTorch.

The model is trained to recognize 102 different flower categories, and we evaluate its performance using top-1 and top-5 accuracy.

---

## Project Structure
```bash
resnet50-classification/
├── data/
│ ├── raw_data/             # Contains all .jpg images of flowers
│ ├── imagelabels.mat       # Label file mapping images to categories
│ └── setid.mat             # File with train/val/test splits (optional)
├── models/                 # Directory for saving trained models
├── outputs/                # Accuracy and loss plots 
├── notebooks/  
│ ├── 01_gpu_check.ipynb     # Check if GPU is available
│ └── 02_eda.ipynb           # Data exploration notebook
├── scripts/    
│ ├── train_resnet.py       # Training script
│ ├── val_resnet.py         # Validation script with top-1 and top-5 accuracy
│ └── split_data.py         # Create train-val-test folders
├── pyproject.toml          # Poetry dependency file
└── README.md
```

---

## Prerequisites

- Python 3.10+
- Poetry (for managing dependencies)
- PyTorch with GPU support (recommended for training)

---

## Dataset Setup

1. Download the dataset from [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
2. `102flowers.tgz` - contains all flower images 
3. `imagelabels.mat` - contains the labels for the images 
4. `setid.mat` - contains the train-val-test splits 

```bash	
data/
├── raw_data/              # extracted images go here
├── imagelabels.mat        # place this file directly in the data folder
└── setid.mat              # place this file directly in the data folder
```

---

## Training the Model

1. Install dependencies and activate the virtual environment

```bash
poetry install --no-root
source .venv/bin/activate
```

2. Command to prepare the data

```bash
poetry run python scripts/split_data.py
```

3. Command to train the model

```bash
poetry run python scripts/train_resnet.py
```
> After training, the model will be saved in the `models/` folder. 
> The training loss and accuracy will be saved in the `outputs/` folder. 

4. Command to validate the model

```bash
poetry run python scripts/val_resnet.py
```

## Example Results

```bash
Top-1 Accuracy: 0.5814
Top-5 Accuracy: 0.8304
```

---

## Future Improvements

- [ ] Add more data augmentation techniques
- [ ] Add more layers to the model
- [ ] Add more regularization techniques        
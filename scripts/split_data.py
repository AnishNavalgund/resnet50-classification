import os
import shutil
from scipy.io import loadmat
from natsort import natsorted

image_dir = "data/raw_data"
label_path = "data/imagelabels.mat"
split_path = "data/setid.mat"
output_dir = "data/split"

# Load given data
image_files = natsorted(os.listdir(image_dir))
labels = loadmat(label_path)["labels"][0]
labels = [l - 1 for l in labels]

# Load official splits
splits = loadmat(split_path)
train_ids = [i - 1 for i in splits['trnid'][0]]
val_ids = [i - 1 for i in splits['valid'][0]]
test_ids = [i - 1 for i in splits['tstid'][0]]

splits_dict = {
    "train": train_ids,
    "val": val_ids,
    "test": test_ids
}

# Create train-val-test folders and copy files
for split, indices in splits_dict.items():
    for idx in indices:
        fname = image_files[idx]
        label = str(labels[idx])
        out_dir = os.path.join(output_dir, split, label)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(os.path.join(image_dir, fname), os.path.join(out_dir, fname))

print("Data split completed.")

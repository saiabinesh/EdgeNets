import os
import sys
import shutil
import random
from tqdm import tqdm

source_dir = "D:\\ROCSAFE\\Datasets\\RQ2\\CIFAR100\\top_80"
# Has to be a descending list because the code samples random directories from previous ones
n_list = [70, 50]

# Create destination directories
destination_dirs = []
for n in n_list:
    destination_dir = os.path.join("D:\\ROCSAFE\\Datasets\\RQ2\\CIFAR100\\top_" + str(n))
    destination_dirs.append(destination_dir)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

# Get list of class directories
class_dirs = os.listdir(os.path.join(source_dir, "train"))
class_dirs.sort()

# Loop through n_list
for i, n in enumerate(n_list):
    if i == 0:
        source_classes = class_dirs
    else:
        source_classes = random.sample(source_classes, n_list[i-1])

    dest_dir = destination_dirs[i]
    dest_classes = random.sample(class_dirs, n)

    for split in ["train", "test", "val"]:
        for c in tqdm(dest_classes, desc=f"{split} {n}"):
            if not os.path.exists(os.path.join(dest_dir, split, c)):
                os.makedirs(os.path.join(dest_dir, split, c))

            src_path = os.path.join(source_dir, split, c)
            dst_path = os.path.join(dest_dir, split, c)

            if not os.path.exists(dst_path):
                shutil.copytree(src_path, dst_path)

            for src_file in os.listdir(src_path):
                dst_file = os.path.join(dst_path, src_file)
                shutil.copy2(os.path.join(src_path, src_file), dst_file)

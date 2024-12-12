import os
import shutil

n_list = [60, 40, 30, 10]

# Define source and destination base directories
source_base_dir = "D:\\ROCSAFE\\Datasets\\RQ2\\CIFAR100\\top_"
destination_base_dir = "D:\\ROCSAFE\\Datasets\\RQ2\\CIFAR100\\top_"

# Move directories to correct location
for n in n_list:
    source_dir = os.path.join(source_base_dir, str(n))
    destination_dir = os.path.join(destination_base_dir + str(n))

    # Move directory if not already in the correct location
    if source_dir != destination_dir:
        if os.path.exists(destination_dir):
            print(f"Destination directory {destination_dir} already exists, removing it")
            shutil.rmtree(destination_dir)
        shutil.move(source_dir, destination_dir)

    # Check number of labels/folders in each split and remove discrepancies
    splits = ["train", "test", "val"]
    split_counts = {}

    for split in splits:
        split_dir = os.path.join(destination_dir, split)
        split_counts[split] = len(os.listdir(split_dir))

    # Find the minimum count among splits
    min_count = min(split_counts.values())

    # Remove extra directories
    for split in splits:
        split_dir = os.path.join(destination_dir, split)
        if split_counts[split] > min_count:
            dirs = os.listdir(split_dir)
            dirs_to_remove = dirs[min_count:]  # dirs that should be removed
            for dir_to_remove in dirs_to_remove:
                shutil.rmtree(os.path.join(split_dir, dir_to_remove))
            print(f"Removed {len(dirs_to_remove)} directories from {split_dir}")

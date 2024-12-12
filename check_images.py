import os
from PIL import Image
from tqdm import tqdm
import PIL
 
n_list = [60]
source_base_dir = "D:\\ROCSAFE\\Datasets\\RQ2\\CIFAR100\\top_"

unidentified_images = []

# Iterate through each n value
for n in n_list:
    # source_dir = os.path.join(source_base_dir, str(n))
    source_dir = source_base_dir + str(n)

    # Iterate through each split directory
    splits = ["train", "test", "val"]
    for split in splits:
        split_dir = os.path.join(source_dir, split)

        # Iterate through each class directory
        classes = os.listdir(split_dir)
        for class_dir in tqdm(classes, desc=f"{split} {n}"):
            class_path = os.path.join(split_dir, class_dir)

            # Iterate through each image file
            images = os.listdir(class_path)
            for image_file in images:
                image_path = os.path.join(class_path, image_file)

                try:
                    # Attempt to open the image with PIL
                    with Image.open(image_path) as img:
                        pass
                except (OSError, IOError, PIL.UnidentifiedImageError):
                    # Store the path of the image that raised the error
                    unidentified_images.append(image_path)

# Print the list of paths to the unidentified images
for image_path in unidentified_images:
    print(image_path)

# import os

# # Define the list of paths to the unidentified images
# unidentified_image_paths = [
#     "path/to/unidentified/image1.jpg",
#     "path/to/unidentified/image2.jpg",
#     # Add the rest of the paths here
# ]

# Iterate through each path and delete the corresponding file
for image_path in unidentified_images:
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Deleted file: {image_path}")
    else:
        print(f"File does not exist: {image_path}")

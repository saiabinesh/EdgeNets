import os
from PIL import Image
import PIL

root_dir = '/mnt/disks/persist/imgnet_top_400'
count_done = 0
count_deleted = 0

def process_image(file_path):
    global count_done, count_deleted
    
    try:
        with Image.open(file_path) as img:
            # Do something with the image, e.g. img.show()
            count_done += 1
            if count_done % 10000 == 0:
                print(f"Processed {count_done} images so far...")
    except (PIL.UnidentifiedImageError, OSError):
        os.remove(file_path)
        count_deleted += 1
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")

def traverse_directory(directory):
    subdirectories = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    for subdir in subdirectories:
        for dirpath, dirnames, filenames in os.walk(os.path.join(directory, subdir)):
            for filename in filenames:
                if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg') or filename.lower().endswith('.png'):
                    file_path = os.path.join(dirpath, filename)
                    process_image(file_path)

traverse_directory(root_dir)

print(f"Processed a total of {count_done} images and deleted {count_deleted} images.")

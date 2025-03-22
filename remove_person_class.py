import os

# Define the paths to the VOC ImageSets directories
VOC2012_PATH = r"D:\ROCSAFE\Repos\EdgeNets\vision_datasets\pascal_voc\VOCdevkit\VOC2012\ImageSets\Main"
VOC2007_PATH = r"D:\ROCSAFE\Repos\EdgeNets\vision_datasets\pascal_voc\VOCdevkit\VOC2007\ImageSets\Main"

def get_person_image_ids(voc_path):
    """Extract image IDs that contain the 'person' class from person_*.txt files."""
    person_images = set()
    
    for file_name in os.listdir(voc_path):
        if file_name.startswith("person_") and file_name.endswith(".txt"):
            file_path = os.path.join(voc_path, file_name)
            with open(file_path, "r") as f:
                for line in f:
                    image_id, label = line.strip().split()
                    if label == "1":  # Only collect images where 'person' is present
                        person_images.add(image_id)
    
    return person_images

def filter_images(file_path, person_images):
    """Modify a given trainval.txt or test.txt file by removing images containing 'person'."""
    if not os.path.exists(file_path):
        return
    
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    filtered_lines = [line for line in lines if line.split()[0] not in person_images]
    
    with open(file_path, "w") as f:
        f.writelines(filtered_lines)

def process_voc2012():
    """Modify trainval.txt in VOC2012 by removing images containing 'person'."""
    person_images = get_person_image_ids(VOC2012_PATH)
    trainval_file = os.path.join(VOC2012_PATH, "trainval.txt")
    filter_images(trainval_file, person_images)
    print(f"Updated trainval.txt in VOC2012. Removed {len(person_images)} images containing 'person'.")

def process_voc2007():
    """Modify trainval.txt and test.txt files for all classes in VOC2007 by removing images containing 'person'."""
    person_images = get_person_image_ids(VOC2007_PATH)
    
    for file_name in os.listdir(VOC2007_PATH):
        if file_name.endswith(".txt"):
            file_path = os.path.join(VOC2007_PATH, file_name)
            filter_images(file_path, person_images)

    print(f"Updated all class trainval.txt and test.txt files in VOC2007. Removed {len(person_images)} images containing 'person'.")

if __name__ == "__main__":
    process_voc2012()
    process_voc2007()
    print("Processing complete.")

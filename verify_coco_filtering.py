import json
import os

def analyze_coco_subset(json_path):
    """
    Returns:
      - num_images_in_file: How many images are declared in 'images'
      - num_annotations: How many total annotations are in 'annotations'
      - unique_image_ids_in_annotations: number of unique image_ids actually used by annotations
      - category_ids: list of category ids included in this subset
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    num_images_in_file = len(images)
    num_annotations = len(annotations)
    
    # Distinct image IDs that have at least one annotation
    annotated_image_ids = set(ann['image_id'] for ann in annotations)
    unique_image_ids_in_annotations = len(annotated_image_ids)
    
    # Category IDs in this subset
    category_ids = [cat['id'] for cat in categories]
    
    return (num_images_in_file, 
            num_annotations, 
            unique_image_ids_in_annotations, 
            category_ids)

def main():
    """
    Modify 'base_dir' to your actual directory containing the top-20, top-40, etc. JSON files.
    Then run the script. It will print out stats for each subset so you can compare.
    """
    base_dir = r"D:\ROCSAFE\Repos\EdgeNets\vision_datasets\coco\annotations"
    # List of subset files you want to analyze (add or remove as needed)
    subset_files = [
        "20instances_train2017.json",
        "40instances_train2017.json",
        "60instances_train2017.json",
        "80instances_train2017.json"
    ]
    
    for subset_file in subset_files:
        json_path = os.path.join(base_dir, subset_file)
        if os.path.exists(json_path):
            (num_imgs, num_anns, unique_img_ids, category_ids) = analyze_coco_subset(json_path)
            print(f"--- {subset_file} ---")
            print(f"Number of images in 'images': {num_imgs}")
            print(f"Number of annotations:        {num_anns}")
            print(f"Unique image_ids in annots:  {unique_img_ids}")
            print(f"Category IDs in subset:      {category_ids}")
            print()
        else:
            print(f"File not found: {json_path}")

if __name__ == "__main__":
    main()

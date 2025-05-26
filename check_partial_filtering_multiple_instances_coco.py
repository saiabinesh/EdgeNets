import json
import os
from collections import defaultdict

def load_annotations(json_path):
    """
    Returns a dict mapping image_id -> set of category_ids
    for each annotation file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_to_cats = defaultdict(set)
    for ann in data['annotations']:
        image_to_cats[ann['image_id']].add(ann['category_id'])
    return image_to_cats

def check_partial_filtering(base_dir, top20_file, top40_file):
    """
    For images that appear in BOTH top-20 and top-40 subsets:
      - Compare the sets of categories in each subset.
      - Print a few examples where top-40 has categories that are
        completely missing in top-20 (which indicates partial filtering).
    """
    # Load image->categories for both subsets
    top20_path = os.path.join(base_dir, top20_file)
    top40_path = os.path.join(base_dir, top40_file)
    
    top20_dict = load_annotations(top20_path)
    top40_dict = load_annotations(top40_path)
    
    # Get intersection of image_ids
    common_image_ids = set(top20_dict.keys()).intersection(set(top40_dict.keys()))
    
    partial_filtered_images = []
    
    for img_id in common_image_ids:
        cats_20 = top20_dict[img_id]
        cats_40 = top40_dict[img_id]
        
        # If top-40 has extra categories not in top-20, that indicates partial filtering
        difference = cats_40 - cats_20
        if difference:  # means top-40 has classes that top-20 doesn't
            partial_filtered_images.append((img_id, cats_20, cats_40))
    
    print(f"Number of images that appear in both {top20_file} and {top40_file}: {len(common_image_ids)}")
    print(f"Number of images that show partial filtering: {len(partial_filtered_images)}")
    
    # Print out a few examples
    for idx, (img_id, cats_20, cats_40) in enumerate(partial_filtered_images[:5], start=1):
        print(f"\nExample #{idx} - image_id {img_id}:")
        print(f"  categories in top-20 subset: {sorted(list(cats_20))}")
        print(f"  categories in top-40 subset: {sorted(list(cats_40))}")
        print(f"  categories in top-40 but not in top-20: {sorted(list(cats_40 - cats_20))}")


if __name__ == "__main__":
    base_dir = r"D:\ROCSAFE\Repos\EdgeNets\vision_datasets\coco\annotations"
    
    check_partial_filtering(base_dir, "20instances_train2017.json", "40instances_train2017.json")

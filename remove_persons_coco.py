import json
import os

PERSON_CAT_ID = 1  # Adjust if your "person" category is labeled differently

def remove_person_instances(input_json, output_json):
    """
    1. Loads COCO annotations from input_json
    2. Removes all annotations where category_id == PERSON_CAT_ID
    3. Removes images that have 0 annotations after step 2
    4. Removes the 'person' category from 'categories'
    5. Writes the filtered dataset to output_json
    """
    with open(input_json, 'r') as f:
        data = json.load(f)

    # 1) Remove all 'person' annotations
    old_annotations = data["annotations"]
    filtered_annotations = [ann for ann in old_annotations if ann["category_id"] != PERSON_CAT_ID]

    # 2) Count annotations by image_id so we know which images remain
    from collections import defaultdict
    ann_count = defaultdict(int)
    for ann in filtered_annotations:
        ann_count[ann["image_id"]] += 1

    # 3) Keep only images that have >=1 annotation
    old_images = data["images"]
    filtered_images = [img for img in old_images if ann_count[img["id"]] > 0]

    # 4) Remove the person category from 'categories'
    old_categories = data["categories"]
    filtered_categories = [cat for cat in old_categories if cat["id"] != PERSON_CAT_ID]

    # 5) Build the new COCO dictionary
    new_data = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": filtered_categories
    }

    # Some COCO files also have "info" or "licenses" sections
    # If they're in your original file, copy them over:
    for key in ["info", "licenses"]:
        if key in data:
            new_data[key] = data[key]

    # 6) Save the filtered annotations
    with open(output_json, 'w') as f:
        json.dump(new_data, f)
    print(f"Created: {output_json}")


def main():
    base_dir = r"D:\ROCSAFE\Repos\EdgeNets\vision_datasets\coco\annotations"

    # List of (original_file, new_file) pairs.
    # Note that for the "80" subset, you might have a file named "instances_train2017.json"
    # or "80instances_train2017.json" – adapt as needed.
    train_pairs = [
        ("20instances_train2017.json", "19instances_train2017.json"),
        ("40instances_train2017.json", "39instances_train2017.json"),
        ("60instances_train2017.json", "59instances_train2017.json"),
        ("instances_train2017.json", "79instances_train2017.json")  # or "80instances_train2017.json"
    ]

    val_pairs = [
        ("20instances_val2017.json", "19instances_val2017.json"),
        ("40instances_val2017.json", "39instances_val2017.json"),
        ("60instances_val2017.json", "59instances_val2017.json"),
        ("instances_val2017.json", "79instances_val2017.json")      # or "80instances_val2017.json"
    ]

    # Process train files
    for (in_file, out_file) in train_pairs:
        in_path = os.path.join(base_dir, in_file)
        out_path = os.path.join(base_dir, out_file)
        if os.path.exists(in_path):
            remove_person_instances(in_path, out_path)
        else:
            print(f"[WARNING] {in_path} not found. Skipping.")

    # Process val files
    for (in_file, out_file) in val_pairs:
        in_path = os.path.join(base_dir, in_file)
        out_path = os.path.join(base_dir, out_file)
        if os.path.exists(in_path):
            remove_person_instances(in_path, out_path)
        else:
            print(f"[WARNING] {in_path} not found. Skipping.")


if __name__ == "__main__":
    main()

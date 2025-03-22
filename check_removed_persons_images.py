import os
import random
import xml.etree.ElementTree as ET

def get_positive_samples_with_class(image_set_file, class_name, annotation_dir, jpeg_dir, num_samples=5, exclude="person"):
    with open(image_set_file, "r") as f:
        lines = [line.strip().split() for line in f if line.strip()]
    
    # Only keep images labeled as positive (i.e., label == "1")
    positive_ids = [img_id for img_id, label in lines if label == "1"]

    random.shuffle(positive_ids)
    checked = 0

    for img_id in positive_ids:
        xml_file = os.path.join(annotation_dir, f"{img_id}.xml")
        jpg_file = os.path.join(jpeg_dir, f"{img_id}.jpg")

        if not os.path.exists(xml_file) or not os.path.exists(jpg_file):
            continue

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            object_classes = [obj.find('name').text for obj in root.findall('object')]

            if exclude in object_classes:
                print(f"❌ {img_id} — contains '{exclude}' — {jpg_file}")
            elif class_name in object_classes:
                print(f"✅ {img_id} — OK — {jpg_file}")
                checked += 1

            if checked >= num_samples:
                break

        except ET.ParseError:
            print(f"⚠️ Error parsing XML for {img_id}")
            continue

def run_for_class(voc_root, class_name, num_samples=5):
    main_dir = os.path.join(voc_root, "ImageSets", "Main")
    ann_dir = os.path.join(voc_root, "Annotations")
    jpeg_dir = os.path.join(voc_root, "JPEGImages")
    
    image_set_file = os.path.join(main_dir, f"{class_name}_trainval.txt")
    if not os.path.exists(image_set_file):
        print(f"❗ File not found: {image_set_file}")
        return
    
    print(f"\n🔍 Checking '{class_name}' in: {voc_root}")
    get_positive_samples_with_class(image_set_file, class_name, ann_dir, jpeg_dir, num_samples)

if __name__ == "__main__":
    VOC2007_ROOT = r"D:\ROCSAFE\Repos\EdgeNets\vision_datasets\pascal_voc\VOCdevkit\VOC2007"
    VOC2012_ROOT = r"D:\ROCSAFE\Repos\EdgeNets\vision_datasets\pascal_voc\VOCdevkit\VOC2012"

    # You can call this for any class and dataset
    run_for_class(VOC2007_ROOT, class_name="horse", num_samples=5)
    run_for_class(VOC2012_ROOT, class_name="car", num_samples=5)

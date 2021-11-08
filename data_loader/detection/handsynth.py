# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import os, json
from torch.utils import data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

#VOC_CLASS_LIST = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train","bottle", "chair",  'dinningtable', 'pottedplant', "sofa", "tv"]
VOC_CLASS_LIST=['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dinningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']


class HandSynthDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, keep_difficult=False, is_training=True):
        self.root = root
        self.transform= transform
        self.target_transform= target_transform
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "image"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "image", self.imgs[idx])
        annotation_path = os.path.join(self.root, "annotations", self.annotations[idx])
        img = Image.open(img_path).convert("RGB")
        annotation=json.load(open(annotation_path, 'r'))
        bounding_box_items_dict = [dict for dict in annotation["items"] if "bounding_box" in dict] # condition 1 satisfied
        # get bounding box coordinates for each mask
        num_objs = len(bounding_box_items_dict)
        # bounding_box_items_dict=annotation["items"]
        # if len(bounding_box_items_dict)<len(annotation["items"]):
            # print(annotation_path)
        boxes = []
        labels_list=[]
        for i in range(num_objs):
            xmin = bounding_box_items_dict[i]["bounding_box"][0]
            xmax =  bounding_box_items_dict[i]["bounding_box"][1]
            ymin =  bounding_box_items_dict[i]["bounding_box"][2]
            ymax =  bounding_box_items_dict[i]["bounding_box"][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels_list.append(VOC_CLASS_LIST.index(bounding_box_items_dict[i]["class"]))
        image=np.array(img)
        boxes, labels=np.array(boxes, dtype=np.float32),np.array(labels_list, dtype=np.int64)
        # image_id = torch.tensor(annotation["id"])
        
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        # if self.transforms is not None:
            # img, target = self.transforms(img, target)
        return (image, boxes, labels)
        
    def _get_annotation(self, image_id):
        annotation_path = os.path.join(self.root, "annotations", self.annotations[idx])
        annotation=json.load(open(annotation_path, 'r'))
        bounding_box_items_dict = [dict for dict in annotation["items"] if "bounding_box" in dict] # condition 1 satisfied
        # get bounding box coordinates for each mask
        num_objs = len(bounding_box_items_dict)
        boxes = []
        labels_list=[]
        for i in range(num_objs):
            xmin = bounding_box_items_dict[i]["bounding_box"][0]
            xmax =  bounding_box_items_dict[i]["bounding_box"][1]
            ymin =  bounding_box_items_dict[i]["bounding_box"][2]
            ymax =  bounding_box_items_dict[i]["bounding_box"][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels_list.append(VOC_CLASS_LIST.index(bounding_box_items_dict[i]["class"]))
        boxes, labels=np.array(boxes, dtype=np.float32),np.array(labels_list, dtype=np.int64)
        return (boxes, labels)
    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)
        # return img, target
    def get_image(self, index):
        img_path = os.path.join(self.root, "image", self.imgs[index])
        annotation_path = os.path.join(self.root, "annotations", self.annotations[index])
        img = Image.open(img_path).convert("RGB")   
        image=np.array(img)        
        if self.transform:
            image, _ = self.transform(image)
        return image
    def __len__(self):
        return len(self.imgs)
    # collate_fn needs for batch


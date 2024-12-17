# ============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
# ============================================

import json
import os
from pycocotools.cocoeval import COCOeval
import numpy as np
import matplotlib.pyplot as plt

# Extend the COCOeval class to add per-class AP summary
class CustomCOCOeval(COCOeval):
    def summarize(self):
        super().summarize()  # Call the original summarize method

        # Compute and print per-class AP50
        ap_per_class = []
        iou_index = 0  # IoU=0.5
        area_index = 0  # "all"
        max_dets_index = 2  # maxDets=100
        for class_index in range(len(self.params.catIds)):
            precision = self.eval['precision'][iou_index, :, class_index, area_index, max_dets_index]
            if precision[precision > -1].size > 0:
                ap = np.mean(precision[precision > -1])
            else:
                ap = -1
            ap_per_class.append(ap)

        print("AP per class (AP50):", ap_per_class)
        self.ap_per_class = ap_per_class  # Store it as an attribute if needed

def compute_confusion_matrix(coco_eval, class_names):
    """
    Computes a confusion matrix using the COCOeval object.

    Args:
        coco_eval: The CustomCOCOeval object after evaluation.
        class_names: List of class names corresponding to class indices.

    Returns:
        confusion_matrix: A numpy array representing the confusion matrix.
    """
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)

    # Loop through evaluation results
    for img_eval in coco_eval.evalImgs:
        if img_eval is None:
            continue

        # Extract matches and categories
        gt_matches = img_eval['gtMatches'][0]  # Take the first IoU threshold for simplicity
        dt_matches = img_eval['dtMatches'][0]
        gt_classes = np.array(img_eval['category_id'], dtype=np.int)  # Ground truth class IDs
        dt_classes = np.array([coco_eval.params.catIds[dt_match - 1] for dt_match in dt_matches if dt_match > 0], dtype=np.int)  # Detected class IDs
        
        # Ensure the arrays are of compatible shape
        if len(gt_classes) != len(dt_classes):
            continue

        # Update confusion matrix
        for gt_class, dt_class in zip(gt_classes, dt_classes):
            confusion_matrix[gt_class, dt_class] += 1

    return confusion_matrix



def save_confusion_matrix(confusion_matrix, class_names, output_dir):
    """
    Save confusion matrix as an image and CSV file.

    Args:
        confusion_matrix: The confusion matrix as a 2D numpy array.
        class_names: List of class names corresponding to class indices.
        output_dir: Directory to save the output files.
    """
    # Save the confusion matrix as an image
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45)
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # Save the confusion matrix to a CSV file
    csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    with open(csv_path, "w") as f:
        f.write("," + ",".join(class_names) + "\n")
        for i, row in enumerate(confusion_matrix):
            f.write(class_names[i] + "," + ",".join(map(str, row)) + "\n")


def coco_evaluation(dataset, predictions, output_dir):
    coco_results = []
    for i, (boxes, labels, scores) in enumerate(predictions):
        image_id, annotation = dataset.get_annotation(i)
        class_mapper = dataset.contiguous_id_to_coco_id
        if labels.shape[0] == 0:
            continue

        boxes = boxes.tolist()
        labels = labels.tolist()
        scores = scores.tolist()
        coco_results.extend(
            [
                {
                    "image_id": image_id,
                    "category_id": class_mapper[labels[k]],
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],  # to xywh format
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    iou_type = 'bbox'
    if output_dir:
        json_result_file = os.path.join(output_dir, iou_type + ".json")
        with open(json_result_file, "w") as f:
            json.dump(coco_results, f)
    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(json_result_file)

    # Use the CustomCOCOeval class instead of the default COCOeval
    coco_eval = CustomCOCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Compute confusion matrix
    class_names = list(dataset.contiguous_id_to_coco_id.values())  # Get class names
    confusion_matrix = compute_confusion_matrix(coco_eval, class_names)
    save_confusion_matrix(confusion_matrix, class_names, output_dir)

    # Return the custom evaluator, which now includes per-class AP
    return coco_eval

import sys
import inspect
import os
import ast
import pickle
from matplotlib.colors import ListedColormap
import argparse
import torch
from utilities.utils import model_parameters, compute_flops
from tqdm import tqdm
from utilities.metrics.evaluate_detection import evaluate
from utilities.print_utils import *
from model.detection.ssd import ssd
from model.detection.box_predictor import BoxPredictor
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import gc
import time

plt.switch_backend('agg')
def eval(model, dataset, predictor):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            image = dataset.get_image(i)
            output = predictor.predict(model, image)
            boxes, labels, scores = [o.to("cpu").numpy() for o in output]
            predictions[i] = (boxes, labels, scores)

    predictions = [predictions[i] for i in predictions.keys()]
    return predictions


def regenerate_features_and_labels_with_background(
    model, dataset, predictor, cfg, output_feature_file, output_label_file
):
    """
    Regenerate features and labels for feature map index 5 (feature map 6) and include background labels.
    """
    model.eval()
    device = next(model.parameters()).device  # Ensure the model is on the correct device

    feature_map_idx = 0  # Only process feature map index 5
    print(f"Processing feature map {feature_map_idx} (feature map 6)")

    all_features = []
    gt_labels = []
    background_label = 0  # Assign 0 as the background label
    coco_80= ['__background__',
                   'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                   'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush'
                ]


    # Removed "boat" to maintain consistency
    top_20 = [
        '__background__', 'person', 'car', 'chair', 'book', 'bottle', 'cup',
        'dining table', 'traffic light', 'bowl', 'handbag', 'bird', 'truck',
        'bench', 'umbrella', 'cow', 'banana', 'motorcycle', 'backpack', 'carrot'
    ]

    top_20_indices = [coco_80.index(class_name) for class_name in top_20]

    def calculate_iou(box1, boxes2):
        """Calculate IoU between one box and a list of boxes."""
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        inter_x1 = np.maximum(box1[0], boxes2[:, 0])
        inter_y1 = np.maximum(box1[1], boxes2[:, 1])
        inter_x2 = np.minimum(box1[2], boxes2[:, 2])
        inter_y2 = np.minimum(box1[3], boxes2[:, 3])

        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        union_area = box1_area + boxes2_area - inter_area

        return inter_area / (union_area + 1e-6)  # Avoid division by zero
    unstored_feature_counts=0

    for i in tqdm(range(len(dataset))):
        image = dataset.get_image(i)
        image_id, annotations = dataset.get_annotation(i)

        # Skip images with no annotations
        if len(annotations[0]) == 0:
            continue

        gt_boxes = annotations[0]
        gt_classes = annotations[1]

        output = predictor.predict(model, image)
        if output[0] is None:
            continue

        feature_maps, pred_boxes, pred_labels, _ = output
        pred_boxes = pred_boxes.int().cpu().numpy()

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            ious = calculate_iou(pred_box, gt_boxes)
            if pred_label not in top_20_indices:
                unstored_feature_counts+=1
                continue            
            max_iou = np.max(ious) if len(ious) > 0 else 0

            # Assign background label if IoU is below threshold
            if max_iou <= 0.5:
                label = background_label
            else:
                max_iou_idx = np.argmax(ious)
                label = gt_classes[max_iou_idx]

            # Crop the image region corresponding to the box
            x1, y1, x2, y2 = pred_box
            height, width, _ = image.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            cropped_image = image[y1:y2, x1:x2, :]

            # Extract features from the cropped image
            cropped_output = predictor.predict(model, cropped_image)
            if cropped_output[0] is None:
                continue

            cropped_feature_maps, _, _, _ = cropped_output
            cropped_feature_map = cropped_feature_maps[feature_map_idx].squeeze(0)
            if len(cropped_feature_map.shape) != 3:
                continue

            feature = cropped_feature_map.cpu().numpy()
            all_features.append(feature)
            gt_labels.append(label)

    print("unstored_feature_counts: ",unstored_feature_counts)
    # Save features and labels
    # Save features and labels
    with open(output_feature_file, "wb") as f:
        pickle.dump(all_features, f)
    with open(output_label_file, "wb") as f:
        pickle.dump(gt_labels, f)

    print(f"Saved features to {output_feature_file}")
    print(f"Saved labels to {output_label_file}")

def compare_features(old_feature_file, new_feature_file, feature_map_count):
    for feature_map_idx in range(feature_map_count):
        with open(old_feature_file.format(feature_map_idx), "rb") as old_f:
            old_features = pickle.load(old_f)
        with open(new_feature_file.format(feature_map_idx), "rb") as new_f:
            new_features = pickle.load(new_f)

        if len(old_features) != len(new_features):
            print(f"Feature count mismatch in feature map {feature_map_idx}: Old={len(old_features)}, New={len(new_features)}")
        else:
            print(f"Feature map {feature_map_idx}: Feature counts match.")

        # Optional: Add further comparisons for feature values if needed

def main(args):
    global COCO_CLASS_LIST
    if args.im_size in [300, 512]:
        print("Getting config")
        from model.detection.ssd_config import get_config
        cfg = get_config(args.im_size)
    else:
        print_error_message('{} image size not supported'.format(args.im_size))
        return

    # Define num_classes and dataset_class based on dataset
    if args.dataset == 'coco':
        from data_loader.detection.coco import COCOObjectDetection, COCO_CLASS_LIST
        dataset_class = COCOObjectDetection(root_dir=args.data_path, transform=None, is_training=False)
        num_classes = len(COCO_CLASS_LIST)
    else:
        print_error_message('{} dataset not supported.'.format(args.dataset))
        exit(-1)

    cfg.NUM_CLASSES = num_classes
    folder_name = f"{num_classes}_classes"
    os.makedirs(folder_name, exist_ok=True)

    print("Loading model")
    model = ssd(args, cfg)
    weight_dict = torch.load(args.weights_test, map_location='cpu')
    model.load_state_dict(weight_dict)  # ['state_dict'])
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = BoxPredictor(cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")

    print("Regenerating features and ground truth labels...")
    feature_file_template = os.path.join(folder_name, "new_features_all_points_{}_map.pkl")
    label_file_template = os.path.join(folder_name, "new_labels_all_points_{}_map.pkl")

    regenerate_features_and_labels_with_background(
        model,
        dataset_class,
        predictor,
        cfg,
        output_feature_file=feature_file_template,
        output_label_file=label_file_template,
    )

    # print("Comparing regenerated features with old features...")
    # old_feature_file_template = "feature_{}_{}_box_v2.pkl"
    # compare_features(old_feature_file_template, feature_file_template, feature_map_count=6)


    # global coco_80
    # coco_80= ['__background__',
    #                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    #                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    #                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    #                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    #                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    #            'kite', 'baseball bat', 'baseball glove', 'skateboard',
    #            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    #            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    #            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    #            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    #            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    #            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    #            'refrigerator', 'book', 'clock', 'vase', 'scissors',
    #            'teddy bear', 'hair drier', 'toothbrush'
    #             ]

    # # removed boat to maintain consistency
    # top_20 = ['__background__', 'person','car', 'chair', 'book', 'bottle', 'cup', 'dining table', 'traffic light', 'bowl', 'handbag', 'bird', 'boat', 'truck', 'bench', 'umbrella', 'cow', 'banana', 'motorcycle', 'backpack', 'carrot'] 

    # top_20_indices= [COCO_CLASS_LIST.index(class_name) for class_name in top_20]
    # #to remove boat as it is not in the features
    # # if 9 in top_20_indices:
    # #     top_20_indices.remove(9)
    # #     top_20_indices.remove(1)
    # # -----------------------------------------------------------------------------
    # # Model
    # # # ---------------------------------------------------------------------------
    # print("Loading model")
    # model = ssd(args, cfg)
    # if args.weights_test:
    #     weight_dict = torch.load(args.weights_test, map_location='cpu')
    #     model.load_state_dict(weight_dict["state_dict"])    # )  # 
    # model_dict = model.state_dict()

    # model.eval()

    # num_gpus = torch.cuda.device_count()
    # device = 'cuda' if num_gpus >= 1 else 'cpu'

    # if num_gpus >= 1:
    #     model = torch.nn.DataParallel(model)
    #     model = model.to(device)
    #     if torch.backends.cudnn.is_available():
    #         import torch.backends.cudnn as cudnn
    #         cudnn.benchmark = True
    #         cudnn.deterministic = True

    # # -----------------------------------------------------------------------------
    # # Evaluate
    # # -----------------------------------------------------------------------------
    # predictor = BoxPredictor(cfg=cfg, device=device)
    # print("Getting predictions")    
    # predictions = eval(model=model, dataset=dataset_class, predictor=predictor)

    # for feature_map_idx in range(6):
    #     if not feature_map_idx==5:
    #         continue        
    #     all_features = []
    #     labels = []
    #     for i in tqdm(range(len(dataset_class))):
    #         image = dataset_class.get_image(i)
    #         output = predictor.predict(model, image)
    #         if output[0] is None:
    #             continue
    #         feature_maps_temp, boxes, label_outputs, scores = output
    #         non_outputs_count = 0  # Count of non-outputs
    #         for box, label in zip(boxes, label_outputs):
    #             x1, y1, x2, y2 = box.int().tolist()  # Convert to a list of integers
    #             height, width, _ = image.shape  # Retrieve the height and width of the image                
    #             x1 = max(0, x1)
    #             y1 = max(0, y1)
    #             x2 = min(width - 1, x2)
    #             y2 = min(height - 1, y2)
    #             cropped_image = image[y1:y2, x1:x2, :]
    #             box_output = predictor.predict(model, cropped_image)
    #             if box_output[0] is None:
    #                 non_outputs_count += 1  # Increment non-outputs count
    #                 continue
    #             feature_maps, boxes_temp, label_outputs_temp, scores = box_output
    #             feature_map = feature_maps[feature_map_idx].squeeze(0)
    #             if len(feature_map.shape) != 3:
    #                 continue
    #             feature = feature_map.cpu().numpy()
    #             all_features.append(feature)
    #             labels.append(label)
    #     print(f"Number of non-outputs in feature map {feature_map_idx}: {non_outputs_count}")
    #     print(f"Total outputs in feature {feature_map_idx}: {len(all_features)}")                
    #     # Save features and labels to pickle files
    #     with open(f"feature_{num_classes}_{feature_map_idx}_box_v2.pkl", "wb") as f:
    #         pickle.dump(all_features, f)
    #     with open(f"labels_{num_classes}_{feature_map_idx}_box_v2.pkl", "wb") as f:
    #         pickle.dump(labels, f)



    # # Loop over all feature maps
    # for i in tqdm(range(6)):
    #     print("Processing feature map: ", i)
    #     if i != 5:  # Only process feature map index 5
    #         continue

    #     start_time = time.time()

    #     # Adjusted to use the new file naming conventions
    #     feature_file = os.path.join(folder_name, f"new_features_{num_classes}_map.pkl")
    #     label_file = os.path.join(folder_name, f"new_labels_{num_classes}_map.pkl")

    #     with open(feature_file, "rb") as f:
    #         feature_i = pickle.load(f)
    #     with open(label_file, "rb") as f:
    #         labels_i = pickle.load(f)

    #     end_time = time.time()
    #     time_taken = end_time - start_time
    #     print(f"Time taken to unpickle feature map {i}: {time_taken:.2f} seconds")

    #     save_path = os.path.join(
    #         folder_name, f"box_feature_map_{num_classes}_classes_feature_{i}_new_cmap_tab_20_no_persons.png"
    #     )

    #     # Remap labels if there are only 20 classes
    #     if num_classes == 21:
    #         labels_i = [coco_80.index(top_20[label]) for label in labels_i]
    #     labels_i = np.array(labels_i)  # Convert labels to numpy array

    #     # Filter feature maps based on labels in the top 20
    #     filter_mask = np.isin(labels_i, top_20_indices)
    #     filtered_features_i = [feature_i[j] for j in range(len(feature_i)) if filter_mask[j]]
    #     filtered_labels_i = labels_i[filter_mask]

    #     # Time the execution of silhouette score calculation
    #     start_time = time.time()
    #     full_sil_score = calculate_silhouette_score(feature_i, labels_i)
    #     end_time = time.time()
    #     time_taken = end_time - start_time
    #     print(f"DB score for feature map {i} with {num_classes} classes: {full_sil_score}")
    #     print(f"Time taken to calculate silhouette score for feature map {i}: {time_taken:.2f} seconds")

    #     # Feature map plotting
    #     start_time = time.time()
    #     plot_feature_map(filtered_features_i, filtered_labels_i, save_path)
    #     end_time = time.time()
    #     time_taken = end_time - start_time
    #     print(f"Time taken to plot feature map {i}: {time_taken:.2f} seconds")
    #     print(time.ctime())

    # Do just labels
    # Loop over all feature maps
    # for i in range(6):
    #     print("Doing feature: ", i)
    #     if not i==5:
    #         continue
    #     # Load label data from pickle file
    #     with open(f"labels_{num_classes}_{i}.pkl", "rb") as f:
    #         labels_i = pickle.load(f)
    #     # labels_i=[top_20.index(COCO_CLASS_LIST[i]) for i in labels_i]
    #     # labels_i = np.array(labels_i)  # convert labels to numpy array
    #     labels_i=[coco_80.index(COCO_CLASS_LIST[i]) for i in labels_i]
    #     labels_i = np.array(labels_i)  # convert labels to numpy array
    #     # Filter feature maps based on labels in the top 20
    #     filter_mask = np.isin(labels_i, top_20_indices)
    #     filtered_labels_i = labels_i[filter_mask]
    #     # Set save path for the t-SNE plot
    #     save_path = os.path.join(folder_name, f"filtered_feature_map_{i}_no_persons_changed_labels.png")

    #     # Call the plot_labels function
    #     plot_labels(filtered_labels_i, save_path)    


def calculate_silhouette_score(features, labels):
    # Stack the feature maps in the list
    f = np.stack(features, axis=0)
    # Reshape each feature map to have a 2D shape of (n_samples, n_features)
    flat_features = np.reshape(f, (f.shape[0], -1))
    # Perform PCA dimensionality reduction
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(flat_features)
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30.0)
    features_tsne = tsne.fit_transform(features_pca)

    score = davies_bouldin_score(features_tsne, labels)
    return score


def plot_feature_map(features, labels, save_path):
    # Stack the feature maps in the list
    f = np.stack(features, axis=0)
    # Reshape each feature map to have a 2D shape of (n_samples, n_features)
    flat_features = np.reshape(f, (f.shape[0], -1))
    # Perform PCA dimensionality reduction
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(flat_features)
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30.0)
    features_tsne = tsne.fit_transform(features_pca)
    
    # Create a ListedColormap
    full_cmap = plt.cm.get_cmap('tab20')
    # Extract 20 colors from the full colormap
    colors = [full_cmap(i) for i in np.linspace(0, 1, 20)]
    # Create a ListedColormap using the extracted colors
    cmap = ListedColormap(colors)
    
    sort_idx = np.argsort(labels)
    labels = labels[sort_idx]
    features_tsne = features_tsne[sort_idx]
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        plt.scatter(features_tsne[labels == label, 0], 
                    features_tsne[labels == label, 1], label=label, color=cmap(i))
    
    # Set plot title
    # plt.title('t-SNE plot of feature maps')
    # Save the scatter plot as an image file
    plt.savefig(save_path)
    # Close the plot to free up memory
    plt.close()
    
    # Create multi-column legend plot
    legend_fig = plt.figure(figsize=(10, 8))
    ax = legend_fig.add_subplot(111)
    # Create a dictionary to map labels to class names
    label_dict = {label_id: coco_80[label_id] for label_id in unique_labels}
    # Create legend handles
    handles = [plt.scatter([], [], s=100, marker='o', color=cmap(i), edgecolor='none') for i in range(len(unique_labels))]
    # Create a list of class names for the legend
    legend_labels = [label_dict[label] for label in unique_labels]
    # Create the legend with multiple columns
    n_columns = 4
    legend = ax.legend(handles, legend_labels, loc='center', frameon=False, ncol=n_columns,
                    bbox_to_anchor=(0.5, 0.5), fontsize=12)
    # Remove the legend border
    legend.get_frame().set_linewidth(0.0)
    # Save the legend as an image file
    legend_fig.savefig(os.path.splitext(save_path)[0] + '_legend.png')
    # Close the legend plot to free up memory
    plt.close(legend_fig)

def verify_labels(feature_file_template, label_file_template, feature_map_count, num_classes):
    for feature_map_idx in range(feature_map_count):
        if not feature_map_idx==5:
            continue
        feature_file = feature_file_template.format(num_classes, feature_map_idx)
        label_file = label_file_template.format(num_classes, feature_map_idx)

        with open(feature_file, "rb") as f:
            features = pickle.load(f)
        with open(label_file, "rb") as f:
            labels = pickle.load(f)

        if len(features) == len(labels):
            print(f"Feature map {feature_map_idx}: Feature and label counts match ({len(features)}).")
        else:
            print(f"Feature map {feature_map_idx}: Mismatch - Features: {len(features)}, Labels: {len(labels)}")

if __name__ == '__main__':
    from commons.general_details import detection_datasets, detection_models

    parser = argparse.ArgumentParser(description='Training detection network')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model', default='espnetv2', choices=detection_models, type=str,
                        help='initialized model path')
    parser.add_argument('--s', default=2.0, type=float, help='Model scale factor')
    parser.add_argument('--dataset', default='pascal', choices=detection_datasets, help='Name of the dataset')
    parser.add_argument('--data-path', default='', help='Dataset path')
    parser.add_argument('--weights-test', default='', help='model weights')
    parser.add_argument('--im-size', default=300, type=int, help='Image size for training')
    # dimension wise network related params
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')

    args = parser.parse_args()
    if not args.weights_test:
        from model.weight_locations.detection import model_weight_map

        model_key = '{}_{}'.format(args.model, args.s)
        dataset_key = '{}_{}x{}'.format(args.dataset, args.im_size, args.im_size)
        assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
        assert dataset_key in model_weight_map[model_key].keys(), '{} does not exist'.format(dataset_key)
        args.weights_test = model_weight_map[model_key][dataset_key]['weights']
        if not os.path.isfile(args.weights_test):
            print_error_message('weight file does not exist: {}'.format(args.weights_test))
    # This key is used to load the ImageNet weights while training. So, set to empty to avoid errors
    args.weights = ''
    args.save_dir = 'results_detection_{}_{}/{}_{}/'.format(args.model, args.s, args.dataset, args.im_size)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # Call main with updated logic
    main(args)

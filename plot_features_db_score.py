import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score
from tqdm import tqdm

# Define COCO classes and top 20 indices (from the previous script)
COCO_CLASS_LIST = [
    '__background__',
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

top_20 = [
    '__background__', 'person', 'car', 'chair', 'book', 'bottle',
    'cup', 'dining table', 'traffic light', 'bowl', 'handbag', 'bird',
    'boat', 'truck', 'bench', 'umbrella', 'cow', 'banana',
    'motorcycle', 'backpack', 'carrot'
]

top_20_indices = [COCO_CLASS_LIST.index(class_name) for class_name in top_20]

def plot_feature_map(features, labels, save_path):
    # Stack the feature maps in the list
    f = np.stack(features, axis=0)
    flat_features = np.reshape(f, (f.shape[0], -1))  # Flatten features

    # PCA reduction
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(flat_features)

    # t-SNE reduction
    tsne = TSNE(n_components=2, perplexity=30.0)
    features_tsne = tsne.fit_transform(features_pca)

    # Define colormap
    full_cmap = plt.cm.get_cmap('tab20')
    colors = [full_cmap(i) for i in np.linspace(0, 1, 20)]
    cmap = ListedColormap(colors)
    
    # Add a distinct color for the background (e.g., gray)
    distinct_colors = ["gray"] + colors  # Gray for background at index 0
    extended_cmap = ListedColormap(distinct_colors)

    # Sort data by labels
    sort_idx = np.argsort(labels)
    labels = labels[sort_idx]
    features_tsne = features_tsne[sort_idx]

    # Create plot
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_color = distinct_colors[label] if label == 0 else cmap(label)
        plt.scatter(
            features_tsne[labels == label, 0],
            features_tsne[labels == label, 1],
            label=f"{COCO_CLASS_LIST[label]}",
            color=label_color,
            alpha=0.6,
        )

    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def calculate_db_score(features, labels):
    f = np.stack(features, axis=0)
    flat_features = np.reshape(f, (f.shape[0], -1))

    # PCA reduction
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(flat_features)

    # t-SNE reduction
    tsne = TSNE(n_components=2, perplexity=30.0)
    features_tsne = tsne.fit_transform(features_pca)

    score = davies_bouldin_score(features_tsne, labels)
    return score

def main(pickle_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    feature_file = os.path.join(pickle_dir, "new_features_all_points_{}_map.pkl")
    label_file = os.path.join(pickle_dir, "new_labels_all_points_{}_map.pkl")

    # Load features and labels
    with open(feature_file, "rb") as f:
        features = pickle.load(f)
    with open(label_file, "rb") as f:
        labels = pickle.load(f)

    # Remap labels
    labels = np.array([top_20_indices.index(label) if label in top_20_indices else -1 for label in labels])
    filter_mask = labels != -1
    features = [features[i] for i in range(len(features)) if filter_mask[i]]
    labels = labels[filter_mask]

    # Calculate DB score
    db_score = calculate_db_score(features, labels)
    print(f"Davies-Bouldin Score: {db_score}")

    # Plot features
    save_path = os.path.join(save_dir, "feature_map_plot.png")
    plot_feature_map(features, labels, save_path)
    print(f"Feature map plot saved to {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot features and calculate DB scores.")
    parser.add_argument("--pickle-dir", required=True, help="Directory containing the pickle files.")
    parser.add_argument("--save-dir", required=True, help="Directory to save the output plots.")

    args = parser.parse_args()
    main(args.pickle_dir, args.save_dir)

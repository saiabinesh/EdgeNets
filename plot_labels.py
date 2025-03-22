import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score

# Define COCO classes and top 20 indices
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

def plot_labels(labels, save_path):
    """
    Plot and save a legend for labels with class names and colors,
    matching the color scheme used in plot_feature_map.
    """
    labels = np.array(labels)
    # Sort the unique labels in ascending order to match the for-loop logic in the first script
    unique_labels = np.sort(np.unique(labels))

    # Define the colormap exactly as in plot_feature_map
    full_cmap = plt.cm.get_cmap('tab20')
    # Generate 20 distinct colors
    colors = [full_cmap(i) for i in np.linspace(0, 1, 20)]
    
    # Create a ListedColormap from these 20 colors
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)

    # Create the figure for legend only
    legend_fig = plt.figure(figsize=(10, 8))
    ax = legend_fig.add_subplot(111)

    handles = []
    legend_labels = []

    for i, label in enumerate(unique_labels):
        # We skip the -1 label if you are using that for "filtered out" classes
        if label == -1:
            continue
        # Use color=cmap(i) to be consistent with plot_feature_map
        color = cmap(i)
        # Create a dummy scatter to represent this label in the legend
        handles.append(
            plt.scatter([], [], s=100, marker='o', color=color, edgecolor='none')
        )
        # Convert from your remapped index to the actual COCO class name
        # top_20_indices[label] gives the real COCO class index
        # so COCO_CLASS_LIST[top_20_indices[label]] is the class name
        class_name = COCO_CLASS_LIST[top_20_indices[label]]
        legend_labels.append(class_name)

    # Create the legend in the center of the axes
    # Adjust ncol to taste (4 or 5 columns, etc.)
    ax.legend(
        handles,
        legend_labels,
        loc='center',
        frameon=False,
        ncol=4,
        bbox_to_anchor=(0.5, 0.5),
        fontsize=12
    )
    ax.axis('off')  # Hide the axes since we only want the legend

    # Save the legend as an image file
    legend_fig.savefig(os.path.splitext(save_path)[0] + '_legend.png')
    plt.close(legend_fig)

def main(pickle_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    label_file = os.path.join(pickle_dir, "new_labels_0_map.pkl")

    # Load labels
    with open(label_file, "rb") as f:
        labels = pickle.load(f)

    # Remap labels
    labels = np.array([
        top_20_indices.index(label) if label in top_20_indices else -1
        for label in labels
    ])
    labels = labels[labels != -1]

    # Plot labels
    save_path = os.path.join(save_dir, "labels_legend.png")
    plot_labels(labels, save_path)
    print(f"Labels legend saved to {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot labels legend.")
    parser.add_argument("--pickle-dir", required=True, help="Directory containing the label pickle file.")
    parser.add_argument("--save-dir", required=True, help="Directory to save the output legend plot.")
    args = parser.parse_args()

    main(args.pickle_dir, args.save_dir)

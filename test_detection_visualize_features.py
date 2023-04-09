import pickle
import argparse, os
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
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2

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

def main(args):
    if args.im_size in [300, 512]:
        print("Getting config")
        from model.detection.ssd_config import get_config
        cfg = get_config(args.im_size)
    else:
        print_error_message('{} image size not supported'.format(args.im_size))
    if args.dataset == 'coco':
        from data_loader.detection.coco import COCOObjectDetection, COCO_CLASS_LIST
        dataset_class = COCOObjectDetection(root_dir=args.data_path, transform=None, is_training=False)
        class_names = COCO_CLASS_LIST
        num_classes = len(COCO_CLASS_LIST)
    else:
        print_error_message('{} dataset not supported.'.format(args.dataset))
        exit(-1)

    cfg.NUM_CLASSES = num_classes
    folder_name = f"{num_classes}_classes"
    # create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    

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


    top_20 = ['__background__','person', 'car', 'chair', 'book', 'bottle', 'cup', 'dining table', 'traffic light', 'bowl', 'handbag', 'bird', 'boat', 'truck', 'bench', 'umbrella', 'cow', 'banana', 'motorcycle', 'backpack', 'carrot'] 

    top_20_indices= [COCO_CLASS_LIST.index(i) for i in top_20]
    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    print("Loading model")
    model = ssd(args, cfg)
    if args.weights_test:
        weight_dict = torch.load(args.weights_test, map_location='cpu')
        model.load_state_dict(weight_dict["state_dict"])    
    model_dict = model.state_dict()

    model.eval()

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus >= 1 else 'cpu'

    if num_gpus >= 1:
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            cudnn.deterministic = True

    # -----------------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------------
    predictor = BoxPredictor(cfg=cfg, device=device)
    print("Getting predictions")    
    # predictions = eval(model=model, dataset=dataset_class, predictor=predictor)

    features = []
    labels = []
    count_non_standard_features=0
    for i in tqdm(range(len(dataset_class))):
        image = dataset_class.get_image(i)
        output = predictor.predict(model, image)
        if output[0] is None:
            continue
        feature_maps, boxes, label_outputs, scores = output 
        label = max(set(label_outputs.cpu().numpy()), key=label_outputs.cpu().numpy().tolist().count) 
        # if label not in top_20_indices:
        #     continue          
        feature_map = feature_maps[0].squeeze(0)
        if len(feature_map.shape) != 3:
            count_non_standard_features += 1
            continue
        feature = feature_map.cpu().numpy()
        feature_resized = cv2.resize(feature.transpose(1, 2, 0), (38, 38))
        feature = feature_resized.transpose(2, 0, 1).mean(axis=(1, 2))
        features.append(feature)
        labels.append(label)

            
    features = np.stack(features, axis=0)
    labels = np.array(labels)

    # Save dictionaries to pickle files
    with open("features_80_classes.pkl", "wb") as f:
        pickle.dump(features, f)

    with open("labels_80_classes.pkl", "wb") as f:
        pickle.dump(labels, f)

    # # Load dictionaries from pickle files
    # with open("features_20_classes.pkl", "rb") as f:
    #     features = pickle.load(f)

    # with open("labels_20_classes.pkl", "rb") as f:
    #     labels = pickle.load(f)        

    # Apply dimensionality reduction
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)

    tsne = TSNE(n_components=2, perplexity=30.0)
    features_tsne = tsne.fit_transform(features_pca)

    #code to alter labels to point to 80 class list
    labels =[coco_80.index(COCO_CLASS_LIST[i]) for i in labels]

    if not isinstance(labels, np.ndarray):
        labels = np.array(labels, dtype=int)    

    # Get the indices that would sort the labels array
    sort_idx = np.argsort(labels)

    # Sort the labels and features_tsne arrays using the sort indices
    labels = labels[sort_idx]
    features_tsne = features_tsne[sort_idx]   

    # Create scatter plot of t-SNE features
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    for label in sorted(top_20_indices, reverse=False):
        class_name = coco_80[label]
        plt.scatter(features_tsne[labels == label, 0], features_tsne[labels == label, 1], label=class_name)
    plt.legend(loc='best')
    plt.savefig(os.path.join(folder_name, "all_features_tsne_sorted.png"))
    plt.close()

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

    main(args)


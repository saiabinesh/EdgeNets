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

    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    print("Loading model")
    model = ssd(args, cfg)
    if args.weights_test:
        weight_dict = torch.load(args.weights_test, map_location='cpu')
        model.load_state_dict(weight_dict["state_dict"])    
    model_dict = model.state_dict()
    # Remove the last layer (classification layer) from the model
    # print(model_dict.keys())
    # del model_dict['classification_headers.0.weight']
    # del model_dict['classification_headers.0.bias']
    # model.load_state_dict(model_dict, strict=False)
    # Remove the last layers from the model
    del model_dict['cls_layers.0.weight']
    del model_dict['cls_layers.0.bias']
    del model_dict['cls_layers.1.weight']
    del model_dict['cls_layers.1.bias']
    del model_dict['cls_layers.2.weight']
    del model_dict['cls_layers.2.bias']
    del model_dict['cls_layers.3.weight']
    del model_dict['cls_layers.3.bias']
    del model_dict['cls_layers.4.weight']
    del model_dict['cls_layers.4.bias']
    del model_dict['cls_layers.5.weight']
    del model_dict['cls_layers.5.bias']

    # Load the modified state dict into the model
    model.load_state_dict(model_dict, strict=False)    
    model.eval()

    # num_params = model_parameters(model)
    # flops = compute_flops(model, input=torch.Tensor(1, 3, cfg.image_size, cfg.image_size))
    # print_info_message('FLOPs for an input of size {}x{}: {:.2f} million'.format(cfg.image_size, cfg.image_size, flops))
    # print_info_message('Network Parameters: {:.2f} million'.format(num_params))

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

    # Extract features within bounding boxes
    features = []
    labels = []
    done = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset_class))):
            if i % 10 == 0:
                print_info_message('Inferring on image {}'.format(i))
            image = dataset_class.get_image(i)
            output = predictor.predict(model, image)
            boxes, labels_, scores = [o.to("cpu").numpy() for o in output]

            for box, label in zip(boxes, labels_):
                x1, y1, x2, y2 = box.astype(int)
                cropped_image = image[:, y1:y2, x1:x2]
                # Pad the image to 300x300 resolution
                # img_h, img_w = image.shape[1:]
                # target_size = 300
                # max_size = 500
                # scale_factor = min(target_size / min(img_h, img_w), max_size / max(img_h, img_w))
                # resized_height = int(round(img_h * scale_factor))
                # resized_width = int(round(img_w * scale_factor))
                # padding_h = target_size - resized_height
                # padding_w = target_size - resized_width
                # top = padding_h // 2
                # bottom = padding_h - top
                # left = padding_w // 2
                # right = padding_w - left
                # # padded_image = F.pad(image, (left, right, top, bottom), "constant", 0)
                # padded_image = F.pad(torch.from_numpy(image), (left, right, top, bottom), "constant", 0)

                # # Crop the padded image and run through network again
                # cropped_image = padded_image[:, y1:y2+padding_h, x1:x2+padding_w]
                feature = model(cropped_image.unsqueeze(0)).squeeze().cpu().numpy()
                features.append(feature)
                labels.append(label)

                if label not in done:
                    features_by_label = feature
                    cropped_images_by_label = cropped_image
                    labels_by_label = [label]
                    done.append(label)
                elif label in done:
                    index = done.index(label)
                    features_by_label[index] = np.concatenate([features_by_label[index], feature])
                    cropped_images_by_label = torch.cat([cropped_images_by_label, cropped_image], dim=0)
                    labels_by_label.append(label)

    # Apply dimensionality reduction
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)
    tsne = TSNE(n_components=2)
    features_tsne = tsne.fit_transform(features_pca)        
    # features = np.vstack(features)
    # labels = np.array(labels)


    # # #Changing code above to get just one feature per image
    # # # Extract features within bounding boxes
    # # print(dir(dataset_class))
    # # features = []
    # # labels = []
    # # count=0
    # # print("len(dataset_class): ",len(dataset_class))
    # # # print("dataset_class.CLASSES: ",dataset_class.CLASSES)

    # # # print(np.unique(dataset_class.CLASSES))
    # # # print(dataset_class.CLASSES[0:10])
    # # with torch.no_grad():
    # #     for label in np.unique(dataset_class.CLASSES):
    # #         indices = np.where(dataset_class.CLASSES == label)[0]
    # #         # print(indices)
    # #         # exit()
    # #         if len(indices) == 0:
    # #             continue
    # #         # Get the first image of this class
    # #         image = dataset_class.get_image(indices[0])
    # #         print(image)
    # #         output = predictor.predict(model, image)
    # #         boxes, labels_, scores = [o.to("cpu").numpy() for o in output]
    # #         count+=1
    # #         # Find the box with this class label and extract features
    # #         for box, label_ in zip(boxes, labels_):
    # #             if label_ == label:
    # #                 x1, y1, x2, y2 = box.astype(int)
    # #                 cropped_image = torch.from_numpy(image[:, y1:y2, x1:x2]).unsqueeze(0)
    # #                 with torch.no_grad():
    # #                     feature = model(torch.from_numpy(cropped_image).unsqueeze(0)).squeeze().cpu().numpy()
    # #                 features.append(feature)
    # #                 labels.append(label)
    # #                 break
    # # print("Images done =",count)
    # # Convert the features and labels to NumPy arrays
    # # features = np.vstack(features)
    # # labels = np.array(labels)
    # Save the features a eatures.pkl', 'wb') as f:
    #     pickle.dump(features_by_label, f)

    # with open('labels.pkl', 'wb') as f:
    #     pickle.dump(labels_by_label, f)    
    # # Extract the last feature map for each image
    # features = []
    # for i in tqdm(range(len(dataset_class))):
    #     image = dataset_class.get_image(i)
    #     with torch.no_grad():
    #         feature = model(image.unsqueeze(0)).squeeze().cpu().numpy()
    #     features.append(feature)

    # # Convert the features to a NumPy array
    # features = np.vstack(features)

    #unpickle

    # Load the pickled variables
    # with open('features.pkl', 'rb') as f:
    #     features = pickle.load(f)

    # with open('labels.pkl', 'rb') as f:
    #     labels = pickle.load(f)
    # Apply dimensionality reduction
    # Save cropped images and corresponding features for each label
    for label, features, cropped_images in zip(labels_by_label, features_by_label, cropped_images_by_label):
        # Plot and save features
        plt.figure(figsize=(12, 8))
        features_pca = pca.transform(features.reshape(1, -1))
        features_tsne = tsne.transform(features_pca)
        plt.scatter(features_tsne[:, 0], features_tsne[:, 1])
        plt.savefig(os.path.join(folder_name, f"{label}_feature.png"))
        plt.close()

        # Save cropped images
        for i in range(len(cropped_images)):
            filename = os.path.join(folder_name, f"{label}_{i}_image.png")
            img = np.transpose(cropped_images[i].cpu().numpy(), (1, 2, 0))
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # Plot features for all labels
    features = np.vstack(features_by_label)
    labels = np.array(labels_by_label)
    features_pca = pca.fit_transform(features)
    features_tsne = tsne.fit_transform(features_pca)
    for label in np.unique(labels):
        plt.figure(figsize=(12, 8))
        indices = np.where(labels == label)
        plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], label=str(label))
        plt.legend()
        filename = os.path.join(folder_name, f"{label}.png")
        plt.savefig(filename)
        print(f"Saving {filename}")
        plt.close()

    # -----------------------------------------------------------------------------
    # Results
    # # -----------------------------------------------------------------------------
    # predictor = BoxPredictor(cfg=cfg, device=device)
    # predictions = eval(model=model, dataset=dataset_class, predictor=predictor)
    # result_info = evaluate(dataset=dataset_class, predictions=predictions, output_dir=args.save_dir,
    #                        dataset_name=args.dataset)

    # if args.dataset == 'coco':
    #     print_info_message('AP_IoU=0.50:0.95: {}'.format(result_info.stats[0]))
    #     print_info_message('AP_IoU=0.50: {}'.format(result_info.stats[1]))
    #     print_info_message('AP_IoU=0.75: {}'.format(result_info.stats[2]))
    # else:
    #     print_error_message('{} dataset not supported.'.format(args.dataset))


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


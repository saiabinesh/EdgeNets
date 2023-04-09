#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================
import re
import torch
from model.detection.generate_priors import PriorBox
from utilities import box_utils
from utilities.nms import nms
from data_loader.detection.augmentation import TestTransform
import numpy as np
class BoxPredictor(object):

    def __init__(self, cfg, device='cpu'):
        super(BoxPredictor, self).__init__()
        #device = 'cuda' if torch.cuda.device_count() > 1 else 'cpu'
        self.priors = PriorBox(cfg)().to(device)
        self.center_var = cfg.center_variance
        self.size_var = cfg.size_variance
        self.filter_threshold = cfg.conf_threshold
        self.nms_threshold = cfg.iou_threshold
        self.top_k = cfg.top_k
        self.nms_threshold = cfg.iou_threshold
        self.top_k = cfg.top_k
        self.transform = TestTransform(size=cfg.image_size)
        self.softmax = torch.nn.Softmax(dim=2)
        self.device = device
        self.nms_threshold=0.2
        self.filter_threshold=0.5

    def predict(self, model, image, is_scaling=True):
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            fms, confidences, locations = model(images)
            scores = self.softmax(confidences)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.center_var, self.size_var
            )
            boxes = box_utils.center_form_to_corner_form(boxes)

        boxes = boxes[0]
        scores = scores[0]

        filtered_box_probs = []
        filtered_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > self.filter_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            masked_boxes = boxes[mask, :]
            box_and_probs = torch.cat((masked_boxes, probs.reshape(-1, 1)), dim=1)
            box_and_probs = nms(box_and_probs, self.nms_threshold, top_k=self.top_k)
            filtered_box_probs.append(box_and_probs)
            filtered_labels.extend([class_index] * box_and_probs.size(0))
        # no object detected
        if not filtered_box_probs:
            return None, torch.empty(0, 4), torch.empty(0), torch.empty(0)
        # concatenate all results
        filtered_box_probs = torch.cat(filtered_box_probs)
        if is_scaling:
            filtered_box_probs[:, 0] *= width
            filtered_box_probs[:, 1] *= height
            filtered_box_probs[:, 2] *= width
            filtered_box_probs[:, 3] *= height
        return fms, filtered_box_probs[:, :4], torch.tensor(filtered_labels), filtered_box_probs[:, 4]

    def predict_output(self, model, image, is_scaling=True):
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            fms, confidences, locations = model(images)
            # Compute the mean of each feature map along the spatial dimensions
            fms = [fm.squeeze().cpu().numpy() for fm in fms]    
            # Squeeze and move the tensors to CPU memory and convert to numpy arrays
            confidences = confidences.squeeze().cpu().numpy()
            locations = locations.squeeze().cpu().numpy()

        # Modify this block to ensure that features is a list of lists
        features = []
        for i in range(len(fms)):
            features.append([fms[i]])

        # Concatenate the feature maps along the channel dimension
        features = [np.concatenate(f, axis=0) for f in features]

        # Return the feature maps, confidences, and locations as a tuple
        return features, confidences, locations 
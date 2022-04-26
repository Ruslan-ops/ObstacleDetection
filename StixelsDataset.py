import os.path

import cv2
from torch.utils.data import Dataset
import pandas as pd
import torch
import settings
import numpy as np
from pathlib import Path


class StixelsDataset(Dataset):
    def __init__(self, annotations_path, target_transform=None):
        self.annotations = self._read_annotations_txt(annotations_path)
        self.dataset_path = Path(annotations_path).parent
        self.images_path = os.path.join(self.dataset_path, 'images/')
        self.targets_path = os.path.join(self.dataset_path, 'targets/')
        self.dataset_parent_path = self.dataset_path.parent
        #self.transform = transform
        self.targets_transform = target_transform

    def __len__(self):
        return 4 #len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.images_path, self.annotations[index][0])
        target_path = os.path.join(self.targets_path, self.annotations[index][1])
        image = cv2.imread(img_path)
        h, w, c = image.shape
        image = image[:, :, (2, 1, 0)]
        have_target, targets, image = self._get_target(target_path, image)
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image, have_target, targets

    def _get_target(self, target_path, image):
        #h, w = image.shape[0], image.shape[1]
        stixel_columns_amount = settings.STIXEL_COLOMNS_AMOUNT
        bins_amount = settings.BINS_AMOUNT
        pointset = self._read_target_file(target_path)
        transformed = self.targets_transform(image=image, keypoints=pointset)
        image, pointset = transformed['image'], transformed['keypoints']
        pointset = np.array(pointset, dtype=np.float32)
        pointset[:, 0] = pointset[:, 0] / (image.shape[1] + 1)
        pointset[:, 1] = pointset[:, 1] / image.shape[0]
        have_target = np.zeros((stixel_columns_amount), dtype=np.float32)
        targets = np.zeros((stixel_columns_amount), dtype=np.float32)
        for p in pointset:
            index = int(p[0] * stixel_columns_amount)
            op = p[1] * bins_amount
            if op > targets[index]:
                targets[index] = op
                have_target[index] = 1
        targets = np.clip(targets, 0.51, 49.49)
        return have_target, targets, image

    def _read_target_file(self, target_path):
        pointset = []
        with open(target_path) as f:
            for line in f:
                line_info = line.split(' ')
                x, y = int(line_info[0]), int(line_info[1].strip())
                pointset.append((x, y))
        return pointset

    def _read_annotations_txt(self, annotation_path):
        annotations = []
        with open(annotation_path) as f:
            for line in f:
                line_info = line.split('\t')
                index, image_path, target_path = line_info[0], line_info[1], line_info[2].strip()
                annotations.append((image_path, target_path))
        return annotations

    def get_original_image(self, index):
        img_path = os.path.join(self.images_path, self.annotations[index][0])
        image = cv2.imread(img_path)

        return image

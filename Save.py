import os
import numpy as np
import cv2


class Save2Folder:
    def __init__(self, img_clean, img_label, save_path):
        self.labels = img_label
        self.src = img_clean
        self.parentDir = save_path

    def createFolder(self):
        mask_label_flatten = self.labels.flatten()
        unique_labels = np.unique(mask_label_flatten)
        for label in unique_labels:
            os.mkdir(os.path.join(self.parentDir, label))


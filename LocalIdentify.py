import numpy as np
import cv2
import matplotlib.pyplot as plt
import PreProcess as pre
from sklearn.cluster import KMeans
from scipy.signal import find_peaks


class imshow:
    def __init__(self, orig, clean_img, mask_label, label_num):
        self.orig = orig
        self.clean_img = clean_img
        self.orig_enhance = pre.im(orig).enhance_img
        self.clean_img_enhance = pre.im(clean_img).enhance_img
        self.mask_label = mask_label
        self.label_num = label_num
        self.xmin, self.ymin, self.xmax, self.ymax = self.extract_small_region()
        self.orig_square = self.draw_hollow_square(orig.copy())
        self.mask_roi = self.mask_roi()
        self.orig_roi = self.orig_roi()
        self.pixel_list = self.pixel_list()
        self.hist_l, self.hist_a, self.hist_b = self.hist(self.clean_img_enhance)
        self.kMeans = self.kmeans()
        self.label_roi = self.kMeans.labels_.reshape(self.mask_roi.shape[:2])
        self.centroids_roi = self.kMeans.cluster_centers_
        self.unique, self.counts = np.unique(self.label_roi, return_counts=True)

    def extract_small_region(self):
        ys, xs = np.where(self.mask_label == self.label_num)
        return xs.min(), ys.min(), xs.max(), ys.max()

    def mask_roi(self):
        return self.mask_label[self.ymin:self.ymax + 1, self.xmin:self.xmax + 1]

    def orig_roi(self):
        return self.orig[self.ymin:self.ymax + 1, self.xmin:self.xmax + 1]

    def pixel_list(self):
        return self.clean_img[self.ymin:self.ymax + 1, self.xmin:self.xmax + 1].reshape((-1, 3))

    def draw_hollow_square(self, img):
        for i in range(self.ymin, self.ymax + 1):
            # Loop through each column
            for j in range(self.xmin, self.xmax + 1):
                # If the current row or column is the first or last, print a "*"
                if i == self.ymin or i == self.ymax or j == self.xmin or j == self.xmax:
                    img[i, j] = (255, 0, 0)
        return img

    def hist(self, img):
        img = img[self.ymin:self.ymax + 1, self.xmin:self.xmax + 1]
        lab_roi = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        # Convert to 2D list
        lab_roi = lab_roi[np.where(self.mask_roi == self.label_num)]

        # Calculate the histogram of each channel
        hist_l, bins_l = np.histogram(lab_roi[:, 0], bins=256, range=(0, 256))
        hist_a, bins_a = np.histogram(lab_roi[:, 1], bins=256, range=(0, 256))
        hist_b, bins_b = np.histogram(lab_roi[:, 2], bins=256, range=(0, 256))
        return hist_l, hist_a, hist_b

    def kmeans(self):
        # Find the peaks in each histogram
        peaks_l, _ = find_peaks(self.hist_l, distance=10)
        peaks_a, _ = find_peaks(self.hist_a, distance=10)
        peaks_b, _ = find_peaks(self.hist_b, distance=10)
        if np.std(self.hist_l) > len(peaks_l):
            # num_clusters = min(len(peaks_a), len(peaks_b))*(max((len(peaks_a), len(peaks_b)))-1)
            num_clusters = int(np.floor((np.log(np.std(self.hist_l) / len(peaks_l)))) * (max(len(peaks_a), len(peaks_b))) + (min(len(peaks_a), len(peaks_b))))
            # Perform K-means clustering on the pixel values using the optimal number of clusters
            if num_clusters <= 1:
                num_clusters = 2

            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(self.pixel_list)

            # Get the labels assigned to each pixel by the K-means clustering algorithm
            labels = kmeans.labels_
            self.centroids_roi = kmeans.cluster_centers_
            self.unique, self.counts = np.unique(labels, return_counts=True)
        return kmeans

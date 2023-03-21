import cv2
import Mask
import numpy as np
import PreProcess as pre
import LocalIdentify as loc
import matplotlib.pyplot as plt


class Input:
    def __init__(self, imgPath):
        self.src_path = imgPath
        self.label_len = 0
        self.label_num = 0
        self.original = cv2.cvtColor(cv2.imread(self.src_path), cv2.COLOR_BGR2RGB)
        self.src = pre.im(self.original)
        self.bg_col = self.src.bgColExtract()
        self.image_clean, self.label_mask, self.label_len = Mask.clean_image(self.original, self.bg_col)
        self.kMeans_output = []
        self.roi = self.roi_mask = self.labeled_roi_img = self.square = None

    def process(self, label_num):
        self.label_num = label_num
        # Copy recognized parts to a single color image. New image called image_clean
        image_clean_blurred = cv2.GaussianBlur(self.image_clean, (5, 5), 3)
        image_clean_blurred = cv2.medianBlur(image_clean_blurred, 5)
        # segment flacks base on label number
        event = loc.imshow(self.original, image_clean_blurred, self.label_mask, label_num=label_num)
        self.roi = event.orig_roi
        self.roi_mask = event.mask_roi
        self.labeled_roi_img = event.label_roi
        self.square = event.orig_square
        self.layer_info_process(event)
        color_labeled_roi = self.label_img_coloring(self.labeled_roi_img)
        self.plt(self.roi, "normalized b value", type='img')
        return self.roi, color_labeled_roi, self.square

    def layer_info_process(self, event):
        cols = event.centroids_roi
        bg = self.bg_col
        '''
        need to add a function to classify different layers
        '''
        layer_1 = 7  # predefined
        layer_2 = 20  # predefined
        layer_3 = 30  # predefined
        layer_4 = 40  # predefined
        layer_5 = 50  # predefined

        diff = []
        output = []
        for i, col in enumerate(cols):
            r_diff = abs(col[0] - bg[0])
            g_diff = abs(col[1] - bg[1])
            b_diff = abs(col[2] - bg[2])
            col_diff = (r_diff ** 2 + g_diff ** 2 + b_diff ** 2) ** 0.5
            output.append(col_diff)

            if col_diff <= layer_1:
                diff.append(0)
            elif layer_1 <= col_diff <= layer_2:
                diff.append(1)
            elif layer_2 <= col_diff <= layer_3:
                diff.append(2)
            elif layer_3 <= col_diff <= layer_4:
                diff.append(3)
            elif layer_4 <= col_diff <= layer_5:
                diff.append(4)
            else:
                diff.append(5)
        # print(diff)
        # print(output)
        tmp = self.labeled_roi_img.copy()
        for j, layer in enumerate(diff):
            if layer == 0:
                tmp[np.where(self.labeled_roi_img == j)] = 0
        self.labeled_roi_img = tmp

    def label_img_coloring(self, labels):
        labels = labels.copy()
        mask_small = self.roi_mask
        # Map component labels to hue val, 0-179 is the hue range in OpenCV
        if np.max(labels) > 0:
            label_hue = np.uint8(179 * labels / np.max(labels))
            blank_ch = 255 * np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

            # Converting cvt to BGR
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

            # set bg label to black
            labeled_img[label_hue == 0] = 0
            labeled_img[np.where(mask_small != self.label_num)] = 0
            labeled_img = cv2.medianBlur(labeled_img, 5)
        else:
            labeled_img = labels
        return labeled_img

    def plt(self, src, xlabel, type):
        if type == "plot":
            plt.plot(src, color="b")
            plt.xlabel(xlabel, fontsize=16)
            plt.ylabel("number of pixels in ROI", fontsize=16)
            plt.show()
        else:
            plt.imshow(src)
            plt.tick_params(left=False, right=False, labelleft=False,
                            labelbottom=False, bottom=False)
            plt.show()

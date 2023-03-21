import cv2
import numpy as np
import PreProcess as pre
import matplotlib.pyplot as plt


class MixEdge:
    """
    A new edge detection combine sobel and laplacian

    Args:
        arg: Original RGB color image

    Attributes:
        src: This is where we store original enhanced image input,
        src_lap: This is where we store laplacian edge detected image
        src_sob: This is where we store sobel edge detected image
        combine: This is where we store image combining two detection methods
    """

    def __init__(self, arg):
        self.src = pre.im(arg).enhance_img
        self.src_lap = self.Laplacian()
        self.src_sob = self.sobel()
        self.combine = self.combineTwo()

    def Laplacian(self):
        """
        Laplacian edge detection
        :return: Laplacian edge detected image
        """
        colImg = self.src
        if len(colImg.shape) == 3:
            R = cv2.Laplacian(colImg[:, :, 0], cv2.CV_16S, ksize=7)
            _, R = cv2.threshold(R, 0, 255, cv2.THRESH_BINARY)
            G = cv2.Laplacian(colImg[:, :, 1], cv2.CV_16S, ksize=7)
            _, G = cv2.threshold(G, 0, 255, cv2.THRESH_BINARY)
            B = cv2.Laplacian(colImg[:, :, 2], cv2.CV_16S, ksize=7)
            _, B = cv2.threshold(B, 0, 255, cv2.THRESH_BINARY)
            dst = np.uint8((R + G + B) / 3)
        else:
            dst = cv2.Laplacian(colImg, cv2.CV_16S, ksize=3)
            _, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)
            dst = abs(dst)
        return dst

    def sobel(self):
        """
        Sobel edge detection
        :return: Sobel edge detected image
        """
        img = self.src
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)

        sobel_img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        dst = sobel_img.copy()
        dst[np.where(sobel_img <= 15)] = 0
        return dst

    def Prewitt(self):
        img = self.src
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        x = cv2.filter2D(img_gaussian, -1, kernelx)
        y = cv2.filter2D(img_gaussian, -1, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        sobel_img = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        dst = sobel_img.copy()
        dst[np.where(sobel_img <= 15)] = 0
        return dst

    def combineTwo(self):
        """
        Combining laplacian detected image and sobel detected image
        use the value of pixel to reduce some noises
        :return: image combining two detection methods
        """
        # dst = self.src_lap.copy()
        # dst[np.where(self.src_sob <= 15)] = 0
        ####################################################
        test = cv2.GaussianBlur(self.src_sob, (3, 3), 0)
        test[np.where(test >= 10)] = 255
        dst = cv2.medianBlur(test, 3)
        return dst


class BinaryMask:
    def __init__(self, edge_img):
        self.src = edge_img
        self.closedEdge = self.edge_close()
        self.minArea = 1000
        self.area = self.areaSelect()
        self.mask = self.fillEdge()

    def edge_close(self):
        # src_blur = self.src
        # src_blur = cv2.medianBlur(self.src, 5)
        src_blur = cv2.GaussianBlur(self.src, (5, 5), 0)
        src_dilate = cv2.dilate(src_blur, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=5)
        src_erode = cv2.erode(src_dilate, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=4)

        rows, cols = src_erode.shape
        dst = np.zeros((rows, cols), np.uint8)
        dst[np.where(src_erode >= 100)] = 255
        return dst

    def areaSelect(self):
        contours, hierarchy = cv2.findContours(self.closedEdge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        useful_areas = []
        for k, contour in enumerate(contours):
            if cv2.contourArea(contour) >= self.minArea:
                useful_areas.append(contour)
        return useful_areas

    def fillEdge(self):
        rows, cols = self.closedEdge.shape
        mask = np.full((rows, cols), 0, np.uint8)
        cv2.fillPoly(mask, self.area, 255)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel, iterations=1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask


def get_mask(original_image):
    edge_combine = MixEdge(original_image).combine
    mask = BinaryMask(edge_combine).mask
    return mask


def clean_image(original_image, background_color):
    mask = get_mask(original_image)
    # get closed region
    num_labels, label_image = cv2.connectedComponents(mask)
    # delete small region
    for i in range(num_labels):
        if len(label_image[np.where(label_image == i)]) < 1000:
            mask[np.where(label_image == i)] = 0
    num_labels, label_image = cv2.connectedComponents(mask)
    # copy mask covered region to a single color region
    rows, cols = mask.shape
    dst = np.full((rows, cols, 3), background_color, np.uint8)
    dst[np.where(mask != 0)] = original_image[np.where(mask != 0)]
    return dst, label_image, num_labels

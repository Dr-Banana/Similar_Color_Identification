import numpy as np
import Img_Segementation as seg
import cv2
import datetime
import matplotlib.pyplot as plt


def PreProcess(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    originimg = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.equalizeHist(img)
    return img, originimg


def EdgeDetection(img):
    mask = img.copy()

    # thre
    thresh, img_th = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    # close
    kernel = np.ones((7, 7), np.uint8)
    img_close = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, kernel)

    # canny
    canny = cv2.Canny(np.uint8(img_close), 100, 210)
    return canny


def fillEdge(areas, img):
    x, y, c = img.shape
    black_masks = np.zeros((x, y))
    # 将区域涂实
    cv2.fillPoly(black_masks, areas, 1)
    kernel = np.ones((5, 5), np.uint8)
    black_masks = cv2.erode(black_masks, kernel, iterations=1)
    colorValue = []
    for i in range(0, x):
        for j in range(0, y):
            if black_masks[i, j] == 1:
                colorValue.append(img[i, j])
    # plt.imshow(black_masks)
    # plt.show()
    return black_masks, colorValue


def contourSelect(img):
    # component
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    useful_areas = []
    for k, contour in enumerate(contours):
        if cv2.contourArea(contour) >= 400:  # 添加筛选条件
            useful_areas.append(contour)
    return useful_areas


def drawContours(areas, img):
    # 此时轮廓已经绘制在原图的拷贝上
    outImg = cv2.drawContours(img, areas, -1, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    return outImg


def imgshow(img):
    cv2.imshow("output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

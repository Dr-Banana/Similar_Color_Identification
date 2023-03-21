import os
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import colorsys


# calculate the mean value for RGB
def computeMean3D(path):
    """

    :param path: 图片路径（已换成RGB颜色）
    :return: 一张图片里的RGB平均值
    """
    file_names = os.listdir(path)
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    for file_name in file_names:
        img = cv2.imread(os.path.join(path, file_name), 1)
        per_image_Rmean.append(math.ceil((np.mean(img[:, :, 0]))))
        per_image_Gmean.append(math.ceil((np.mean(img[:, :, 1]))))
        per_image_Bmean.append(math.ceil((np.mean(img[:, :, 2]))))
    return per_image_Rmean, per_image_Gmean, per_image_Bmean


# make the 3D plot


def TDPlot3D(list, xlab, ylab, zlab):
    C = list
    per_image_1 = list[:, :, 0]
    per_image_2 = list[:, :, 1]
    per_image_3 = list[:, :, 2]
    # change HSV to RGB
    # for pointLen in range(0, len(per_image_1)):
    #     tmp = [per_image_1[pointLen], per_image_2[pointLen], per_image_3[pointLen]]
    #     C.append(tmp)

    ax = plt.axes(projection='3d')
    ax.scatter3D(per_image_1, per_image_2, per_image_3,
                 color=np.array(C) / 255.0)
    ax.set_xlabel(xlab, fontdict={'size': 15})
    ax.set_ylabel(ylab, fontdict={'size': 15})
    ax.set_zlabel(zlab, fontdict={'size': 15})

    plt.show()


# read image from path and convert into equalize gray image.
# calculate the mean value of image
def computeMean1D(path):
    file_names = os.listdir(path)
    per_image_Gmean = []
    for file_name in file_names:
        img = cv2.imread(os.path.join(path, file_name), cv2.IMREAD_GRAYSCALE)
        dst = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
        grayEq_b = cv2.equalizeHist(dst)
        per_image_Gmean.append(math.ceil((np.mean(grayEq_b))))
    return per_image_Gmean


def plotSideBySide(image, title, savePath=None, gray=False):
    plotSize = len(image)
    rows = int(np.floor(plotSize/2))
    if rows == 0:
        rows = 1
    cols = int(np.ceil(plotSize/rows))
    f = plt.figure()
    for i in range(plotSize):
        f.add_subplot(rows, cols, i+1)
        if gray:
            plt.imshow(image[i],cmap="gray")
        else:
            plt.imshow(image[i])
        plt.title(title[i])
    # plt.show()
    if savePath:
        f.savefig(savePath)
    return f

import numpy as np
import plotD3 as pd3
import kmeans1d


def segment(picture, cut_width, cut_length):
    """
    将图片分割成小片并当初为一个长list，主要用于图片分析

    :param picture:导入需要分割的图片
    :param cut_width: 想要被分割的宽度
    :param cut_length: 想要被分割的高度
    :return: [[[[R G B]...[R G B]](3层长度是图片宽度)](2层长度是图片高度)](1层长度是分出的图片数量)
    """
    (width, length, depth) = picture.shape
    # 预处理生成0矩阵
    pic = np.zeros((cut_width, cut_length, depth))
    # 计算可以划分的横纵的个数
    num_width = int(width / cut_width)
    num_length = int(length / cut_length)
    segment_list = []
    # for循环迭代生成
    for i in range(0, num_width):
        for j in range(0, num_length):
            pic = picture[i * cut_width: (i + 1) * cut_width, j * cut_length: (j + 1) * cut_length, :]
            segment_list.append(pic)
    return segment_list


def kMeans_gray_1d(img, k, path=None, saveImg=False):
    flatten_grayEq_b = img.flatten()
    _, centroids = kmeans1d.cluster(flatten_grayEq_b, k)

    centroid_1 = np.floor(centroids[0])
    rows, cols = img.shape
    out_list = np.zeros((rows, cols, 1))
    for i in range(rows):  # 遍历每一行
        for j in range(cols):  # 遍历每一列
            if float(img[i, j]) > centroid_1:
                out_list[i, j] = 255
    if saveImg:
        pd3.plotSideBySide(out_list, "image from cluster",
                           "result/binary_image/" + path + "/" + str(k) + "_" + path)
    return out_list


import cv2
import numpy as np
from numba import jit
from itertools import product
import datetime

# starttime = datetime.datetime.now()
#
# endtime = datetime.datetime.now()
# print("1", endtime - starttime)

@jit()
def getPeak(img):
    """
    遍历每个点
    用3*3 kernel决定点是否为顶点(peak)

    :param img: 一定为binary图片
    :return: 识别出的顶点list
    """
    p = []
    rows, cols = img.shape
    for i in range(1, rows - 1):  # 遍历每一行
        for j in range(1, cols - 1):  # 遍历每一列
            if img[i, j] == 255:  # 判断该点是否为bai点，0代表黑点
                r = []
                for y in range(i - 1, i + 2):
                    for x in range(j - 1, j + 2):
                        if y == i and x == j:
                            continue
                        if img[y, x] == 255:
                            r.append([y, x])

                rLen = len(r)
                if rLen > 0:
                    if rLen == 1:
                        p.append([i, j])
                    elif rLen == 2:
                        dy = r[0][0] - r[1][0]
                        dx = r[0][1] - r[1][1]
                        t = dx * dx + dy * dy
                        if t == 1:
                            p.append([i, j])
    return p


def get_line(x, y, img):
    """
    Bresenham 直线算法
    连接给定两点

    :param x:[x1,x2]起始点和重点x坐标
    :param y:[y1,y2]起始点和重点y坐标
    :param img: binary图片
    :return: 连接好线段的图片
    """
    # Setup initial conditions
    x1, x2 = x
    y1, y2 = y
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        img[coord[1], coord[0]] = 255
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points


def connectEdge(img, distanceControl=None):
    """


    :param img: binary图片
    :param distanceControl: 预设200
    :return: 经过距离筛选后并连接了顶点的图片
    """
    pList = getPeak(img)
    while len(pList) > 1:
        x = []
        for j in range(1, len(pList)):
            dy = pList[0][0] - pList[j][0]
            dx = pList[0][1] - pList[j][1]
            t = dx ** 2 + dy ** 2
            x.append(t)

        min_index = x.index(min(x)) + 1
        if min(x) < 200:
            x = [pList[0][1], pList[min_index][1]]
            y = [pList[0][0], pList[min_index][0]]
            get_line(x, y, img)
        pList.pop(min_index)
        pList.pop(0)
    kernel = np.ones((3, 3), np.uint8)
    img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img_close


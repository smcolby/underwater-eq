import cv2
import numpy as np
import matplotlib.pyplot as plt
from process import equalizeBGR


def nothing(x):
    pass

f = '/home/sean/Pictures/Dive 5 - Apr 10 (Mala Pier)/IMG_7135.JPG'

# load image
im = cv2.imread(f, -1)
im = cv2.resize(im, None, 0, 0.5, 0.5, cv2.INTER_AREA)
bgr = im.copy()

cv2.namedWindow('CLAHE')
cv2.createTrackbar('Clip Limit', 'CLAHE', 5, 200, nothing)
cv2.createTrackbar('Grid Dimension', 'CLAHE', 1, 32, nothing)

while True:
    cv2.imshow('CLAHE', bgr)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    c = cv2.getTrackbarPos('Clip Limit', 'CLAHE') / 5.0
    w = cv2.getTrackbarPos('Grid Dimension', 'CLAHE')

    c = max(0.001, c)
    w = max(1, w)

    # equalize bgr
    bgr = equalizeBGR(im, clip=c, width=w)
    # plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

    # plt.show()

cv2.destroyAllWindows()

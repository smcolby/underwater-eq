import cv2
import numpy as np
import matplotlib.pyplot as plt
from process import equalizeBGR


def nothing(x):
    pass

f = '/home/sean/Pictures/Dive 5 - Apr 10 (Mala Pier)/IMG_7135.JPG'

# load image
im = cv2.imread(f, -1)
im = cv2.resize(im, None, 0, 0.25, 0.25, cv2.INTER_AREA)

rowims = []
colims = []
for i, c in enumerate(np.linspace(0, 2, 9)):
    if i == 0:
        rowims.append(im)
    else:
        rowims.append(equalizeBGR(im, clip=i, width=1))

    if i % 3 == 0:
        print(len(rowims))
        cv2.imshow('CLAHE', np.hstack(tuple(rowims)))
        colims.append(np.hstack(rowims))

result = np.vstack(colims)
cv2.imshow('CLAHE', result)


cv2.destroyAllWindows()

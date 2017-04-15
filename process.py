import os
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt


def generateHistogram(image, overall=False, plot=False):
    # build overall histogram
    if overall:
        gs = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        overall = cv2.calcHist([gs], [0], None, [256], [0, 256])

        if plot:
            plt.plot(overall, c='black')
            plt.xlim((0, 256))
            plt.ylim(0)
            plt.show()

        return overall

    # per-channel histograms
    else:
        bgr = np.zeros((256, 3))
        for i, color in enumerate('bgr'):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])[:, 0]
            bgr[:, i] = hist
            if plot:
                plt.plot(hist, c=color)

        if plot:
            plt.xlim((0, 256))
            plt.ylim(0)
            plt.show()

        return bgr


def equalizeLAB(image, clip=2.0, width=8):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    out = np.copy(image)
    for i in [0, 1, 2]:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(width, width))
        out[:, :, i] = clahe.apply(image[:, :, i])

    # return out
    out[:, :, 0] = image[:, :, 0]
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def equalizeBGR(image, clip=2.0, width=8):
    out = np.empty_like(image)
    for i in [0, 1, 2]:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(width, width))
        out[:, :, i] = clahe.apply(image[:, :, i])

    # return out
    return out


def equalizeLum(image, clip=2.0, width=8):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    out = np.copy(lab)

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(width, width))
    out[:, :, 0] = clahe.apply(lab[:, :, 0])

    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


if __name__ == '__main__':
    inExt = '.JPG'
    outExt = '.PNG'

    folders = ['Dives 1,2 - Apr 1 (Kahekili)',
               'Dives 3,4 - Apr 2 (Kahekili)',
               'Dive 5 - Apr 10 (Mala Pier)',
               'Dive 6 - Apr 11 (Mala Pier)'
               ]

    for folder in folders:
        print(folder)

        # get input files
        inDir = os.path.join('/home/sean/Pictures/', folder)
        # files = glob.glob(os.path.join(inDir, 'IMG_7135' + inExt))
        files = glob.glob(os.path.join(inDir, '*' + inExt))

        # setup output folder
        outF = os.path.join('/home/sean/Pictures/output', folder)
        if not os.path.exists(outF):
            os.mkdir(outF)

        for f in files:
            print('\t' + os.path.basename(f))

            # load image
            im = cv2.imread(f, -1)
            # generateHistogram(im, plot=True)

            # equalize bgr
            bgr = equalizeBGR(im, clip=1, width=1)


            # equalize lum
            lum = equalizeBGR(im, clip=1.5, width=1)
            # generateHistogram(lum, plot=True)

            res = np.hstack((im, lum))
            # plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

            # plt.show()

            cv2.imwrite(os.path.join(outF, os.path.splitext(os.path.basename(f))[0] + outExt), res)

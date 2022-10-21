import cv2
import numpy as np
import stats as stats
from matplotlib import pyplot as plt
import statistics

path = r'img.png'
img = cv2.imread("img.png")
img2= cv2.imread(path, 0)
def rgbimage():
    cv2.imshow('RGB', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def grayimage():
    img = cv2.imread(path, 0)
    cv2.imshow('gray', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.show()
def Hestogarmergb():
    img = cv2.imread('img.png',0)
    histr = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(histr)
    plt.show()

def Hestogarmergbgray():
    histr = cv2.calcHist([img2],[0],None,[256],[0,256])
    plt.plot(histr)
    plt.show()

def gammaCorrection(src, gamma):
    invGamma = gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)

def gamaimgprintrgb():
    gammaImg = gammaCorrection(img, 0.5)
    cv2.imshow('Gamma corrected image rgb', gammaImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gamaimgprintgray():
    gammaImg = gammaCorrection(img2, 0.5)
    cv2.imshow('Gamma corrected image gray', gammaImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def equlrgb():
    img = cv2.imread('img.png', 0)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(img2.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()
    img = cv2.imread('img.png', 0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))
    cv2.imwrite('resrgb.png', res)

def equlgray():
    hist, bins = np.histogram(img2.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(img2.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()
    img = cv2.imread('img.png', 0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))
    cv2.imwrite('resgray.png', res)

def equlrgbimage():
    img = cv2.imread("img.png",0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))
    cv2.imshow('image', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def equlgrayimage():
    equ = cv2.equalizeHist(img2)
    res = np.hstack((img2, equ))
    cv2.imshow('image', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def histoInfogray():
    im = img2.ravel()
    m = statistics.mean(im)
    m1 = statistics.mode(im)
    m2 = statistics.median(im)
    v=statistics.variance(im,0)
    print("Gray image ")
    print("The Mean = " + str(m))
    print("The Mode = " + str(m1))
    print("The Median = " + str(m2))
    print("The Variance = " + str(v))

def main():
    rgbimage()
    grayimage()
    Hestogarmergb()
    Hestogarmergbgray()
    gamaimgprintrgb()
    gamaimgprintgray()
    equlrgb()
    equlgray()
    equlrgbimage()
    equlgrayimage()
    histoInfogray()

if __name__ == "__main__":
    main()
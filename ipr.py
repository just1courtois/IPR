import numpy as np
from scipy import ndimage, misc, spatial
import matplotlib.pyplot as plt
import cv2


def loadImageInBinary(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    (thresh, image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # print(image)
    return image

def showImage(image):
    plt.imshow(image)
    plt.show()
image = loadImageInBinary("/home/justin/Downloads/simulation.bmp")
showImage(image)

def nopenings(n): #do n openings with an increasing binary structure
    results = []
    struct = ndimage.generate_binary_structure(2,1)
    for i in range(n):
        temp = ndimage.iterate_structure(struct, i)
        opening = ndimage.binary_opening(image, temp)
        results.append(opening)
    return results

def area(openings): #store the successive area
    results = []
    for i in range(len(openings)):
        temp = np.count_nonzero(openings[i])
        results.append(temp)
    return results



def hit_or_miss(image, imageC, T1, T2, show=0):
    eroded1 = ndimage.binary_erosion(image, T1)
    eroded2 = ndimage.binary_erosion(imageC, T2)
    intersection = merge(eroded1, eroded2)
    if show == 1:
        showImage(image)
        showImage(eroded1)
        showImage(eroded2)
        showImage(intersection)
    else:
        print("not whowing")
    return intersection

def merge(im1, im2):
    print(im1.shape)
    result = np.zeros(im1.shape).astype(bool)
    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            result[i, j] = im1[i, j] * im2[i, j]
    return result
    # print(result)


def merge2(im1, im2):
    result = np.minimum(im1, im2)
    return result

def random_binary_shape():
    A = np.zeros((20, 20)).astype('bool')
    A[3:14, 9:17] = True
    A[1:18, 11:16] = True
    return A

def external_boundary(dilated, binary_shape):
    contourExt = dilated ^ binary_shape
    #plt.imshow(contourExt)
    return contourExt

def iternal_boundary(eroded, binary_shape):
    countourInt = binary_shape ^ eroded
    plt.imshow(countourInt)

def freeman_chain(contourExt):
    points = np.argwhere(contourExt)
    firstPoint = points[0]
    previous = -1
    array = []
    x = firstPoint[0]
    y = firstPoint[1]
    if contourExt[x + 1, y] != 0:
        array.append(0)
        previous = 0  ##pour eviter de scaner le point en direction 0
        x += 1
    elif contourExt[x, y + 1] != 0:
        array.append(1)
        previous = 1
        y += 1
    elif contourExt[x - 1, y] != 0:
        array.append(2)
        previous = 2
        x -= 1
    else:
        array.append(3)
        previous = 3
        y -= 1
    i = 0
    while [x, y] != [firstPoint[0], firstPoint[1]]:
        i += 1
        print([x, y])
        if contourExt[
            x + 1, y] != 0 and previous != 2:  ##pour eviter de tomber sur un False ou sur le point précédent vu qu'on tourne dans le sens anti-horraire
            array.append(0)
            previous = 0
            x += 1
        elif contourExt[x, y + 1] != 0 and previous != 3:
            array.append(1)
            previous = 1
            y += 1
        elif contourExt[x - 1, y] != 0 and previous != 0:
            array.append(2)
            previous = 2
            x -= 1
        elif contourExt[x, y - 1] != 0 and previous != 1:
            array.append(3)
            previous = 3
            y -= 1
    print("chaine :")
    print(array)


def disk( radius ) :
    # defines a circular structuring element with radius given by ’ radius ’
    x = np.arange(- radius , radius +1, 1)
    xx, yy = np.meshgrid(x, x)
    d = np. sqrt ((xx ∗∗ 2) + (yy ∗∗ 2))
    return d<=radius;


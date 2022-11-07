import os

import numpy as np
import skimage
import matplotlib.pyplot as plt
import cv2

from skimage import io
from skimage import color
from matplotlib import pyplot as plt
from skimage import data
from skimage.color import rgb2hsv


import numpy as np
import glob

import skimage.io
import skimage.color
import skimage.filters




# # way to load car image from file
# filepath = "/Users/Eryk/Desktop/deski/2_proba\DSCF6344.JPG"
#
# photo = io.imread(filepath)
#
# # way to show the input image
# #io.imshow(photo)
# img = color.rgb2gray(photo)
# io.imshow(img)
# #io.imsave("skimage-greyscale.png",img)
# io.show()

# Converting RGB Image to HSV Image
# hsv_coffee = rgb2hsv(cars)
# io.imshow(hsv_coffee)
# io.show()

#img2 = cv2.imread('/Users/Eryk/Desktop/deski/2_proba\DSCF6344.JPG', 0)

# find frequency of pixels in range 0-255
# histr = cv2.calcHist([img], [0], None, [256], [0, 256])

# show the plotting graph of an image
# plt.plot(histr)
# plt.show()
#
# plt.hist(img.ravel(),256,[0,256])
# plt.show()


def slabo():

    filepath = "/Users/Eryk/Desktop/deski/2_proba\DSCF6344.JPG"

    image = io.imread(filepath)
    io.imshow(image)


    # convert the image to grayscale
    gray_image = skimage.color.rgb2gray(image)

    # blur the image to denoise
    blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)

    # show the histogram of the blurred image
    histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))
    fig, ax = plt.subplots()
    plt.plot(bin_edges[0:-1], histogram)
    plt.title("Graylevel histogram")
    plt.xlabel("gray value")
    plt.ylabel("pixel count")
    plt.xlim(0, 1.0)

    # perform automatic thresholding
    t = skimage.filters.threshold_otsu(blurred_image)
    print("Found automatic threshold t = {}.".format(t))

    # create a binary mask with the threshold found by Otsu's method
    binary_mask = blurred_image > t

    fig, ax = plt.subplots()
    plt.imshow(binary_mask, cmap="gray")

    # apply the binary mask to select the foreground
    selection = image.copy()
    selection[~binary_mask] = 0

    fig, ax = plt.subplots()
    plt.imshow(selection)

    io.show()


def kMeans():
    # Read in the image
    image_source = cv2.imread('/Users/Eryk/Desktop/deski/2_proba\DSCF6344.JPG')

    img_res = cv2.resize(image_source, (800, 600))

    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = img_res.reshape((-1, 3))

    # Convert to float type
    pixel_vals = np.float32(pixel_vals)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    # then perform k-means clustering wit h number of clusters defined as 3
    # also random centres are initially choosed for k-means clustering
    k = 3
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((img_res.shape))

    cv2.imshow('Resizing', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#kMeans()

def lalal():

    imgE = cv2.imread("/Users/Eryk/Desktop/deski/2_proba\DSCF6344.JPG", 0)
    h,w = imgE.shape[:12]


    new_h, new_w = int(h / 4), int(w / 4)
    resizeImg = cv2.resize(imgE, (new_w, new_h))

    (thresh, binary_image) = cv2.threshold(imgE, 100, 255, cv2.THRESH_BINARY)

    resizeImg2 = cv2.resize(binary_image, (new_w, new_h))

    #cv2.imshow('Original', imgE)
    cv2.imshow('Resizing', resizeImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Resizing', resizeImg2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Wmiaredziala():

    # Read image as grayscale
    img_gray =cv2.imread("/Users/Eryk/Desktop/deski/2_proba\DSCF6344.JPG", cv2.IMREAD_GRAYSCALE)

    # Read image
    img = cv2.imread("/Users/Eryk/Desktop/deski/2_proba\DSCF6344.JPG")



    # Resize image

    img_res= cv2.resize(img, (800,600))

    # Make a copy
    new_image = img_res.copy()

    cv2.imshow('Gray image', img_res)
    cv2.waitKey(0) # Wait for keypress to continue
    cv2.destroyAllWindows() # Close windows

    # Convert to grayscale

    img_to_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

    # Display the grayscale image
    cv2.imshow('Gray image', img_to_gray)
    cv2.waitKey(0) # Wait for keypress to continue
    cv2.destroyAllWindows() # Close windows

    # Convert to binary

    #(thresh, binary) =cv2.threshold(img_to_gray,80,255,cv2.THRESH_BINARY)  #simple threshold
    #(thresh, binary) =cv2.threshold(img_to_gray,0,255,cv2.THRESH_OTSU)     #otsu1
    binary =cv2.adaptiveThreshold(img_to_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 195, 50)     #adaptive1
    #binary =cv2.adaptiveThreshold(img_to_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 195, 40)     #adaptive1
    #(thresh, binary) =cv2.threshold(img_to_gray,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #otsu2


    # Display the binary image
    cv2.imshow('Binary image', binary)
    cv2.waitKey(0) # Wait for keypress to continue
    cv2.destroyAllWindows() # Close windows


    # apply morphology
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)


    cv2.imshow("morph", morph)
    cv2.waitKey(0)  # Wait for keypress to continue
    cv2.destroyAllWindows()  # Close windows


    # To detect object contours, we want a black background and a white object

    inverted_binary = ~binary
    cv2.imshow('Inverted binary image', inverted_binary)
    cv2.waitKey(0) # Wait for keypress to continue
    cv2.destroyAllWindows() # Close windows

    # Find contours

    contours, hierarchy = cv2.findContours(inverted_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original

    with_contours = cv2.drawContours(img_res, contours, -1,(255,0,255),2)
    cv2.imshow('Detected contours', with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




    # Show the total number of contours that were detected
    print('Total number of contours detected: ' + str(len(contours)))

    # Draw just the first contour
    # The 0 means to draw the first contour
    first_contour = cv2.drawContours(new_image, contours, 0, (255, 0, 255), 3)
    cv2.imshow('First detected contour', first_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Draw a bounding box around the first contour
    # x is the starting x coordinate of the bounding box
    # y is the starting y coordinate of the bounding box
    # w is the width of the bounding box
    # h is the height of the bounding box
    x, y, w, h = cv2.boundingRect(contours[0])
    cv2.rectangle(first_contour, (x, y), (x + w, y + h), (255, 0, 0), 1)
    cv2.imshow('First contour with bounding box', first_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Draw a bounding box around all contours
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Make sure contour area is large enough
        if (cv2.contourArea(c)) > 5:
            cv2.rectangle(with_contours, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cv2.imshow('All contours with bounding box', with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




#img33= cv2.resize(img3, (800,600))
#(thresh, binary33) =cv2.threshold(img33,120,255,cv2.THRESH_BINARY)

# contours, hierarchy = cv2.findContours(binary33, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# imageNew = cv2.drawContours(binary33, contours, -1, (0,255,0),2)


# plt.imshow(imageNew)
# plt.show()


#cv2.imshow('Resizing', binary33)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print("Shape of the loaded image is", img33.shape)

Wmiaredziala()
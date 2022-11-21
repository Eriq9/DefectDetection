import math
import os
from turtle import pd

import numpy as np
import skimage
import matplotlib.pyplot as plt
import cv2
import argparse

from skimage import io
from skimage import color
from matplotlib import pyplot as plt
from skimage import data
from skimage.color import rgb2hsv

import numpy as np
import skimage.io
import skimage.color
import skimage.filters
import mediapipe as mp

from skimage.measure import label, regionprops, regionprops_table


# def histogram():
#
#     filepath = "/Users/Eryk/Desktop/deski/2_proba\DSCF6344.JPG"
#
#     image = io.imread(filepath)
#     io.imshow(image)
#
#
#     # convert the image to grayscale
#     gray_image = skimage.color.rgb2gray(image)
#
#     # blur the image to denoise
#     blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)
#
#     # show the histogram of the blurred image
#     histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))
#     fig, ax = plt.subplots()
#     plt.plot(bin_edges[0:-1], histogram)
#     plt.title("Graylevel histogram")
#     plt.xlabel("gray value")
#     plt.ylabel("pixel count")
#     plt.xlim(0, 1.0)
#
#     # perform automatic thresholding
#     t = skimage.filters.threshold_otsu(blurred_image)
#     print("Found automatic threshold t = {}.".format(t))
#
#     # create a binary mask with the threshold found by Otsu's method
#     binary_mask = blurred_image > t
#
#     fig, ax = plt.subplots()
#     plt.imshow(binary_mask, cmap="gray")
#
#     # apply the binary mask to select the foreground
#     selection = image.copy()
#     selection[~binary_mask] = 0
#
#     fig, ax = plt.subplots()
#     plt.imshow(selection)
#
#     io.show()

#
# def kMeans():
#     # Read in the image
#     image_source = cv2.imread("/Users/Eryk/Desktop/deskinowe\DSCF6497.JPG")
#
#     img_res = cv2.resize(image_source, (800, 600))
#
#
#     # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
#     # pixel_vals = img_res.reshape((-1, 3))
#     #
#     # # Convert to float type
#     # pixel_vals = np.float32(pixel_vals)
#     #
#     # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
#     #
#     # # then perform k-means clustering wit h number of clusters defined as 3
#     # # also random centres are initially choosed for k-means clustering
#     # k = 5
#     # retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#     #
#     # # convert data into 8-bit values
#     # centers = np.uint8(centers)
#     # segmented_data = centers[labels.flatten()]
#     #
#     # # reshape data into the original image dimensions
#     # segmented_image = segmented_data.reshape((img_res.shape))
#
#     # Convert to grayscale
#
#     img_to_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
#
#     cv2.imshow("morph-open", img_to_gray)
#     cv2.waitKey(0)  # Wait for keypress to continue
#     cv2.destroyAllWindows()  # Close windows
#
#     (thresh, binary) =cv2.threshold(img_to_gray,150,255,cv2.THRESH_BINARY)  #simple threshold
#
#     # defining the kernel matrix
#     kernel_open = np.ones((20, 20), np.uint8)
#     # using morphologyEx function by specifying the MORPH_OPEN operation on the input image
#     openimage = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
#
#     cv2.imshow("morph-open", openimage)
#     cv2.waitKey(0)  # Wait for keypress to continue
#     cv2.destroyAllWindows()  # Close windows
#
#     # apply morphology close
#     kernel_close = np.ones((2, 2), np.uint8)
#     # using morphologyEx function by specifying the MORPH_CLOSE operation on the input image
#     closingimage = cv2.morphologyEx(openimage, cv2.MORPH_CLOSE, kernel_close)
#     # displaying the morphed image as the output on the screen
#     cv2.imshow('morph-close', closingimage)
#     cv2.waitKey(0)




    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # Draw contours on original
    #
    # with_contours = cv2.drawContours(img_res, contours, -1, (255, 0, 255), 2)
    # cv2.imshow('Detected contours', with_contours)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




#kMeans()

#
# def tlo():
#     # Read in the image
#     image_source = cv2.imread("/Users/Eryk/Desktop/deskinowe\DSCF6497.JPG")
#
#     img_res = cv2.resize(image_source, (800, 600))
#
#     # threshold on white
#     # Define lower and uppper limits
#     lower = np.array([150, 150, 150])
#     upper = np.array([255, 255, 255])
#
#     # Create mask to only select black
#     thresh = cv2.inRange(img_res, lower, upper)
#
#     cv2.imshow('Gray image', thresh)
#     cv2.waitKey(0)  # Wait for keypress to continue
#     cv2.destroyAllWindows()  # Close windows
#
#     # apply morphology
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#
#     cv2.imshow('Gray image', morph)
#     cv2.waitKey(0)  # Wait for keypress to continue
#     cv2.destroyAllWindows()  # Close windows
#
#     # apply morphology
#     # Close contour
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
#     close = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
#
#     cv2.imshow('Gray image', close)
#     cv2.waitKey(0)  # Wait for keypress to continue
#     cv2.destroyAllWindows()  # Close windows
#
#     inverted_binary = ~close
#
#     # Find outer contour and fill with white
#     cnts = cv2.findContours(inverted_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     cv2.fillPoly(inverted_binary, cnts, [255, 255, 255])
#
#     cv2.imshow('Gray image', inverted_binary)
#     cv2.waitKey(0)  # Wait for keypress to continue
#     cv2.destroyAllWindows()  # Close windows
#
#     # Apply the Component analysis function
#     analysis = cv2.connectedComponentsWithStats(inverted_binary,
#                                                 4,
#                                                 cv2.CV_32S)
#     (totalLabels, label_ids, values, centroid) = analysis
#
#     for i in range(1, totalLabels):
#         area = values[i, cv2.CC_STAT_AREA]
#         print("area:",area)



# def tlo2():
#     # To detect object contours, we want a black background and a white object
#
#     image_source = cv2.imread("/Users/Eryk/Desktop/deskinowe\prosze.JPG")
#
#     inverted_binary = ~image_source
#     cv2.imshow('Inverted binary image', inverted_binary)
#     cv2.waitKey(0)  # Wait for keypress to continue
#     cv2.destroyAllWindows()  # Close windows
#
#     # flood fill background to find inner holes
#     holes = inverted_binary.copy()
#     cv2.floodFill(holes, None, (150, 150), 255)
#
#     # invert holes mask, bitwise or with img fill in holes
#     holes = cv2.bitwise_not(holes)
#     filled_holes = cv2.bitwise_or(image_source, holes)
#     cv2.imshow('', filled_holes)
#     cv2.waitKey()
#
# #tlo2()


def Wmiaredziala():

    # Read image as grayscale
    img_gray =cv2.imread("/Users/Eryk/Desktop/deski/2_proba\DSCF6344.JPG", cv2.IMREAD_GRAYSCALE)

    # Read image
    #img = cv2.imread("/Users/Eryk/Desktop/deski/2_proba\DSCF6344.JPG")
    #img = cv2.imread("/Users/Eryk/Desktop/deski2\DSCF6409.JPG")
    img = cv2.imread("/Users/Eryk/Desktop/deseczki\DSCF6558.JPG")



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

    #(thresh, binary) =cv2.threshold(img_to_gray,200,255,cv2.THRESH_BINARY)  #simple threshold
    #(thresh, binary) =cv2.threshold(img_to_gray,0,255,cv2.THRESH_OTSU)     #otsu1
    binary =cv2.adaptiveThreshold(img_to_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 199, 70)     #adaptive1
    #binary =cv2.adaptiveThreshold(img_to_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 195, 10)     #adaptive1
    #(thresh, binary) =cv2.threshold(img_to_gray,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #otsu2


    # Display the binary image
    cv2.imshow('Binary image', binary)
    cv2.waitKey(0) # Wait for keypress to continue
    cv2.destroyAllWindows() # Close windows


    # apply morphology open


    # defining the kernel matrix
    kernel_open = np.ones((8, 8), np.uint8)
    # using morphologyEx function by specifying the MORPH_OPEN operation on the input image
    openimage = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)


    cv2.imshow("morph-open", openimage)
    cv2.waitKey(0)  # Wait for keypress to continue
    cv2.destroyAllWindows()  # Close windows

    # apply morphology close
    kernel_close = np.ones((2, 2), np.uint8)
    # using morphologyEx function by specifying the MORPH_CLOSE operation on the input image
    closingimage = cv2.morphologyEx(openimage, cv2.MORPH_CLOSE, kernel_close)
    # displaying the morphed image as the output on the screen
    cv2.imshow('morph-close', closingimage)
    cv2.waitKey(0)


    # To detect object contours, we want a black background and a white object

    inverted_binary = ~closingimage
    cv2.imshow('Inverted binary image', inverted_binary)
    cv2.waitKey(0) # Wait for keypress to continue
    cv2.destroyAllWindows() # Close windows


    # Apply the Component analysis function
    analysis = cv2.connectedComponentsWithStats(inverted_binary,
                                                4,
                                                cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    # Initialize a new image to
    # store all the output components
    output = np.zeros(img_to_gray.shape, dtype="uint8")

    # Loop through each component
    for i in range(1, totalLabels):
        area = values[i, cv2.CC_STAT_AREA]

        if area > 1500:

            componentMask = (label_ids == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask)
            #print(componentMask)
        print("area:",area)

    # Creating the Final output mask


    cv2.imshow("Filtered Components", output)
    cv2.waitKey(0)


    #substract

    substract_img = cv2.subtract(inverted_binary, output)

    cv2.imshow("Substract Components", substract_img)
    cv2.waitKey(0)

    # Find contours

    contours, hierarchy = cv2.findContours(substract_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    # Draw contours on original

    with_contours = cv2.drawContours(img_res, contours, -1,(255,0,255),2)
    cv2.imshow('Detected contours', with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ###############################

    regions = regionprops(inverted_binary)

    print(len(regions))




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


#Wmiaredziala()

class ImageProcessingAlgorithms:
    def __init__(self, ImagePath):
        self.random_image_path = ImagePath


    def ImageProcess(self):
        # Read image

        global DefectAreaSum


        img = cv2.imread(str(self.random_image_path))
        #img = cv2.imread("/Users/Eryk/Desktop/deski/2_proba\DSCF6344.JPG")

        # Resize image

        img_res = cv2.resize(img, (800, 600))

        # Make a copy
        new_image = img_res.copy()

        # Convert to grayscale

        img_to_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

        # Convert to binary

        binary = cv2.adaptiveThreshold(img_to_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 70)  # adaptive1

        # apply morphology open
        # defining the kernel matrix

        kernel_open = np.ones((8, 8), np.uint8)
        openimage = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        # apply morphology close

        kernel_close = np.ones((2, 2), np.uint8)

        closingimage = cv2.morphologyEx(openimage, cv2.MORPH_CLOSE, kernel_close)

        # To detect object contours, we want a black background and a white object

        inverted_binary = ~closingimage

        ############################## Znajdowanie konturów #####################################


        area_list = []
        width_list = []
        id_list = []

        # Apply the Component analysis function
        analysis = cv2.connectedComponentsWithStats(inverted_binary, 4, cv2.CV_32S)
        (totalLabels, label_ids, values, centroid) = analysis


        # Initialize a new image to store output components

        output = np.zeros(img_to_gray.shape, dtype="uint8")

        # Loop through each component to store a data
        for i in range(1, totalLabels):
            area = values[i, cv2.CC_STAT_AREA]
            area_list.append(area)

            width = values[i, cv2.CC_STAT_WIDTH]
            width_list.append(width)

            height = values[i, cv2.CC_STAT_HEIGHT]

            id = i
            id_list.append(id)

            print("id " + str(id) + " " + "area: " + str(area) + " " + "width: " + str(width) + " " + "height: " + str(height))

        max_area = max(area_list)
        max_width = max(width_list)

        max_area_index = area_list.index(max_area)
        max_width_index = width_list.index(max_width)

        print(max_area)
        print(max_area_index)
        print(max_width)
        print(max_width_index)

        # Loop through each component to filter
        for i in range(1, totalLabels):

            if width_list[i - 1] == max_width:
                componentMask = (label_ids == max_width_index + 1).astype("uint8") * 255
                # Creating the Final output mask
                output = cv2.bitwise_or(output, componentMask)
                # print(componentMask)


        # substract

        substract_img = cv2.subtract(inverted_binary, output)

        # Find contours

        contours, hierarchy = cv2.findContours(substract_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print('Total number of contours detected: ' + str(len(contours)))

        # Draw contours on original

        with_contours = cv2.drawContours(img_res, contours, -1, (255, 0, 255), 2)

        # Draw a bounding box around contours
        # x is the starting x coordinate of the bounding box
        # y is the starting y coordinate of the bounding box
        # w is the width of the bounding box
        # h is the height of the bounding box

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            # area = cv2.contourArea(contours)
            # print(area)

            # Make sure contour area is large enough
            if (cv2.contourArea(c)) > 5:
                cv2.rectangle(with_contours, (x, y), (x + w, y + h), (255, 0, 0), 1)

        cv2.imshow('All contours with bounding box', with_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        ############################################# Liczenie powierzchni ############################################

        # Count detects area

        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])

            # Count detects area, trzeba znaleźć jakoś pole całości i wyliczyć jak procentowo się ma suma do całości

        DefectAreaList = []
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            DefectAreaList.append(area)
            print("Detects area", area)

        DefectAreaSum = sum(DefectAreaList)
        DefectCounters = str(len(contours))
        print("Sum", DefectAreaSum)

        ########################################## Liczenie powierzchni całej deski #######################################


        # Make a copy
        CountAreaImage = img_res.copy()

        # threshold on white
        # Define lower and uppper limits
        lower = np.array([150, 150, 150])
        upper = np.array([255, 255, 255])

        # Create mask
        CountAreaThresh = cv2.inRange(CountAreaImage, lower, upper)

        # apply morphology
        kernelCount = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morphCount = cv2.morphologyEx(CountAreaThresh, cv2.MORPH_CLOSE, kernelCount)

        # apply morphology
        # Close contour
        kernelCount2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closeCount = cv2.morphologyEx(morphCount, cv2.MORPH_CLOSE, kernelCount2)

        inverted_binary_count = ~closeCount

        # Find outer contour and fill with white
        cnts = cv2.findContours(inverted_binary_count, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cv2.fillPoly(inverted_binary_count, cnts, [255, 255, 255])

        # Apply the Component analysis function
        analysisCount = cv2.connectedComponentsWithStats(inverted_binary_count, 4, cv2.CV_32S)
        (totalLabels, label_ids, values, centroid) = analysisCount

        for i in range(1, totalLabels):
            areaCount = values[i, cv2.CC_STAT_AREA]
        print("Full area:", areaCount)

        FailurePercentage = (DefectAreaSum / area) * 100

        #############################################################################################################################

        return with_contours, DefectAreaSum, DefectCounters, FailurePercentage


    # def CountPlankArea(self):
    #
    #     # Read in the image
    #     img = cv2.imread(str(self.random_image_path))
    #
    #     img_res = cv2.resize(img, (800, 600))
    #
    #     # threshold on white
    #     # Define lower and uppper limits
    #     lower = np.array([150, 150, 150])
    #     upper = np.array([255, 255, 255])
    #
    #     # Create mask to only select black
    #     thresh = cv2.inRange(img_res, lower, upper)
    #
    #     # apply morphology
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #     morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #
    #     # apply morphology
    #     # Close contour
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    #     close = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    #
    #     inverted_binary = ~close
    #
    #     # Find outer contour and fill with white
    #     cnts = cv2.findContours(inverted_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #     cv2.fillPoly(inverted_binary, cnts, [255, 255, 255])
    #
    #     # Apply the Component analysis function
    #     analysis = cv2.connectedComponentsWithStats(inverted_binary,4,cv2.CV_32S)
    #     (totalLabels, label_ids, values, centroid) = analysis
    #
    #     for i in range(1, totalLabels):
    #         area = values[i, cv2.CC_STAT_AREA]
    #     print("area:", area)
    #
    #     FailurePercentage = (DefectAreaSum/area)*100
    #
    #     return FailurePercentage



#
# def ImageProcessLast():
#     # Read image
#
#     img = cv2.imread("/Users/Eryk/Desktop/deski/2_proba\DSCF6344.JPG")
#
#     # Resize image
#
#     img_res = cv2.resize(img, (800, 600))
#
#     # Make a copy
#     new_image = img_res.copy()
#
#     # Convert to grayscale
#
#     img_to_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
#
#     # Convert to binary
#
#     binary = cv2.adaptiveThreshold(img_to_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 45)  # adaptive1
#
#     # apply morphology open
#     # defining the kernel matrix
#
#     kernel = np.ones((4, 4), np.uint8)
#     openimage = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
#
#     # apply morphology close
#
#     closingimage = cv2.morphologyEx(openimage, cv2.MORPH_CLOSE, kernel)
#
#     # To detect object contours, we want a black background and a white object
#
#     inverted_binary = ~closingimage
#
#     # Apply the Component analysis function
#     analysis = cv2.connectedComponentsWithStats(inverted_binary, 4, cv2.CV_32S)
#     (totalLabels, label_ids, values, centroid) = analysis
#
#     # Initialize a new image to store output components
#
#     output = np.zeros(img_to_gray.shape, dtype="uint8")
#
#     # Loop through each component
#     for i in range(1, totalLabels):
#         area = values[i, cv2.CC_STAT_AREA] # tu można dorzucić jakiegoś ifa, który jeśli area coś to nie przepuszcza, ale nie do końca rozumiem czym jest ta area
#         if area > 2000:
#
#             componentMask = (label_ids == i).astype("uint8") * 255
#             # Creating the Final output mask
#             output = cv2.bitwise_or(output, componentMask)
#             # print(componentMask)
#
#     # substract
#
#     substract_img = cv2.subtract(inverted_binary, output)
#
#     # Find contours
#
#     contours, hierarchy = cv2.findContours(substract_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     #print('Total number of contours detected: ' + str(len(contours)))
#
#     # Draw contours on original
#
#     with_contours = cv2.drawContours(img_res, contours, -1, (255, 0, 255), 2)
#
#     # Draw a bounding box around contours
#     # x is the starting x coordinate of the bounding box
#     # y is the starting y coordinate of the bounding box
#     # w is the width of the bounding box
#     # h is the height of the bounding box
#
#     for c in contours:
#         x, y, w, h = cv2.boundingRect(c)
#
#         # Make sure contour area is large enough
#         if (cv2.contourArea(c)) > 5:
#             cv2.rectangle(with_contours, (x, y), (x + w, y + h), (255, 0, 0), 1)
#
#
#
#     # Count detects area, trzeba znaleźć jakoś pole całości i wyliczyć jak procentowo się ma suma do całości
#
#     DefectAreaList = []
#     for i in range(len(contours)):
#         area = cv2.contourArea(contours[i])
#         DefectAreaList.append(area)
#         print("Detects area", area)
#
#     DefectAreaSum = sum(DefectAreaList)
#     print("Sum", DefectAreaSum)
#
#
#
#
#
#     #return with_contours
#     cv2.imshow('All contours with bounding box', with_contours)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# #ImageProcessLast()

def Test():

    # Read image as grayscale
    img_gray =cv2.imread("/Users/Eryk/Desktop/deski/2_proba\DSCF6344.JPG", cv2.IMREAD_GRAYSCALE)

    # Read image
    #img = cv2.imread("/Users/Eryk/Desktop/deski/2_proba\DSCF6344.JPG")
    #img = cv2.imread("/Users/Eryk/Desktop/deski2\DSCF6409.JPG")
    img = cv2.imread("/Users/Eryk/Desktop/deseczki\DSCF6558.JPG")



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

    #(thresh, binary) =cv2.threshold(img_to_gray,200,255,cv2.THRESH_BINARY)  #simple threshold
    #(thresh, binary) =cv2.threshold(img_to_gray,0,255,cv2.THRESH_OTSU)     #otsu1
    binary =cv2.adaptiveThreshold(img_to_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 199, 70)     #adaptive1
    #binary =cv2.adaptiveThreshold(img_to_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 195, 10)     #adaptive1
    #(thresh, binary) =cv2.threshold(img_to_gray,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #otsu2


    # Display the binary image
    cv2.imshow('Binary image', binary)
    cv2.waitKey(0) # Wait for keypress to continue
    cv2.destroyAllWindows() # Close windows


    # apply morphology open


    # defining the kernel matrix
    kernel_open = np.ones((8, 8), np.uint8)
    # using morphologyEx function by specifying the MORPH_OPEN operation on the input image
    openimage = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)


    cv2.imshow("morph-open", openimage)
    cv2.waitKey(0)  # Wait for keypress to continue
    cv2.destroyAllWindows()  # Close windows

    # apply morphology close
    kernel_close = np.ones((2, 2), np.uint8)
    # using morphologyEx function by specifying the MORPH_CLOSE operation on the input image
    closingimage = cv2.morphologyEx(openimage, cv2.MORPH_CLOSE, kernel_close)
    # displaying the morphed image as the output on the screen
    cv2.imshow('morph-close', closingimage)
    cv2.waitKey(0)


    # To detect object contours, we want a black background and a white object

    inverted_binary = ~closingimage
    cv2.imshow('Inverted binary image', inverted_binary)
    cv2.waitKey(0) # Wait for keypress to continue
    cv2.destroyAllWindows() # Close windows

    # label_img = label(inverted_binary)
    # regions = regionprops(label_img)
    #
    # props = regionprops_table(label_img, properties=('centroid',
    #                                                  'orientation',
    #                                                  'axis_major_length',
    #                                                  'axis_minor_length',
    #                                                  'bbox'))
    # print(props)
    #
    # fig, ax = plt.subplots()
    #
    # for props in regions:
    #     y0, x0 = props.centroid
    #     orientation = props.orientation
    #     x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
    #     y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
    #     x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
    #     y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length
    #
    #     ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    #     ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    #     ax.plot(x0, y0, '.g', markersize=15)
    #
    #     minr, minc, maxr, maxc = props.bbox
    #     bx = (minc, maxc, maxc, minc, minc)
    #     by = (minr, minr, maxr, maxr, minr)
    #     ax.plot(bx, by, '-b', linewidth=2.5)
    #
    # ax.axis((0, 600, 600, 0))
    # plt.show()

    # Apply the Component analysis function
    analysis = cv2.connectedComponentsWithStats(inverted_binary,
                                                4,
                                                cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    area_list = []
    width_list = []
    id_list = []


    # Initialize a new image to
    # store all the output components
    output = np.zeros(img_to_gray.shape, dtype="uint8")

    # Loop through each component to store a data
    for i in range(1, totalLabels):
        area = values[i, cv2.CC_STAT_AREA]
        area_list.append(area)

        width = values[i, cv2.CC_STAT_WIDTH]
        width_list.append(width)

        height = values[i, cv2.CC_STAT_HEIGHT]

        id = i
        id_list.append(id)

        print("id " + str(id) + " " + "area: " + str(area) + " " +  "width: " + str(width) + " " + "height: " + str(height))


    max_area = max(area_list)
    max_width = max(width_list)

    max_area_index = area_list.index(max_area)
    max_width_index = width_list.index(max_width)

    print(max_area)
    print(max_area_index)
    print(max_width)
    print(max_width_index)

    # Loop through each component to filter
    for i in range(1, totalLabels):

        if width_list[i-1] == max_width:

            componentMask = (label_ids == max_width_index + 1).astype("uint8") * 255
            # Creating the Final output mask
            output = cv2.bitwise_or(output, componentMask)
            # print(componentMask)

    cv2.imshow("Filtered Components", output)
    cv2.waitKey(0)

    # substract

    substract_img = cv2.subtract(inverted_binary, output)

    # Find contours

    contours, hierarchy = cv2.findContours(substract_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print('Total number of contours detected: ' + str(len(contours)))

    # Draw contours on original

    with_contours = cv2.drawContours(img_res, contours, -1, (255, 0, 255), 2)

    # Draw a bounding box around contours
    # x is the starting x coordinate of the bounding box
    # y is the starting y coordinate of the bounding box
    # w is the width of the bounding box
    # h is the height of the bounding box

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Make sure contour area is large enough
        if (cv2.contourArea(c)) > 5:
            cv2.rectangle(with_contours, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cv2.imshow('All contours with bounding box', with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




#Test()
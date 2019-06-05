import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def sobel_edge_detect(img,kernel_size):
    """
    This function applies a sobel edge detection algorithm
    on the image to find edges
    Inputs: Undistorted Image
    Output: Sobel filtered edge image
    """

    # Convert to a gray scale image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Calculate the sobel magnitudes
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    scaled_sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    scaled_sobel = np.uint8(scaled_sobel / np.max(scaled_sobel) * 255)

    # Apply the thresholding and create the binary image
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output

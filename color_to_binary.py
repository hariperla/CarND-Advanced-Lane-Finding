import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

def hls_select(img,thrsh=(120,255)):
    """
    This function returns the binary output after 
    applying a color thresholding
    Inputs: RGB image
    Output: binary image
    """

    # Convert the image to hls
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    binary_output = np.zeros_like(s_channel)

    binary_output[(s_channel > thrsh[0]) & (s_channel <= thrsh[1])] = 1

    return binary_output

def sobel_edge_detect(img,kernel_size):
    """
    This function applies a sobel edge detection algorithm
    on the image to find edges
    Inputs: Undistorted Image
    Output: Sobel filtered edge image
    """

    # Convert to a gray scale image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh_min = 10
    thresh_max = 100

    # Calculate the sobel magnitudes
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize = kernel_size)
    abs_sobelx = np.absolute(sobel_x)
    scaled_sobel = np.uint8(abs_sobelx * 255 / np.max(abs_sobelx))

    # Apply the thresholding and create the binary image
    edge_output = np.zeros_like(scaled_sobel)
    edge_output[(scaled_sobel > thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return edge_output


if __name__ == "__main__":

    # Read the test image
    #test_img = mpimg.imread("output_images/test_calibration_after.jpg") 
    test_img = cv2.imread("output_images/test_calibration_after.jpg")
    sbinary = hls_select(test_img,thrsh=(100,255))
    sxbinary = sobel_edge_detect(test_img,3)

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(sbinary == 1) & (sxbinary == 1)] = 1

    cv2.imshow("binary image",sbinary)
    cv2.waitKey(5000)
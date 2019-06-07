import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

def thresholding(img,color_thrsh=(120,255),gradient_thrsh=(30,100)):
    """
    This function returns the binary output after 
    applying a color thresholding and gradient thresholding
    using a sobel filter
    Inputs: RGB image
    Output: binary image
    """

    # Convert the image to hls
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Calculate the sobel magnitudes
    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0,ksize = kernel_size)
    abs_sobelx = np.absolute(sobel_x)
    scaled_sobelx = np.uint8(abs_sobelx * 255 / np.max(abs_sobelx))

    # Color thresholding
    color_binary = np.zeros_like(s_channel)
    color_binary[(s_channel > color_thrsh[0]) & (s_channel <= color_thrsh[1])] = 1

    # Gradient thresholding
    gradient_binary = np.zeros_like(scaled_sobelx)
    gradient_binary[(scaled_sobelx > gradient_thrsh[0]) & (gradient_thrsh <= thrsh[1])] = 1

    # Combined threshold
    comb_thrsh_binary = np.zeros_like(gradient_binary)
    # If any of the pixels in the color threshold image or the gradient image are
    # true apply it to the combined binary image
    comb_thrsh_binary[(color_binary == 1) | (gradient_binary ==1)] = 1

    return comb_thrsh_binary

if __name__ == "__main__":

    # Read the test image
    test_img = cv2.imread("output_images/test_calibration_after.jpg")
    binary_output = thresholding(test_img,color_thrsh=(120,255),gradient_thrsh=(30,100))

    cv2.imshow("binary image",binary_output)
    cv2.waitKey(5000)
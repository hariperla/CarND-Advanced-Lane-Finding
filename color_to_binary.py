import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

def thresholding(img,kernel_size,color_thrsh=(120,255),gradient_thrsh=(10,100)):
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

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Calculate the sobel magnitudes
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize = kernel_size)
    abs_sobelx = np.absolute(sobel_x)
    scaled_sobelx = np.uint8(255*abs_sobelx / np.max(abs_sobelx))

    # Color thresholding
    color_binary = np.zeros_like(s_channel)
    color_binary[(s_channel > color_thrsh[0]) & (s_channel <= color_thrsh[1])] = 1

    # Gradient thresholding
    gradient_binary = np.zeros_like(scaled_sobelx)
    gradient_binary[(scaled_sobelx > gradient_thrsh[0]) & (scaled_sobelx <= gradient_thrsh[1])] = 1
    cv2.imshow("binary image",gradient_binary)
    cv2.waitKey(5000)

    # Combined threshold
    comb_thrsh_binary = np.zeros_like(gradient_binary)
    # If any of the pixels in the color threshold image or the gradient image are
    # true apply it to the combined binary image
    comb_thrsh_binary[(color_binary == 1) | (gradient_binary ==1)] = 1

    return color_binary

if __name__ == "__main__":

    # Read the test image
    test_img = cv2.imread("output_images/test_calibration_after.jpg")
    binary_output = thresholding(test_img,5,color_thrsh=(110,255),gradient_thrsh=(30,100))

    # Save your files to output folder - astype uint8 converts it to 8 bit unsigned integer, 
    # which gives 0 for False and 1 for True, and multiplies it by 255 to make a (bit-)mask before writing it.
    cv2.imwrite("output_images/test_after_binarization.jpg", binary_output.astype('uint8') * 255)
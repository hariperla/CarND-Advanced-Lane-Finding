import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def cam_calibrate():
    """
    This function extracts the camera matrix and distortion 
    coefficients from a set of camera images
    Inputs: Folder containing set of camera images
    Output: Distortion coefficient and camera matrix
    """

    # prepare object points
    # Number of inside corners for x
    nx = 9 
    # Number of inside corners for y
    ny = 6 

    # Read the test images folder
    chessBoardImages = os.listdir("camera_cal/")

    # Setup arrays to store object points and image points
    objpoints = [] # 3D points in real world space
    imgp = [] # 2D points in image plane

    # Create the object points now
    objp = np.zeros((6*9,3),np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    for i in chessBoardImages:
        i = "camera_cal/" + i
        img = cv2.imread(i)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        corners_found, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, Draw and display the corners
        if corners_found is True:
            # Update the image points with the corners found
            imgp.append(corners)
            #Update the object points
            objpoints.append(objp)
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, corners_found)

    corners_found, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgp, gray.shape[::-1], None, None)

    return corners_found,mtx,dist,rvecs,tvecs

def img_undistort(img, mtx, dist):
    """
    Undistort an image considering the distortion coefficients and
    camera matrix
    Inputs: Image you want undistorted
    Output: Undistorted image
    """
    return cv2.undistort(test_img, mtx, dist, None, mtx)

if __name__ == "__main__":

    # Function call to camera calibrate function to obtain 
    # the distortion coefficients and the camera matrix
    corners_found,mtx,dist,rvecs,tvecs = cam_calibrate()

    # Function call to undistort an image
    test_img = cv2.imread("test_images/straight_lines1.jpg")
    undist_img = img_undistort(test_img,mtx,dist)

    # Save your files to output folder
    cv2.imwrite("output_images/test_calibration_before.jpg", test_img)
    cv2.imwrite("output_images/test_calibration_after.jpg", undist_img)

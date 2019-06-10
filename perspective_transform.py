import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def perspective_transform(img):
    """
    This function defines how to transform an image
    into a top view or a bird eye view to help 
    find the lanes better
    Inputs:Undistorted binary image
    Output:Transformed image
    """
    # Convert the image to grayscale and the get the dimensions of it
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_size = (gray.shape[1],gray.shape[0])

    # Define the source points for the perspective transform
    src_pts = np.float32([[690,450],[1110,img_size[1]],[175,img_size[1]],[595,450]])

    # Define an offset for the destination points
    ofst = 300
    dst_pts = np.float32([[img_size[0]-ofst, 0],[img_size[0]-ofst, img_size[1]],
                      [ofst, img_size[1]],[ofst, 0]])

    # Calculate the transform matrix
    M = cv2.getPerspectiveTransform(src_pts,dst_pts)

    # Apply the transformation matrix on the image to warp it
    transformed_image = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)

    return transformed_image,M

if __name__ == "__main__":
    
    # Read the image
    undist_binary_img = cv2.imread("output_images/test_after_binarization.jpg")

    warped_image, perspective_matrix = perspective_transform(undist_binary_img)
    cv2.imwrite("output_images/test_after_transform.jpg", warped_image)

    # Display the warped image
    cv2.imshow("Warped Image",warped_image)
    cv2.waitKey(5000)

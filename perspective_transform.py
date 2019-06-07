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
    ofst = 200
    dst_pts = np.float32([[img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]],
                      [offset, img_size[1]],[offset, 0]])

    # Calculate the transform matrix
    M = cv2.getPerspectiveTransform(src_pts,dst_pts)

    # Apply the transformation matrix on the image to warp it
    transformed_image = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)

    return transformed_image,M

    if __name == "__main__":
        
        # Read the image - Replace it with the color/gradient threshold image
        undist_binary_img = cv2.imread("output_images/test_calibration_after.jpg")

        warped_image, perspective_matrix = perspective_transform(undist_binary_img)

        # Display the warped image
        cv2.imshow(warped_image)
        cv2.waitKey(5000)

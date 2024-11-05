import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display an image in Jupyter Notebook
def display_image(image, title="Image", is_color=False):
    if is_color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper display
    plt.imshow(image, cmap='gray' if not is_color else None)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Read the image in color
img = cv2.imread('ferrari.jpg')
display_image(img, "Original", is_color=True)

# Crop the image (check indices to ensure they are within the image bounds)
cropped_image = img[300:1300, 200:2000]
display_image(cropped_image, "Cropped Image", is_color=True)

# Resize the image down and up
down_points = (300, 200)
resized_down = cv2.resize(img, down_points, interpolation=cv2.INTER_LINEAR)
up_points = (600, 400)
resized_up = cv2.resize(img, up_points, interpolation=cv2.INTER_LINEAR)
display_image(resized_down, 'Resized Down', is_color=True)
display_image(resized_up, 'Resized Up', is_color=True)

# Thresholding operations
src = cv2.imread("ferrari.jpg", cv2.IMREAD_GRAYSCALE)
_, dst_binary = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY)
_, dst_binary_maxval = cv2.threshold(src, 127, 128, cv2.THRESH_BINARY)
_, dst_binary_fixed = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY)
display_image(dst_binary, "Binary Thresholding")
display_image(dst_binary_maxval, "Binary Thresholding with maxVal 128")
display_image(dst_binary_fixed, "Binary Thresholding with threshold 127")

# Contour detection for each color channel
blue, green, red = cv2.split(img)
contours_blue, _ = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
image_contour_blue = cv2.drawContours(img.copy(), contours_blue, -1, (0, 255, 0), 2)
display_image(image_contour_blue, "Contour on Blue Channel", is_color=True)

contours_green, _ = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
image_contour_green = cv2.drawContours(img.copy(), contours_green, -1, (0, 255, 0), 2)
display_image(image_contour_green, "Contour on Green Channel", is_color=True)

contours_red, _ = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
image_contour_red = cv2.drawContours(img.copy(), contours_red, -1, (0, 255, 0), 2)
display_image(image_contour_red, "Contour on Red Channel", is_color=True)

# Blob detection
im = cv2.imread("ferrari.jpg")
detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(im)
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
display_image(im_with_keypoints, "Blob Detection", is_color=True)

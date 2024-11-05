import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg' )
plt.imshow(img)
plt.title('Original Image')
plt.show()

#line

pointA = (20,20)
pointB = (200,20)
cv2.line(img, pointA, pointB, (255, 255, 0), thickness=3, lineType=cv2.LINE_AA)
plt.imshow(img)
plt.title("Image Line")
plt.show()

#rectangle 

start_point =(10, 10)
end_point =(100,100)
cv2.rectangle(img, start_point, end_point, (225, 24, 255), thickness= 3, lineType=cv2.LINE_8) 
plt.imshow(img)
plt.title("Image Rectangle")
plt.show()

#circle
circle_center = (50,50)
radius =20
cv2.circle(img, circle_center, radius, (255, 225, 255), thickness=3, lineType=cv2.LINE_AA) 
plt.imshow(img)
plt.title("Image Circle")
plt.show()

#ellipse

center = (120,70)
axis1 = (30,64)
axis2 = (25,75)
cv2.ellipse(img, center, axis1, 0, 0, 360, (255, 255, 0), thickness=3)
cv2.ellipse(img, center, axis2, 90, 0, 360, (0, 255, 255), thickness=3)
plt.imshow(img)
plt.title("Image Ellipse")
plt.show()
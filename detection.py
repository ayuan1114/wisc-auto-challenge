import cv2 as cv
import numpy as np

im_color = cv.imread("red.png", cv.IMREAD_COLOR)
hsv = cv.cvtColor(im_color, cv.COLOR_BGR2HSV)

#create red mask
lower_red = np.array([0,170,170])
upper_red = np.array([10,230,230])

mask = cv.inRange(hsv, lower_red, upper_red)

#cv.imshow("binary_mask", mask)

#find contours
edged = cv.Canny(mask, 30, 200)

contours, hierarchy = cv.findContours(mask,  
    cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 

#separate left from right cones
left_cones = []
right_cones = []
for contour in contours:
    if contour[0, 0, 0] <= len(im_color[0]) // 2:
        left_cones.append(contour)
    else:
        right_cones.append(contour)

#draw contours for left and right cones on final
cv.drawContours(im_color, left_cones, -1, (0, 255, 0), 3) 
cv.drawContours(im_color, right_cones, -1, (255, 0, 0), 3) 

#combine all clusters for line fitting
left = left_cones[0]
for i in range(1, len(left_cones)):
    left = np.concatenate((left, left_cones[i]), axis = 0)
right = right_cones[0]
for i in range(1, len(right_cones)):
    right = np.concatenate((right, right_cones[i]), axis = 0)

#fit and draw lines
rows,cols = im_color.shape[:2]
[vx,vy,x,y] = cv.fitLine(left, cv.DIST_L2, 0, 0.01, 0.01)
l_int1 = int((-x * vy / vx) + y)
l_int2 = int(((cols - x)*vy / vx) + y)
cv.line(im_color, (cols - 1, l_int2), (0, l_int1), (0, 255, 0), 2)

[vx,vy,x,y] = cv.fitLine(right, cv.DIST_L2, 0, 0.01, 0.01)
r_int1 = int((-x * vy / vx) + y)
r_int2 = int(((cols - x) * vy / vx) + y)
cv.line(im_color, (cols - 1, r_int2), (0, r_int1), (255, 0, 0), 2)

cv.imshow('Contours', im_color)
cv.imwrite('./final.png',im_color) 
cv.waitKey(0) 
cv.destroyAllWindows()
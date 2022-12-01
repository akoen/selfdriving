#!/usr/bin/python3

import numpy as np
import cv2 as cv
import time
import os


path = os.path.dirname(os.path.realpath(__file__)) + "/"
img_folder = "pictures"
img_name = "p_right.png"
full_path = os.path.join(path, img_folder, img_name)

img = cv.imread(full_path,cv.IMREAD_COLOR)
# img = cv.medianBlur(img,5)
# Convert BGR to HSV
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# # isolates blue
# uh = 125
# us = 255
# uv = 255
# lh = 118
# ls = 40
# lv = 40

# # isolates partial rectangle around P2 and plate
# uh = 122
# us = 117
# uv = 105
# lh = 114
# ls = 000
# lv = 89

# used with blur
uh = 176
us = 11
uv = 117 # 98
lh = 115
ls = 0
lv = 89

mb = 13
img = cv.medianBlur(img,mb)
# img = cv.GaussianBlur(img,(5,5),0)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)


lower_hsv = np.array([lh,ls,lv])
upper_hsv = np.array([uh,us,uv])

# Threshold the HSV image to get only blue colors
# mask = cv.inRange(hsv, lower_hsv, upper_hsv) # if in range, 1, else, 0
window_name = "HSV Calibrator"
cv.namedWindow(window_name)

def nothing(x):
    print("Trackbar value: " + str(x))
    pass

# create trackbars for Upper HSV
cv.createTrackbar('UpperH',window_name,0,255,nothing)
cv.setTrackbarPos('UpperH',window_name, uh)

cv.createTrackbar('UpperS',window_name,0,255,nothing)
cv.setTrackbarPos('UpperS',window_name, us)

cv.createTrackbar('UpperV',window_name,0,255,nothing)
cv.setTrackbarPos('UpperV',window_name, uv)

# create trackbars for Lower HSV
cv.createTrackbar('LowerH',window_name,0,255,nothing)
cv.setTrackbarPos('LowerH',window_name, lh)

cv.createTrackbar('LowerS',window_name,0,255,nothing)
cv.setTrackbarPos('LowerS',window_name, ls)

cv.createTrackbar('LowerV',window_name,0,255,nothing)
cv.setTrackbarPos('LowerV',window_name, lv)

font = cv.FONT_HERSHEY_SIMPLEX

while(1):
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)
    
    # dilate, then do edge detection
    kernel = np.ones((21,21),np.uint8)
    dilation = cv.dilate(mask,kernel,iterations=1)
    edged = cv.Canny(dilation,75,200)
    cv.imshow("dilated edges", edged)

    # https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (0,255,0),2)
    cv.imshow("contours", img)
    # cv.putText(img,'num_contours = ' + str(len(contours)), (10,90), font, 0.5, (200,255,155), 1, cv.LINE_AA)

    cv.putText(mask,'Lower HSV: [' + str(lh) +',' + str(ls) + ',' + str(lv) + ']', (10,30), font, 0.5, (200,255,155), 1, cv.LINE_AA)
    cv.putText(mask,'Upper HSV: [' + str(uh) +',' + str(us) + ',' + str(uv) + ']', (10,60), font, 0.5, (200,255,155), 1, cv.LINE_AA)

    cv.imshow(window_name,mask)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # get current positions of Upper HSV trackbars
    uh = cv.getTrackbarPos('UpperH',window_name)
    us = cv.getTrackbarPos('UpperS',window_name)
    uv = cv.getTrackbarPos('UpperV',window_name)
    upper_blue = np.array([uh,us,uv])
    # get current positions of Lower HSCV trackbars
    lh = cv.getTrackbarPos('LowerH',window_name)
    ls = cv.getTrackbarPos('LowerS',window_name)
    lv = cv.getTrackbarPos('LowerV',window_name)

    upper_hsv = np.array([uh,us,uv])
    lower_hsv = np.array([lh,ls,lv])

    time.sleep(.1)

cv.destroyAllWindows()
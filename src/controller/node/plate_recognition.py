#!/usr/bin/python3
from __future__ import print_function

import roslib
roslib.load_manifest('controller')
import sys
import rospy
import cv2 as cv
import time
import os
import csv
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
import numpy as np
import copy

fourcc = cv.VideoWriter_fourcc(*'MJPG') 
out = cv.VideoWriter('/home/fizzer/ros_ws/src/controller/node/pictures/output.avi',fourcc,20.0,(1280,720)) # need to be same dimensions as input image

# blue hsv mask values
uh_plate = 125
us_plate = 255
uv_plate = 255
lh_plate = 118
ls_plate = 100 # 85 to 100
lv_plate = 40
lower_hsv_plate = np.array([lh_plate,ls_plate,lv_plate])
upper_hsv_plate = np.array([uh_plate,us_plate,uv_plate])

uh = 176 # Upper Hue
us = 11 # Upper Sat
uv = 190 # Upper Value
lh = 105 # Lower Hue
ls = 5 # Lower Sat
lv = 90 # Lower Value
# mb = 1 # Median Blur, 13
high_cntr_thresh_y = 575
low_cntr_thresh_y = 450
high_cntr_thresh_x = 1200
low_cntr_thresh_x = 5
plate_cntr_area_low_thresh = 6500 # increasing this helped to only capture plate when close (very close since it's pretty high)
plate_cntr_area_high_thresh = 100000 # infinite
lower_hsv = np.array([lh,ls,lv])
upper_hsv = np.array([uh,us,uv])
kernel = np.ones((17,17),np.uint8) # dilation kernel


class plate_recognizer():
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    
    def callback(self,data):
        
        try:
            img=self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        img_height, img_width, _ = img.shape
        # img_blur = cv.medianBlur(img,mb)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_hsv, upper_hsv)
        dilation = cv.dilate(mask,kernel,iterations=1) # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        edged = cv.Canny(dilation,75,200) # edges
        
        contours_edge, _ = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_edge = sorted(contours_edge,key=cv.contourArea,reverse=True) # largest to smallest contours
        
        plate_cntr_found = False
        if len(contours_edge) != 0: # prevent indexing errors when no contour found
            for cntr in contours_edge:
                x_cntr, y_cntr, width_cntr, height_cntr = cv.boundingRect(cntr)
                # check whether contour in acceptable area of screen
                #TODO: probably want two areas, one for the plates on the left, one for the plate(s) on the right
                if y_cntr > low_cntr_thresh_y and y_cntr+height_cntr < high_cntr_thresh_y and x_cntr > low_cntr_thresh_x and x_cntr < high_cntr_thresh_x:
                    largest_contour = cntr
                    plate_cntr_found = True
                    break
                else:
                    continue
                
        img_copy = copy.deepcopy(img)
        cv.rectangle(img_copy,(low_cntr_thresh_x,low_cntr_thresh_y),(high_cntr_thresh_x,high_cntr_thresh_y), (0,0,255),2)
        
        if plate_cntr_found:
            x,y,width,height = cv.boundingRect(largest_contour) # coordinates of largest contour
            largest_contour_area = width*height
            # accept contour if area between bounds
            if largest_contour_area > plate_cntr_area_low_thresh and largest_contour_area < plate_cntr_area_high_thresh:
                cv.drawContours(img_copy, [largest_contour], -1, (0,255,0),2)
                plate = img[y:y+height,x:x+width] # crop to isolate plate
                cv.imshow("plate",plate)
                
                #TODO: perspective transform could help for plates that are on corners, since we'll catch them right at the edges of our camera
                
                # mask blue
                hsv_plate = cv.cvtColor(plate, cv.COLOR_BGR2HSV)
                mask_plate = cv.inRange(hsv_plate,lower_hsv_plate,upper_hsv_plate)
                cv.imshow("mask plate",mask_plate)

                # get contours (no erode seems ok)
                contours_plate, _ = cv.findContours(mask_plate,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                contours_plate = sorted(contours_plate,key=cv.contourArea,reverse=True)
                plate_copy = copy.deepcopy(plate)
                cv.drawContours(plate_copy,contours_plate,-1,(0,255,0),1)
                cv.imshow("contours plate",plate_copy)         
                
                # contour's bounding rectangles
                bounding_rects = []
                for cntr in contours_plate:
                    x,y,width,height = cv.boundingRect(cntr)
                    bounding_rects.append([[x,y,width,height],(x+width * y+height)]) #[[[x,y,w,h],area], [[x,y,w,h],area], [[x,y,w,h],area]]]
                
                # remove bounding rectangles if blue edges present
                plate_width = plate.shape[1]
                hystersis = 5 # pixels from width of image
                bounding_rects_cropped = copy.deepcopy(bounding_rects)
                for i in bounding_rects:
                    x_i = i[0][0]
                    y_i = i[0][1]
                    width_i = i[0][2]
                    height_i = i[0][3]
                    if x_i == 0 or x_i + width_i == plate_width or x_i+width_i >= plate_width - hystersis:
                        bounding_rects_cropped.remove(i)
        
                # sort bounding rects by area
                areas = np.asarray([bounding_rects_cropped[i][1] for i in range(len(bounding_rects_cropped))])
                max_area_indicies = list(reversed(areas.argsort())) # index of max area is 0th element, descending
                
                # largest four bounding rectangles
                plate_rect_display = copy.deepcopy(plate)
                char_bounding_rects = []
                for k in range(4):
                    rect_idx = max_area_indicies[k]
                    x_i = bounding_rects_cropped[rect_idx][0][0]
                    y_i = bounding_rects_cropped[rect_idx][0][1]
                    width_i = bounding_rects_cropped[rect_idx][0][2]
                    height_i = bounding_rects_cropped[rect_idx][0][3]
                    char_bounding_rects.append([x_i,y_i,width_i,height_i])
                    cv.rectangle(plate_rect_display,(x_i,y_i+height_i),(x_i+width_i,y_i),(0,0,255),1) # (top left), (bottom right)
                
                char_bounding_rects.sort(key=lambda char_bounding_rects : char_bounding_rects[0]) # sort based on x value, lowest = leftmost = 0th
                cv.imshow("char bounding rectangles",plate_rect_display)

                # isolate characters
                for c in range(len(char_bounding_rects)):
                    x_c = char_bounding_rects[c][0]
                    y_c = char_bounding_rects[c][1]
                    width_c = char_bounding_rects[c][2]
                    height_c = char_bounding_rects[c][3]
                    padding = 1 # additional pixels to include, for if we crop in too much
                    char = plate[y_c-padding:y_c+height_c+padding,x_c-padding:x_c+width_c+padding] # cropped char from plate

        # out.write(img_copy)
        cv.imshow("contours with bounds",img_copy)
        cv.waitKey(3)

def main(args):
    rospy.init_node('plate_recognition', anonymous=True)
    pr = plate_recognizer()

    try:
        rospy.spin()
        return
    except KeyboardInterrupt:
        print("Shutting down")
    cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
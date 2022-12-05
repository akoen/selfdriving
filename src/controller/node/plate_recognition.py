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

class plate_recognizer():
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    
    def callback(self,data):
        
        uh = 176 # Upper Hue
        us = 11 # Upper Sat
        uv = 190 # Upper Value
        lh = 105 # Lower Hue
        ls = 5 # Lower Sat
        lv = 90 # Lower Value
        # mb = 1 # Median Blur, 13
        high_cntr_thresh_y = 525
        low_cntr_thresh_y = 400
        high_cntr_thresh_x = 1020
        low_cntr_thresh_x = 200
        plate_cntr_area_low_thresh = 2750
        plate_cntr_area_high_thresh = 8000


        lower_hsv = np.array([lh,ls,lv])
        upper_hsv = np.array([uh,us,uv])
        kernel = np.ones((17,17),np.uint8) # dilation kernel

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
                #TODO: perspective transform would help for plates that are on corners, since we'll catch them right at the edges of our camera           
                # cv.imshow("plate",plate)
        
        out.write(img_copy)
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
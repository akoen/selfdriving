#!/usr/bin/python3
from __future__ import print_function

import roslib
roslib.load_manifest('controller')
import sys
import rospy
import cv2 as cv
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
import numpy as np
import copy


class plate_recognizer():
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    dd
    def callback(self,data):
        
        uh = 176 # Upper Hue
        us = 11 # Upper Sat
        uv = 190 # Upper Value
        lh = 105 # Lower Hue
        ls = 5 # Lower Sat
        lv = 90 # Lower Value
        mb = 13 # Median Blur

        lower_hsv = np.array([lh,ls,lv])
        upper_hsv = np.array([uh,us,uv])
        kernel = np.ones((17,17),np.uint8) # dilation kernel

        try:
            img=self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        height, width, _ = img.shape
        img_cropped = img[int(height/2): height, :] # y:y+h,x:x+w
        img_blur = cv.medianBlur(img_cropped,mb)
        hsv = cv.cvtColor(img_blur, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_hsv, upper_hsv)
        dilation = cv.dilate(mask,kernel,iterations=1) # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        edged = cv.Canny(dilation,75,200) # edges
        
        contours_edge, _ = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_edge = sorted(contours_edge,key=cv.contourArea,reverse=True) # largest to smallest contours
        largest_contour = contours_edge[0]
        # TODO: probably want the contour to have at least a certain size before we accept it as "valid"
        x,y,width,height = cv.boundingRect(largest_contour) # coords of largest contour
        plate = img[y:y+height,x:x+width] # isolate plate

        # image display
        img_copy = copy.deepcopy(img_cropped)
        cv.drawContours(img_copy, [largest_contour], -1, (0,255,0),2)
        cv.imshow("Image",img_copy)
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
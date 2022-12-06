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
from tensorflow import keras

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
high_cntr_thresh_x_right = 1275
low_cntr_thresh_x_right = 1000
high_cntr_thresh_x_left = 300
low_cntr_thresh_x_left = 3
plate_cntr_area_low_thresh = 5500 # 6500 good (spotty on right)
plate_cntr_area_high_thresh = 100000 # infinite
lower_hsv = np.array([lh,ls,lv])
upper_hsv = np.array([uh,us,uv])
font = cv.FONT_HERSHEY_SIMPLEX
initial_dilate_kernel = np.ones((17,17),np.uint8) # dilation kernel
model_dilate_kernel = np.ones((2,2),np.uint8)

class plate_recognizer():

    def __init__(self):
        self.bridge = CvBridge()
        # self.camera_callback = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.camera_callback)
        self.camera_instance = "/R1/pi_camera/image_raw"
        self.conv_model = keras.models.load_model('/home/fizzer/ros_ws/src/controller/node/conv_model_74k')
        
        self.timer_started = False
        self.timer = 0 # float seconds
        self.timer_elapsed_threshold = 4 # seconds, 5 worked fine
        self.plates_in_duration = []

    
    def detect_plate_in_image(self, img):
        plate = 0
        plate_area = 0
        plate_cntr_found = False

        # plate mask
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_hsv, upper_hsv)
        dilation = cv.dilate(mask,initial_dilate_kernel,iterations=1) # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        edged = cv.Canny(dilation,75,200) # edges
        
        contours_edge, _ = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_edge = sorted(contours_edge,key=cv.contourArea,reverse=True) # largest to smallest contours
        
        if len(contours_edge) != 0: # prevent indexing errors when no contour found
            for cntr in contours_edge:
                x_cntr, y_cntr, width_cntr, height_cntr = cv.boundingRect(cntr)
                # check whether contour in acceptable area of screen           
                if (y_cntr > low_cntr_thresh_y and y_cntr+height_cntr < high_cntr_thresh_y and x_cntr > low_cntr_thresh_x_left and x_cntr+width_cntr < high_cntr_thresh_x_left) \
                    or (y_cntr > low_cntr_thresh_y and y_cntr+height_cntr < high_cntr_thresh_y and x_cntr > low_cntr_thresh_x_right and x_cntr + width_cntr < high_cntr_thresh_x_right):
                    largest_contour = cntr
                    plate_cntr_found = True
                    break
                
        img_copy = copy.deepcopy(img)
        cv.rectangle(img_copy,(low_cntr_thresh_x_left,low_cntr_thresh_y),(high_cntr_thresh_x_left,high_cntr_thresh_y), (0,0,255),2)
        cv.rectangle(img_copy,(low_cntr_thresh_x_right,low_cntr_thresh_y),(high_cntr_thresh_x_right,high_cntr_thresh_y), (0,0,255),2)
        
        if plate_cntr_found:
            # accept contour only if has certain area
            x,y,width,height = cv.boundingRect(largest_contour) # coordinates of largest contour
            largest_contour_area = width*height
            # TODO: maybe two area thresholds, one for left and one for right (right one would be smaller than left one)
            if largest_contour_area > plate_cntr_area_low_thresh and largest_contour_area < plate_cntr_area_high_thresh:
                cv.drawContours(img_copy, [largest_contour], -1, (0,255,0),2)
                plate = img[y:y+height,x:x+width] # crop to isolate plate
                plate_area = height*width

        out.write(img_copy)
        cv.imshow("contours with bounds",img_copy)
        cv.waitKey(3)

        return plate_cntr_found, plate, plate_area

    def get_best_plate(self):
        # get plate with max area
        plate_areas = np.asarray([self.plates_in_duration[i][1] for i in range(len(self.plates_in_duration))])
        plate_areas_sorted = list(reversed(plate_areas.argsort())) # index of max area is 0th element, descending
        plate = self.plates_in_duration[plate_areas_sorted[0]][0]
        plate_area = plate_areas[plate_areas_sorted[0]] # area of plate with max area
        
        print(f"plate areas: {plate_areas}")

        if(plate_area!=0):
            cv.imshow("plate", plate)
            cv.waitKey(3)

        return plate, plate_area

    def process_plate(self, plate):
        # mask blue
        hsv_plate = cv.cvtColor(plate, cv.COLOR_BGR2HSV)
        mask_plate = cv.inRange(hsv_plate,lower_hsv_plate,upper_hsv_plate)
        cv.imshow("mask plate",mask_plate)

        # get contours (no erode seems ok)
        contours_plate, hierarchy = cv.findContours(mask_plate,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
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

        cv.waitKey(3)
        return bounding_rects_cropped

    def process_characters(self, bounding_rects_cropped, plate):
        # sort bounding rects by area
        areas = np.asarray([bounding_rects_cropped[i][1] for i in range(len(bounding_rects_cropped))])
        max_area_indicies = list(reversed(areas.argsort())) # index of max area is 0th element, descending
        
        # split large contours (when blue bleeds between letters)
        bounding_rects_split = copy.deepcopy(bounding_rects_cropped)
        max_rect_index = max_area_indicies[0]
        max_rect_area = areas[max_rect_index]
        split_rectangle_threshold = 1.5 # multiplication factor
        second_max_rect_area = areas[max_area_indicies[1]]
        
        # if max_rect_area >= split_rectangle_threshold*second_max_rect_area: # requires tweaking threshold, more finnickey
        if len(areas) <= 3: # if only three contours
            x_max = bounding_rects_cropped[max_rect_index][0][0]
            y_max = bounding_rects_cropped[max_rect_index][0][1]
            width_max = bounding_rects_cropped[max_rect_index][0][2]
            height_max = bounding_rects_cropped[max_rect_index][0][3]

            x_left = x_max
            width_left = int(width_max / 2) - 1
            x_right = x_max + width_left + 1
            width_right = int(width_max / 2)

            del bounding_rects_split[max_rect_index] # remove large rectangle
            bounding_rects_split.append(([[x_left,y_max,width_left,height_max],(x_left+width_left * y_max+height_max)]))
            bounding_rects_split.append(([[x_right,y_max,width_right,height_max],(x_right+width_right * y_max+height_max)]))

        areas_split = np.asarray([bounding_rects_split[i][1] for i in range(len(bounding_rects_split))])
        max_area_indicies_split = list(reversed(areas_split.argsort())) # index of max area is 0th element, descending

        # largest four bounding rectangles
        plate_rect_display = copy.deepcopy(plate)
        char_bounding_rects = []
        for k in range(4):
            rect_idx = max_area_indicies_split[k]
            x_i = bounding_rects_split[rect_idx][0][0]
            y_i = bounding_rects_split[rect_idx][0][1]
            width_i = bounding_rects_split[rect_idx][0][2]
            height_i = bounding_rects_split[rect_idx][0][3]
            char_bounding_rects.append([x_i,y_i,width_i,height_i])
            cv.rectangle(plate_rect_display,(x_i,y_i+height_i),(x_i+width_i,y_i),(0,0,255),1) # (top left), (bottom right)
        
        char_bounding_rects.sort(key=lambda char_bounding_rects : char_bounding_rects[0]) # sort based on x value, lowest = leftmost = 0th
        cv.imshow("char bounding rectangles",plate_rect_display)

        # isolate characters
        chars = []
        for c in range(len(char_bounding_rects)):
            x_c = char_bounding_rects[c][0]
            y_c = char_bounding_rects[c][1]
            width_c = char_bounding_rects[c][2]
            height_c = char_bounding_rects[c][3]
            padding = 1 # additional pixels to include, for if we crop in too much
            char = plate[y_c-padding:y_c+height_c+padding,x_c-padding:x_c+width_c+padding] # cropped char from plate
            chars.append(char)

        return chars

    def predict_characters(self, chars, plate):
        # predict characters
        predictions = []
        for k in chars:
            char_img = cv.erode(cv.cvtColor(k,cv.COLOR_BGR2GRAY),model_dilate_kernel,iterations=1) # helps "sharpen" letters, can experment with kernel size
            char_img_resized = cv.resize(char_img, (50,80)) # resize for network
            prediction = [self.conv_model.predict(np.expand_dims(char_img_resized,axis=0))[0]]
            predictions.append(prediction)

        max_predictions = [np.argmax(i) for i in predictions]
        max_predictions_chars = [chr(i-10+65) if i >= 11 else str(i) for i in max_predictions]
        cv.putText(plate,'[' + max_predictions_chars[0] + ',' + max_predictions_chars[1] + ',' + max_predictions_chars[2] + ',' + max_predictions_chars[3] + ']', (1,10), font, 0.5, (255,255,255), 1, cv.LINE_AA)
        cv.imshow("plate", plate)

        cv.waitKey(3)
        return max_predictions_chars


def main(args):
    rospy.init_node('plate_recognition', anonymous=True)
    pr = plate_recognizer()

    while True:
        # wait for data from camera
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(pr.camera_instance, Image, timeout=5) # timeout in seconds
                print('got image')
            except:
                pass

        # get image, check if contains plate
        try:
            img=pr.bridge.imgmsg_to_cv2(data, "bgr8")
            plate_found, plate, plate_area = pr.detect_plate_in_image(img)
            print(f"plate_found: {plate_found}")
        except CvBridgeError as e:
            print(e)

        # TODO: could also try continuous plates 
        # if img contains plate and last image contains plate, add to buffer, as soon as get image that doesn't contain plate when last img did contain plate, process buffer)
        if plate_found and pr.timer_started == False: # first plate in sequence
            print("started timer")
            pr.timer_started = True
            pr.timer = rospy.get_time() # this is the "start of sequence" time
            pr.plates_in_duration.append([plate, plate_area])
            continue # beginning of while True
        
        elif plate_found and pr.timer_started and rospy.get_time() - pr.timer < pr.timer_elapsed_threshold: # add plate to sequence
            print("added plate to sequence")
            print(f"num sequence plates: {len(pr.plates_in_duration)}")
            pr.plates_in_duration.append([plate, plate_area])
        
        elif pr.timer_started and rospy.get_time() - pr.timer >= pr.timer_elapsed_threshold: # if timer has elapsed
            print('processing plate batch')
            plate, plate_area = pr.get_best_plate() # get plate (in duration) with largest area
            
            if plate_area == 0:
                pr.plates_in_duration = []
                pr.timer_started = False
                continue # beginning of while True

            bounding_rects_cropped = pr.process_plate(plate) # get bounding rects
            chars = pr.process_characters(bounding_rects_cropped, plate) # get char bounding rects
            prediction = pr.predict_characters(chars, plate) # get final prediction

            pr.plates_in_duration = [] # reset plate buffer
            pr.timer_started = False # reset timer
        
        # else, no plate was found, do nothing

if __name__ == '__main__':
    main(sys.argv)
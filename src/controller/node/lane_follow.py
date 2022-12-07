import cv2
import numpy as np
import logging
import math
import datetime
import sys

_SHOW_IMAGE = False


class HandCodedLaneFollower(object):

    def __init__(self, car=None):
        # logging.info('Creating a HandCodedLaneFollower...')
        self.car = car
        self.curr_steering_angle = 90

    def follow_lane(self, frame):
        # Main entry point of the lane follower

        lane_lines, frame = detect_lane(frame)
        final_frame = self.steer(frame, lane_lines)

        return final_frame

    def steer(self, frame, lane_lines):
        logging.debug('steering...')
        if len(lane_lines) == 0:
            logging.error('No lane lines detected, nothing to do.')
            return frame

        new_steering_angle = compute_steering_angle(frame, lane_lines)
        # self.curr_steering_angle = stabilize_steering_angle(self.curr_steering_angle, new_steering_angle, len(lane_lines))

        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)
        curr_heading_image = display_heading_line(frame, self.curr_steering_angle)

        return curr_heading_image, new_steering_angle


############################
# Frame processing steps
############################
def detect_lane(frame):

    # frame = cv2.GaussianBlur(frame, (11,11), cv2.BORDER_DEFAULT)
    frame_out = frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([0, 0, 140])
    upper_blue = np.array([255, 0, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # detect edges
    edges = cv2.Canny(mask, 200, 400)

    # cv2.imshow("Mask", mask)
    # cv2.imshow("Canny", edges)

    cropped_edges = edges

    # edges = cv2.GaussianBlur(edges, (15,15), cv2.BORDER_DEFAULT)
    edges = cv2.dilate(edges, np.ones((5,5)), iterations=1)

    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # degree in radian, i.e. 1 degree
    min_threshold = 40  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=10, maxLineGap=20)
    # lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE, 0.2)
    # line_segments = lsd.detect(mask)[0]

    # lines = np.array([[[x1, y1], [x2, y2]] for [[x1,y1,x2,y2]] in line_segments]).astype(int)
    # lines = lines[np.array([np.linalg.norm(e[1]-e[0]) for e in lines]) > 20]

    # cv2.imshow("Lines", cv2.polylines(np.zeros(edges.shape), lines, False, (255,100,0), 2))
    # cv2.imshow("Mask", edges)

    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    left_weights = []
    right_fit = []
    right_weights = []
    lines = []
    reject_lines = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

    Y_MIN = 550
    X_MIN = 450

    height, width, _ = frame.shape

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            # if np.abs(slope) > 0.08 and (max(y1,y2)>Y_MIN or (min(x1,x2) > X_MIN and (max(x1, x2) < width-X_MIN))):
            if np.abs(slope) > 0.08 and (max(y1,y2)>Y_MIN or (min(x1,x2) > X_MIN)):
                # print(max(y1,y2), min(x1,x2), width-max(x1,x2))
                if slope < 0:
                    if x1 < left_region_boundary and x2 < left_region_boundary:
                        left_fit.append((slope, intercept))
                        left_weights.append(np.sqrt((x2-x1)**2+(y2-y1)**2))
                else:
                    if x1 > right_region_boundary and x2 > right_region_boundary:
                        right_fit.append((slope, intercept))
                        right_weights.append(np.sqrt((x2-x1)**2+(y2-y1)**2))
                
                lines.append(np.array([[x1, y1], [x2, y2]], dtype=int))
            else:
                reject_lines.append(np.array([[x1, y1], [x2, y2]], dtype=int))



    # print(lines)
    frame_out = cv2.addWeighted(frame_out, 0.6, cv2.polylines(np.zeros_like(frame), lines, False, (255,255,255), 2), 0.6, 1)
    frame_out = cv2.addWeighted(frame_out, 1, cv2.polylines(np.zeros_like(frame), reject_lines, False, (0,0,255), 2), 0.3, 1)

    frame_out = cv2.rectangle(frame_out, (0, 0), (X_MIN, Y_MIN), (0, 0, 255, 0.3), 1)

    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0, weights=left_weights)
        lane_lines.append(make_points(frame, left_fit_average))

    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0, weights=right_weights)
        lane_lines.append(make_points(frame, right_fit_average))

    logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    # Display Lines
    line_width=4
    line_image = np.zeros_like(frame)
    if lane_lines is not None:
        for line in lane_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255,100,0), line_width)
    frame_out = cv2.addWeighted(frame_out, 1, line_image, 1, 1)

    heading_image = np.zeros_like(frame)

    if len(lane_lines) == 0:
        logging.info('No lane lines detected, do nothing')
        return -90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        # return None, None, lane_lines, frame_out
        logging.debug('Only detected one lane line, just follow it. %s' % lane_lines[0])
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = (x2 - x1) / 8
        # x_offset = 0
        mid=width/2
        offset_1 = 0
    else:
        left_x1, _, left_x2, _ = lane_lines[0][0]
        right_x1, _, right_x2, _ = lane_lines[1][0]
        camera_mid_offset_percent = 0.00 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid
        offset_1 = (left_x1 + right_x1) / 2 - mid

    # find the steering angle, which is angle between navigation direction to end of center line
    y_offset = int(height / 2)

    steering_angle = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right 
    steering_angle_radian = steering_angle
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 + height / 2 * math.tan(steering_angle_radian ))
    y2 = int(height / 2)

    cv2.line(heading_image, (int(offset_1+mid), y1), (x2, y2), (255,0,100), line_width)
    frame_out = cv2.addWeighted(frame_out, 1, heading_image, 1, 1)

    return steering_angle, offset_1/(width/2), lane_lines, frame_out

############################
# Utility Functions
############################



def length_of_line_segment(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)



def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

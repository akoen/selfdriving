#!/usr/bin/python3
from __future__ import print_function
import lane_follow
from typing import Union
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import rospy
import sys

import roslib
roslib.load_manifest('controller')


GRAYSCALE_THRESHOLD = 254


def PID(kp, ki, kd):
    value = 0
    prev_error = 0
    I = 0

    while True:
        error = yield value
        value = kp*error + (I+ki*error) + kd*(error-prev_error)
        prev_error = error


class Selfdriving:
    def __init__(self):
        # self.image_pub = rospy.Publisher("image_topic_2",Image)
        self.pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/R1/pi_camera/image_raw", Image, self.callback)

        self.state = DriveWithPID()


        self.waiting = False
        self.waiting_started = False
        # self.wait_last_time = -10
        self.speeding_end_time = 0

    def callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        move = Twist()

        frame_out, new_state = self.state.run(frame, move)
        if new_state != self.state:
            print(f"Transitioning to state: {new_state.name}")
        self.setState(new_state)


        cv2.imshow("Image window", frame_out)
        cv2.waitKey(3)

        try:
            self.pub.publish(move)
        except CvBridgeError as e:
            print(e)
    
    def setState(self, state):
        self.state = state
        self.state.context = self


class State(ABC):

    @abstractproperty
    def name(self):
        pass

    @abstractmethod
    def run(self, frame):
        pass

class DriveWithPID(State):
    def __init__(self):
        self.angle_control = PID(0.8, 0.5, 0.8)
        self.angle_control.send(None)  # Initialize
        self.offset_control = PID(0.7, 0.5, 0)
        self.offset_control.send(None)  # Initialize
        self.PID_value = -0

        self.speeding_end_time = 0
        self.wait_last_time = -10

        self.lane_follower = lane_follow.HandCodedLaneFollower()
    
    @property
    def name(self):
        return "Drive With PID"
    
    def run(self, frame, move):
        (rows, cols, channels) = frame.shape
        time = rospy.get_time()

        hsv = cv2.cvtColor(frame[-int(rows/7):, :], cv2.COLOR_BGR2HSV)
        lower = np.array([0, 200, 40])
        upper = np.array([1, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)


        if np.sum(mask) > 1e6 and time - self.wait_last_time > 10:
            self.waiting = True
            print("delete")
            return frame, WaitForPedestrian()

        steer_angle, offset, lane_lines, frame_out = lane_follow.detect_lane(
            frame)

        move.linear.x = 0.3 if time < self.speeding_end_time else 0.1

        if steer_angle is not None:
            self.PID_value = self.angle_control.send(
                steer_angle) + self.offset_control.send(offset)
            # print(f"{steer_angle:.2f}, {offset:.2f}, {self.PID_value:.2f}")

        move.angular.z = -self.PID_value

        return frame_out, self

class WaitForPedestrian(State):

    def __init__(self):
        self.waiting_started = False
        self.prev_frame = None

    @property
    def name(self):
        return "Waiting for Pedestrian"

    def run(self, frame, move):
        if self.prev_frame is None:
            self.prev_frame = frame
        time = rospy.get_time()

        move.linear.x = 0

        def togray(mat): return cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
        def blur(mat): return cv2.GaussianBlur(mat, (5, 5), sigmaX=0)

        # x = np.where(np.abs(blur(togray(frame))-blur(togray(self.prev_frame)))>20, 255, 0).astype(np.uint8)
        frame_diff = cv2.absdiff(
            blur(togray(frame[:, 1:-1])), blur(togray(self.prev_frame[:, 1:-1])))
        # frame_diff = cv2.dilate(frame_diff, np.ones((5,5)), 1)

        frame_diff_thresh = cv2.threshold(
            frame_diff, 20, 255, type=cv2.THRESH_BINARY)[1]
        if self.waiting_started:
            if np.sum(frame_diff_thresh) > 10:
                print("Moving")
                self.prev_frame = frame
                return frame_diff_thresh, self
            else:
                print("Exiting")
                s = DriveWithPID()
                s.speeding_end_time = time+2
                s.wait_last_time = time+10
                return frame_diff_thresh, s
        else:
            if np.sum(frame_diff_thresh) > 10:
                self.waiting_started = True
                print("Moving started")
                self.prev_frame = frame
                return frame_diff_thresh, self
            else:
                print("Moving not started")
                return frame_diff_thresh, self


def main(args):
    ic = Selfdriving()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
        return
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)

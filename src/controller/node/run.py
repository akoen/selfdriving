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
import time

from helpers import *
import roslib
roslib.load_manifest('controller')

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

        self.state = InitialTurn()
        # self.state = DriveWithPID()

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

        print(rospy.get_time())

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
    num_crosswalks = 0

    def __init__(self):
        # self.angle_control = PID(0.8, 0.5, 0.8)
        self.angle_control = PID(1.6, 1, 1.6)
        # self.angle_control = PID(2, 1, 1)
        self.angle_control.send(None)  # Initialize
        self.offset_control = PID(1.4, 1, 0)
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
            DriveWithPID.num_crosswalks += 1
            if DriveWithPID.num_crosswalks == 3:
                return frame, InsideTurn()
            else:
                return frame, WaitForPedestrian()
            # return frame, WaitForPedestrian()

        steer_angle, offset, lane_lines, frame_out = lane_follow.detect_lane(
            frame)

        move.linear.x = 0.3 if time < self.speeding_end_time else 0.2

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
        self.start_time = rospy.get_time()

    @property
    def name(self):
        return "Waiting for Pedestrian"

    def run(self, frame, move):
        # Wait for car to stop
        time = rospy.get_time()
        while time < self.start_time + 0.4:
            self.prev_frame = frame
            return frame, self

        frame_diff_thresh, motion = detectMotion(frame, self.prev_frame, bounds=(200,200,0,0))
        self.prev_frame = frame

        if self.waiting_started:
            if motion:
                print("Moving")
            else:
                print("Exiting")
                s = DriveWithPID()
                s.speeding_end_time = time+2
                s.wait_last_time = time+10
                return frame_diff_thresh, s
        else:
            if motion:
                self.waiting_started = True
                print("Moving started")
                self.prev_frame = frame
            else:
                print("Moving not started")

        
        return frame_diff_thresh, self

class InitialTurn(State):
    start_time = 3
    @property
    def name(self):
        return "InitialTurn"

    def run(self, frame, move):

        # if not hasattr(self, "start_time"): self.start_time = rospy.get_time()
        # time = rospy.get_time() - self.start_time

        if rospy.get_time() < InitialTurn.start_time:
            return frame, self

        time = rospy.get_time() - InitialTurn.start_time # physics unpaused time
        print(time)
        if time < 0.75:
            move.linear.x = 0.2
            return frame, self
        elif time < 2.5:
            move.linear.x = 0.2
            move.angular.z = 1
            return frame, self
        else:
            return frame, DriveWithPID()

class InsideTurn(State):
    @property
    def name(self):
        return "Inside Turn"

    def __init__(self):
        self.start_time = rospy.get_time()

    def run(self, frame, move):
        time = rospy.get_time() - self.start_time
        if time < 0.65:
            move.linear.x = -0.2
            return frame, self
        elif time < 2.05:
            move.linear.x = -0.2
            move.angular.z = 1.5
            return frame, self
        elif time < 4.5:
            move.linear.x = 0.2
            return frame, self
        else:
            return frame, WaitForCar()

class PostCarTurn(State):
    @property
    def name(self):
        return "Post Car Turn"

    def __init__(self):
        self.start_time = rospy.get_time()

    def run(self, frame, move):
        time = rospy.get_time() - self.start_time
        if time < 1.6:
            move.linear.x = 0.2
            move.angular.z = 1
            return frame, self
        else:
            return frame, DriveWithPID()

class WaitForCar(State):
    @property
    def name(self):
        return "Waiting for Car"

    def __init__(self):
        self.waiting_started = False
        self.startTime = rospy.get_time()

    def run(self, frame, move):
        # Wait for car to stop
        time = rospy.get_time()
        while time < self.startTime + 0.4:
            self.prev_frame = frame
            return frame, self

        motion_frame, motion = detectMotion(frame, self.prev_frame, bounds=(400, 450, 400, 0))
        self.prev_frame = frame

        if self.waiting_started:
            if motion:
                print("Moving")
            else:
                print("Exiting")
                return motion_frame, PostCarTurn()
        else:
            if motion:
                self.waiting_started = True
                print("Moving started")
            else:
                print("Moving not started")

        return motion_frame, self


def main(args):
    rospy.init_node('state_PID_runner', anonymous=True)
    ic = Selfdriving()

    try:
        rospy.spin()
        return
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)

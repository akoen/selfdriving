#!/usr/bin/python3
from __future__ import print_function

import roslib
roslib.load_manifest('controller')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
import numpy as np

import lane_follow

GRAYSCALE_THRESHOLD=254

def PID(kp, ki, kd):
  value = 0
  prev_error = 0
  I = 0

  while True:
    error = yield value
    value = kp*error + (I+ki*error) + kd*(error-prev_error)
    prev_error = error

class selfdriving:
  def __init__(self):
    # self.image_pub = rospy.Publisher("image_topic_2",Image)
    self.pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)

    self.lane_follower = lane_follow.HandCodedLaneFollower()

    self.angle_control = PID(0.8, 0.5, 0.8)
    self.angle_control.send(None) # Initialize
    self.offset_control = PID(0.7, 0.5, 0)
    self.offset_control.send(None) # Initialize
    self.PID_value = -0

    self.prev_frame = None
    self.waiting = False
    self.waiting_started = False
    self.wait_last_time = -10
    self.speeding_end_time = 0

  def callback(self,data):
    try:
      frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
      if self.prev_frame is None: self.prev_frame = frame
    except CvBridgeError as e:
      print(e)

    # print(type(Frame))


    (rows,cols,channels) = frame.shape

    move = Twist()
    time = rospy.get_time()

    print(rospy.get_time())

    if self.waiting:
      move.linear.x = 0
      self.pub.publish(move) 

      togray = lambda mat: cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
      blur = lambda mat: cv2.GaussianBlur(mat, (5,5), sigmaX=0)

      # x = np.where(np.abs(blur(togray(frame))-blur(togray(self.prev_frame)))>20, 255, 0).astype(np.uint8)
      frame_diff = cv2.absdiff(blur(togray(frame[:,1:-1])), blur(togray(self.prev_frame[:,1:-1])))
      # frame_diff = cv2.dilate(frame_diff, np.ones((5,5)), 1)

      frame_diff_thresh = cv2.threshold(frame_diff, 20, 255, type=cv2.THRESH_BINARY)[1]
      cv2.imshow("Image window", frame_diff_thresh)
      cv2.waitKey(3)
      if self.waiting_started:
        if np.sum(frame_diff_thresh) > 10:
          print("Moving")
          self.prev_frame = frame
          return
        else:
          self.waiting = False
          self.waiting_started = False
          self.wait_last_time = time
          self.speeding_end_time = time+2
          print("Exiting")
      else:
        if np.sum(frame_diff_thresh) > 10:
          self.waiting_started = True
          print("Moving started")
          self.prev_frame = frame
          return
        else:
          print("Moving not started")


    hsv = cv2.cvtColor(frame[-int(rows/7):,:], cv2.COLOR_BGR2HSV)
    lower = np.array([0, 200, 40])
    upper = np.array([1, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    print(np.sum(mask))

    if np.sum(mask) > 1e6 and time - self.wait_last_time > 10:
      self.waiting = True
      return

    steer_angle, offset, lane_lines, frame_out = lane_follow.detect_lane(frame)

    # Show camera view
    cv2.imshow("Image window", frame_out)
    cv2.waitKey(3)

    move.linear.x = 0.3 if time < self.speeding_end_time else 0.1
    print("Linear x")

    if steer_angle is not None:
      self.PID_value = self.angle_control.send(steer_angle) + self.offset_control.send(offset)
      # print(f"{steer_angle:.2f}, {offset:.2f}, {self.PID_value:.2f}")

    move.angular.z = -self.PID_value

    self.prev_frame = frame
    try:
        self.pub.publish(move) 
    except CvBridgeError as e:
        print(e)


def main(args):
  ic = selfdriving()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
    return
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
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

class line_following:
  def __init__(self):
    # self.image_pub = rospy.Publisher("image_topic_2",Image)
    self.pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)

    self.lane_follower = lane_follow.HandCodedLaneFollower()

    self.angle_control = PID(0.8, 0.5, 0.8)
    self.angle_control.send(None) # Initialize
    # self.offset_control = PID(0.7, 0, 0)
    self.offset_control = PID(0.7, 0.5, 0)
    self.offset_control.send(None) # Initialize
    self.PID_value = -0

  def callback(self,data):
    try:
      frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)


    (rows,cols,channels) = frame.shape
    # Crop frame to ignore sky
    # frame_crop = frame[-300:-1, 100:1000]
    # frame_gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
    
    # lane_lines, frame = self.lane_follower.detectL(frame)

    steer_angle, offset, lane_lines, frame_out = lane_follow.detect_lane(frame)

    # Show camera view
    # frame_out = cv2.circle(frame_out, (int(x), 700), 4, (0, 0, 255), -1)
    cv2.imshow("Image window", frame_out)
    cv2.waitKey(3)


    move = Twist()
    move.linear.x = 0.1

    if steer_angle is not None:
      self.PID_value = self.angle_control.send(steer_angle) + self.offset_control.send(offset)
      print(f"{steer_angle:.2f}, {offset:.2f}, {self.PID_value:.2f}")

    move.angular.z = -self.PID_value
    try:
        self.pub.publish(move) 
    except CvBridgeError as e:
        print(e)


def main(args):
  ic = line_following()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
    return
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
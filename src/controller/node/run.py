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

    self.controller = PID(0.7, 0.5, 0)
    self.controller.send(None) # Initialize
    self.error = -0.2

  def callback(self,data):
    try:
      frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)


    (rows,cols,channels) = frame.shape
    # Crop frame to ignore sky
    frame_crop = frame[-300:-1, 100:1000]
    # frame_gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
    frame_hsv = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
    _, frame_thresh_sat = cv2.threshold(frame_hsv[:,:,1], 1, 255, cv2.THRESH_BINARY_INV)
    frame_thresh_val = cv2.inRange(frame_hsv[:,:,2], 80, 90)
    frame_bin_thresh = np.bitwise_and(frame_thresh_sat, frame_thresh_val)
    # img_hsv = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
    # saturation = img_hsv[:, :, 1]
    # _, frame_bin_thresh = cv2.threshold(saturation, 1, 255, cv2.THRESH_BINARY_INV)
    # _, frame_bin_thresh = cv2.threshold(frame_gray, 80, 255, cv2.THRESH_BINARY)
    # Compute the average x value of the black pixels (approx position of line)
    M = cv2.moments(frame_bin_thresh)
    x = M["m10"] / (M["m00"]+1)
    
    # Show camera view
    frame_out = frame
    # frame_out = cv2.circle(frame_out, (int(x), 700), 4, (0, 0, 255), -1)
    cv2.imshow("Image window", frame)
    cv2.waitKey(3)

    # move = Twist()
    # move.linear.x = 0.1
    # if not np.isnan(x):
    #   self.error = (x-400)/800

    # PID_value = self.controller.send(self.error) 
    # move.angular.z = -PID_value
    # print(self.error, PID_value)
    # try:
    #     self.pub.publish(move) 
    # except CvBridgeError as e:
    #     print(e)


def main(args):
  ic = line_following()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
#!/usr/bin/python3
from __future__ import print_function

import roslib
roslib.load_manifest('controller')
import rospy
from std_msgs.msg import String
import numpy as np
import time

class score_tracker():
    def __init__(self):
        self.pub_plate = rospy.Publisher("/license_plate", String)
        self.team_name = "Fish"
        self.team_password = "password"
        self.register_plate_location = "0"
        self.stop_plate_location = "-1"
        self.garbage_plate = "XXXX"
        # self.pub_clock = rospy.Subscriber("/clock", int)

    def send_plate(self, plate_location, plate_id):
        self.pub_plate.publish(f"{self.team_name},{self.team_password},{plate_location},{plate_id}")

def main(args):
    tracker = score_tracker()
    rospy.init_node('tracker', anonymous=False)

    tracker.send_plate(tracker.register_plate_location,tracker.garbage_plate)

    start_time = time.time()
    while(time.time() - start_time < 3):
        continue #wait for three seconds

    tracker.send_plate(tracker.stop_plate_location,tracker.garbage_plate)
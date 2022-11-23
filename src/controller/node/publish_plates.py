#!/usr/bin/python3
from __future__ import print_function

import roslib
roslib.load_manifest('controller')
import rospy
from std_msgs.msg import String
import numpy as np
import time
import sys

class score_tracker():
    def __init__(self):
        self.pub_plate = rospy.Publisher("/license_plate", String, queue_size=4)
        self.team_name = "test_team"
        self.team_password = "password"
        self.register_plate_location = "0"
        self.stop_plate_location = "-1"
        self.garbage_plate = "XXXX"
        # self.pub_clock = rospy.Subscriber("/clock", int)

        # time.sleep(1)
        # self.pub_plate.publish("test_team,password,0,XXXX")

    def send_plate(self, plate_location, plate_id):
        self.pub_plate.publish(f"{self.team_name},{self.team_password},{plate_location},{plate_id}")

def main(args):
    rospy.init_node('tracker', anonymous=False)
    tracker = score_tracker()

    time.sleep(1) # wait for node init
    tracker.send_plate(tracker.register_plate_location,tracker.garbage_plate) # register team
    time.sleep(1) # delay for testing purposes only
    tracker.send_plate(tracker.stop_plate_location,tracker.garbage_plate) # stop scoring and timer

    # uncomment the following try,except block if require node to persist until explicitly killed
    
    # try:
    #     rospy.spin() # node persists until explicitly shutdown
    # except KeyboardInterrupt:
    #     print("Shutting Down")

if __name__ == "__main__": # ros runs a node as main
    main(sys.argv)
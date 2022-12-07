#!/usr/bin/python3
from __future__ import print_function

import roslib
roslib.load_manifest('controller')
import rospy
from std_msgs.msg import String


global num_plates_captured
global stop_sim
global pub_plate
global team_name
global team_password
global register_plate_location
global stop_plate_location
global garbage_plate

num_plates_captured = 0
stop_sim = False
pub_plate = rospy.Publisher("/license_plate", String, queue_size=4)
team_name = "Alex and Mischa"
team_password = "password"
register_plate_location = "0"
stop_plate_location = "-1"
garbage_plate = "XXXX"

def send_plate(plate_location, plate_id):
    pub_plate.publish(f"{team_name},{team_password},{plate_location},{plate_id}")
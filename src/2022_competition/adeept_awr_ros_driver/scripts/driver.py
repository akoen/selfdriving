#!/usr/bin/env python

import RPi.GPIO as GPIO
import move
import rospy
import time
import ultra

from std_msgs.msg import String, Bool, Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range
from adeept_awr_ros_driver.msg import ArrayIR

class AdeeptAWRRos():

	def __init__(self):
		self._lin_vel = 0
		self._ang_vel = 0
		self._cmd_lin_vel = 0
		self._cmd_ang_vel = 0
		self._wheel_vel = (0,0)
		self._axle_length = 0.155
		self._wheel_radius = 0.070
		self._left_motor_dir = 1
		self._right_motor_dir = 0
		self._line_pin_right = 35
		self._line_pin_middle = 36
		self._line_pin_left = 38
		GPIO.setwarnings(False)
		GPIO.setmode(GPIO.BOARD)
		GPIO.setup(self._line_pin_right,GPIO.IN)
		GPIO.setup(self._line_pin_middle,GPIO.IN)
		GPIO.setup(self._line_pin_left,GPIO.IN)
		move.setup()
		move.motorStop()

		self._twist_sub = rospy.Subscriber('cmd_vel', Twist, self._twist_callback, queue_size=1)
		self._sonar_pub = rospy.Publisher('sonar', Range, queue_size = 1)
		self._IR_pub = rospy.Publisher('ir', ArrayIR, queue_size = 1)

	def _publish_ir(self):
		if self._IR_pub.get_num_connections() == 0:
			return
		now = rospy.Time.now()
		ir = ArrayIR()
		ir.header.stamp = now
		ir.left = GPIO.input(self._line_pin_left)
		ir.middle = GPIO.input(self._line_pin_middle)
		ir.right = GPIO.input(self._line_pin_right)
		self._IR_pub.publish(ir)

	def _publish_sonar(self):

		if self._sonar_pub.get_num_connections() == 0:
			return
		now = rospy.Time.now()
		sonar = Range()
		sonar.header.stamp =  now
		sonar.radiation_type = 0
		sonar.field_of_view = 0.2618
		sonar.min_range = 0.02
		sonar.max_range = 4.0
		sonar.range = ultra.checkdist()
		self._sonar_pub.publish(sonar)

	def _twist_callback(self, msg):
		self._cmd_lin_vel = msg.linear.x
		self._cmd_ang_vel = msg.angular.z

		print "x: " + str(self._cmd_lin_vel)
		print "yaw: " + str(self._cmd_ang_vel)

		# Disable both motors
		if self._cmd_lin_vel == 0 and self._cmd_ang_vel == 0:
			move.motorStop()
		else:
			# Forward driving
			if self._cmd_lin_vel > 0:
				self._left_motor_dir = 1
				self._right_motor_dir = 0
			# Reverse driving
			elif self._cmd_lin_vel < 0:
				self._left_motor_dir = 0
				self._right_motor_dir = 1
			# CCW Rotation
			elif self._cmd_ang_vel < 0:
				self._left_motor_dir = 0
				self._right_motor_dir = 0
			# CW Rotation
			elif self._cmd_ang_vel > 0:
				self._left_motor_dir = 1
				self._right_motor_dir = 1
			move.motor_left(1,self._left_motor_dir,100)
			move.motor_right(1,self._right_motor_dir,100)

	def run(self):
		rospy.init_node('adeept_awr_driver')
		rate = rospy.Rate(10)
		while not rospy.is_shutdown():
			self._publish_sonar()
			self._publish_ir()
			rate.sleep()
		move.destroy()

if __name__ == '__main__':
    try:
        awr = AdeeptAWRRos()
        awr.run()
    except rospy.ROSInterruptException as e:
        sys.exit('Connection Error: {}'.format(e))

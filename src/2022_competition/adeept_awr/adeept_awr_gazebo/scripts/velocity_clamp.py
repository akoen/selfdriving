#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

class FakeAdeeptAWR:

    def __init__(self, src_vel_topic, dst_vel_topic):

        self.__src_vel_topic = src_vel_topic
        self.__dst_vel_topic = dst_vel_topic

        if not self.__src_vel_topic:
          raise ValueError("source topic is an empty string")

        if not self.__dst_vel_topic:
          raise ValueError("dest topic is an empty string")

        self.vel_sub = rospy.Subscriber(self.__src_vel_topic, Twist, self.callback)
        self.vel_pub = rospy.Publisher(self.__dst_vel_topic, Twist, queue_size = 1)

    def callback(self, msg):
        lin = 0
        ang = 0

        # Forward driving
        if msg.linear.x > 0:
            lin = 0.271
        # Reverse driving
        elif msg.linear.x < 0:
            lin = -0.271
        # CW Rotation
        elif msg.angular.z < 0:
            ang = -1.8
        # CCW Rotation
        elif msg.angular.z > 0:
            ang = 1.8

        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = lin
        cmd_vel_msg.angular.z = ang

        self.vel_pub.publish(cmd_vel_msg)

if __name__ == '__main__':
    try:
        rospy.init_node('fake_adeept_awr_driver', anonymous=True)
        ad = FakeAdeeptAWR(rospy.get_param('~src_topic'),rospy.get_param('~dst_topic'))
        rate = rospy.Rate(rospy.get_param('~publish_rate', 10))
        while not rospy.is_shutdown():
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
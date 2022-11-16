#!/usr/bin/env python

import rospy
import tf2_ros
import tf_conversions
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import *

class GazeboLinkPose:
  link_name = ''
  frame_name = ''
  link_pose = PoseStamped()
  def __init__(self, link_name, frame_name):
    self.link_name = link_name
    self.frame_name = frame_name
    self.link_name_rectified = link_name.replace("::", "_")
    # print(self.link_name_rectified)
    # print(self.frame_name)
    # print(self.link_name_rectified.split('_')[-1])

    if not self.link_name:
      raise ValueError("'link_name' is an empty string")

    if not self.frame_name:
      raise ValueError("'frame_name' is an empty string")

    self.states_sub = rospy.Subscriber("/gazebo/link_states", LinkStates, self.callback)
    self.pose_pub = rospy.Publisher("/gazebo/" + self.link_name_rectified, PoseStamped, queue_size = 1)

  def callback(self, data):
    try:
      ind = data.name.index(self.link_name)
      self.link_pose.pose = data.pose[ind]
      self.link_pose.header.frame_id = self.frame_name
      br = tf2_ros.TransformBroadcaster()
      t = geometry_msgs.msg.TransformStamped()
      t.header.stamp = rospy.Time.now()
      t.header.frame_id = self.frame_name
      t.child_frame_id = self.link_name_rectified.split('_')[-1]
      t.transform.translation.x = self.link_pose.pose.position.x
      t.transform.translation.y = self.link_pose.pose.position.y
      t.transform.translation.z = self.link_pose.pose.position.z
      t.transform.rotation.x = self.link_pose.pose.orientation.x
      t.transform.rotation.y = self.link_pose.pose.orientation.y
      t.transform.rotation.z = self.link_pose.pose.orientation.z
      t.transform.rotation.w = self.link_pose.pose.orientation.w
      br.sendTransform(t)

    except ValueError:
      pass

if __name__ == '__main__':
  try:
    rospy.init_node('gazebo_link_pose', anonymous=True)
    gp = GazeboLinkPose(rospy.get_param('~link_name'),rospy.get_param('~frame_name'))
    publish_rate = rospy.get_param('~publish_rate', 10)

    rate = rospy.Rate(publish_rate)
    while not rospy.is_shutdown():
      gp.pose_pub.publish(gp.link_pose)
      rate.sleep()

  except rospy.ROSInterruptException:
    pass
cmake_minimum_required(VERSION 2.8.3)

project(enph353_utils)

find_package(catkin REQUIRED rospy std_msgs genmsg actionlib_msgs geometry_msgs)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

generate_messages(
  DEPENDENCIES actionlib_msgs std_msgs geometry_msgs
)

catkin_package(
	CATKIN_DEPENDS actionlib_msgs
)
# For source
SHELL := /bin/bash

.PHONY: files
all: files
	-killall rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python3
	source devel/setup.bash; \
	roslaunch src/controller/launch/master.launch

kill: files
	-killall rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python3
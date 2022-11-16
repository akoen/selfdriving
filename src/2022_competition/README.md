# 2022 UBC Parking competition

The repository contains the following ROS packages:

| Folders         | Description      |
|:--------------- |:---------------- |
| enph353_gazebo  | describes simulation world |
| enph353_npcs    | describes and controls pedestrian and Ford truck |
| enph353_utils   | contains competition startup scripts |
| adeept_awr      | describes and controls simulated Adeept AWR robot |
| adeept_awr_ros_driver | controls real world Adeept AWR robot |

## Installation instructions:
** Prerequisites: Ubuntu 20.04 with ROS Noetic installed **

* If you **do not** have a workspace already create one in your home directory.
```
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src
```

* Clone the repository into a catkin workspace src folder.
```
git clone https://github.com/ENPH353/2022_competition.git
```

* Build the packages
```
cd ~/ros_ws
catkin_make
```

* Source the environment
```
source devel/setup.bash
```

* Start the simulated world
```
cd src/2022_competition/enph353/enph353_utils/scripts
./run_sim.sh -vpg
```
The available options are:

| Option | Description      |
|:-------|:---------------- |
| -v     | spawn vehicle    |
| -p     | spawn pedestrian |
| -g     | generate new license plates |

* Start the score tracking app
Open a new tab in the current terminal window by pressing Ctrl+Shift+T 
The new terminal should already be in:
```
~/ros_ws/src/2022_competition/enph353/enph353_utils/scripts
```
Launch the score tracking app:
```
./score_tracker.py
```

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
    <img alt="Creative Commons Licence" style="border-width:0" 
        src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" />
</a><br />
This work is licensed under a 
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
    Creative Commons Attribution-ShareAlike 4.0 International License</a>.
#!/bin/bash

spawn_vehicles='false'
spawn_pedestrians='false'
generate_plates='false'
label_plates='false'

print_usage() {
  echo "Usage:"
  echo "-v to spawn vehicles"
  echo "-p to spawn pedestrians"
  echo "-g to generate new license plates"
  echo "-l to spawn QR code labels"
}

while getopts 'vpgl' flag; do
  case "${flag}" in
    v) spawn_vehicles='true' ;;
    p) spawn_pedestrians='true' ;;
    g) generate_plates='true' ;;
    l) label_plates='true' ;;
    *) print_usage
       exit 1 ;;
  esac
done

if $generate_plates = 'true'
then
	DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
	RELATIVE_PATH="/../../enph353_gazebo/scripts/plate_generator.py"
	FULL_PATH=".$RELATIVE_PATH"
	python3 $FULL_PATH
fi

if $label_plates = 'true'
then
  ln -sfn labelled ../../enph353_gazebo/media/materials/textures/license_plates
else
  ln -sfn unlabelled ../../enph353_gazebo/media/materials/textures/license_plates
fi

roslaunch enph353_utils sim.launch spawn_pedestrians:=$spawn_pedestrians spawn_vehicles:=$spawn_vehicles
#!/usr/bin/env python3
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import (QPixmap)
from PyQt5.QtCore import (Qt, QTimer, pyqtSignal)
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from python_qt_binding import loadUi

import csv
import os
import rospy
import sys

NUM_LOCATIONS = 8

class Window(QtWidgets.QMainWindow):
    message_received_signal = pyqtSignal(str)

    def __init__(self):
        super(Window, self).__init__()
        loadUi("./score_tracker.ui", self)

        pixmap = QPixmap('ENPH_vs_UBC_Parking.svg')
        self.label_QL.setPixmap(pixmap)

        # Populate log file name
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
        self.log_file_path = (self.team_ID_value_QL.text() + "_" + 
                              date_time + '.txt')
        self.log_file_value_QL.setText(self.log_file_path)

        # Populate tables
        LICENSE_PLATE_FILE = '/../../enph353_gazebo/scripts/plates.csv'
        SCRIPT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
        with open(SCRIPT_FILE_PATH + LICENSE_PLATE_FILE, "r") as plate_file:
            platereader = csv.reader(plate_file)
            i=0
            for row in platereader:
                if i < NUM_LOCATIONS:
                    self.license_scores_QTW.item(i, 1).setText(row[0])
                    self.log_msg("Position {}: {}".format(i+1, row[0]))
                else:
                    break
                i += 1

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.SLOT_timer_update)
        self.elapsed_time_s = 0

        # Initialize other variables:
        self.full_lap_points = 0

        self.first_cmd_vel = True

        # Connect widgets

        # Table values changed:
        self.license_scores_QTW.itemChanged.connect(self.SLOT_license_scores_changed)
        self.penalties_scores_QTW.itemChanged.connect(self.SLOT_penalties_changed)

        # Penalties deducted:
        self.penalty_vehicle_QPB.clicked.connect(self.SLOT_penalty_vehicle)
        self.penalty_pedestrian_QPB.clicked.connect(self.SLOT_penalty_pedestrian)
        self.penalty_track_QPB.clicked.connect(self.SLOT_penalty_track)

        self.lap_completed_QPB.clicked.connect(self.SLOT_lap_completed)
        self.manual_control_QPB.clicked.connect(self.SLOT_manual_control)

        self.message_received_signal.connect(self.SLOT_message_received)

        # Set-up ROS subscribers
        self.sub_license_plate = rospy.Subscriber("license_plate", String, 
                                                  self.licensePlate_callback)
        self.sub_cmd_vel = rospy.Subscriber("/R1/cmd_vel", Twist,
                                            self.cmd_vel_callback)
        rospy.init_node('competition_listener')

    def cmd_vel_callback(self, data):
        if self.first_cmd_vel:
            self.log_msg("First command velocity received.")
            self.first_cmd_vel = False

    def licensePlate_callback(self, data):
        self.message_received_signal.emit(str(data.data))


    def log_msg(self, message):
        now = datetime.now()
        date_time = now.strftime("%H:%M:%S.%f")[:-3]
        log_output = "<font color='blue'>{}</font>: {}".format(date_time, message)
        self.comms_log_QTE.append(log_output)
        # self.comms_log_QTE.insertHtml(log_output)

        log_file_content = self.comms_log_QTE.toPlainText()

        with open(self.log_file_path, "w") as html_file:
            html_file.write(log_file_content)


    def SLOT_lap_completed(self):
        if self.full_lap_points == 5:
            self.log_msg("Full lap completed already awarded points.")
            return
        self.log_msg("Full lap completed: +5 points")
        self.full_lap_points = 5
        self.update_license_total()


    def SLOT_license_scores_changed(self):
        self.update_license_total()


    def SLOT_manual_control(self):
        if (self.manual_control_QPB.isChecked()):
            self.log_msg("Manual control enabled.")
        else:
            self.log_msg("Manual control disabled.")


    def SLOT_message_received(self, license_string):
        self.log_msg("Message received: {}".format(license_string))

        teamID, teamPswd, plateLocation, plateID = str(license_string).split(',')
        
        # Use to start the timer and register the team name (not for points)
        if plateLocation == '0':
            # Update team ID and log file name:
            if teamID !=  self.team_ID_value_QL.text():
                now = datetime.now()
                date_time = now.strftime("%Y%m%d_%H%M%S")
                self.log_file_path = teamID + "_" + date_time + '.txt'
                self.log_file_value_QL.setText(self.log_file_path)

            self.team_ID_value_QL.setText(teamID)

            self.start_timer()
            return

        # Use to stop the timer
        if plateLocation == '-1':
            self.stop_timer()
            return

        if not plateLocation.isdigit():
            self.log_msg("Plate location is not a number.")
            return

        plateLocation = int(plateLocation)

        # Check out of bounds plate location
        if plateLocation < -1 or plateLocation > 8:
            self.log_msg("Invalid plate location: {}".format(plateLocation))
            return

        # Check submitted license plate ID and location against gnd truth:
        self.license_scores_QTW.blockSignals(True)
        self.license_scores_QTW.item(plateLocation-1, 2).setText(plateID)
        gndTruth = str(self.license_scores_QTW.item(plateLocation-1, 1).text())
        self.license_scores_QTW.blockSignals(False)

        manual_control_factor = 1
        if self.manual_control_QPB.isChecked():
            manual_control_factor = 0.5

        if gndTruth == plateID:
            # award 8 points for license plates on the inside track and 6 
            # points for the outside track
            points_awarded = int(6 * manual_control_factor)
            if plateLocation > 6:
                points_awarded = int(8 * manual_control_factor)

            self.log_msg("Awarded: {} pts".format(points_awarded))
            self.license_scores_QTW.item(plateLocation-1, 3).setText(str(points_awarded))
        else:
            self.log_msg("Awarded: {} pts".format(0))
            self.license_scores_QTW.item(plateLocation-1, 3).setText(str(0))


    def SLOT_penalties_changed(self):
        self.update_penalty_total()


    def SLOT_penalty_pedestrian(self):
        numEvents       = int(self.penalties_scores_QTW.item(1, 1).text()) + 1
        penaltyPerEvent = int(self.penalties_scores_QTW.item(1, 2).text())
        penaltyTotal    = numEvents * penaltyPerEvent
        self.log_msg("Penalty: pedestrian collision: {} pts".format(penaltyPerEvent))
        
        self.penalties_scores_QTW.item(1, 1).setText(str(numEvents))


    def SLOT_penalty_track(self):
        numEvents       = int(self.penalties_scores_QTW.item(2, 1).text()) + 1
        penaltyPerEvent = int(self.penalties_scores_QTW.item(2, 2).text())
        penaltyTotal    = numEvents * penaltyPerEvent
        self.log_msg("Penalty: track limit: {} pts".format(penaltyPerEvent))
        
        self.penalties_scores_QTW.item(2, 1).setText(str(numEvents))


    def SLOT_penalty_vehicle(self):
        numEvents       = int(self.penalties_scores_QTW.item(0, 1).text()) + 1
        penaltyPerEvent = int(self.penalties_scores_QTW.item(0, 2).text())
        penaltyTotal    = numEvents * penaltyPerEvent
        self.log_msg("Penalty: vehicle collision: {} pts".format(penaltyPerEvent))
        
        self.penalties_scores_QTW.item(0, 1).setText(str(numEvents))

    def SLOT_timer_update(self):
        ROUND_DURATION_s = 240
        self.elapsed_time_s += 1
        self.sim_current_time_s = rospy.get_time()
        sim_time_s = self.sim_current_time_s - self.sim_start_time_s
        self.elapsed_time_value_QL.setText(
            "{:03d} sec".format(int(sim_time_s)))
        if (sim_time_s > ROUND_DURATION_s):
            self.log_msg("Out of time: {}sec sim time (real time: {}sec).".
                format(sim_time_s, self.elapsed_time_s))
            self.timer.stop()

    def start_timer(self):
        self.elapsed_time_s = 0
        self.sim_start_time_s = rospy.get_time()
        self.elapsed_time_value_QL.setText(
            "{:03d} sec".format(self.elapsed_time_s))
        self.timer.start(1000)
        self.log_msg("Timer started.")

    def stop_timer(self):
        self.sim_current_time_s = rospy.get_time()
        sim_time_s = self.sim_current_time_s - self.sim_start_time_s
        self.log_msg("Timer stopped: {}sec sim time (real time: {}sec).".
                format(sim_time_s, self.elapsed_time_s))
        self.timer.stop()

    def update_license_total(self):
        licenseTotal = 0
        for i in range(NUM_LOCATIONS):
            licenseTotal += int(self.license_scores_QTW.item(i, 3).text())

        self.license_total_value_QL.setText(str(licenseTotal))

        penaltyTotal = int(self.penalties_total_value_QL.text())
        teamTotal = penaltyTotal + licenseTotal + self.full_lap_points
        self.total_score_value_QL.setText(str(teamTotal))
        self.log_msg("Team total: {} pts".format(str(teamTotal)))


    def update_penalty_total(self):
        self.penalties_scores_QTW.blockSignals(True)

        #update vehicle penalties total:
        numEvents         = int(self.penalties_scores_QTW.item(0, 1).text())
        penaltyPerEvent   = int(self.penalties_scores_QTW.item(0, 2).text())
        penaltyVehicle    = numEvents * penaltyPerEvent
        self.penalties_scores_QTW.item(0, 3).setText(str(penaltyVehicle))
        #update pedestrian penalties total:
        numEvents         = int(self.penalties_scores_QTW.item(1, 1).text())
        penaltyPerEvent   = int(self.penalties_scores_QTW.item(1, 2).text())
        penaltyPedestrian = numEvents * penaltyPerEvent
        self.penalties_scores_QTW.item(1, 3).setText(str(penaltyPedestrian))

        #update track penalties total
        numEvents         = int(self.penalties_scores_QTW.item(2, 1).text())
        penaltyPerEvent   = int(self.penalties_scores_QTW.item(2, 2).text())
        penaltyTrack      = numEvents * penaltyPerEvent
        self.penalties_scores_QTW.item(2, 3).setText(str(penaltyTrack))

        penaltyTotal = penaltyVehicle + penaltyPedestrian + penaltyTrack
        self.penalties_total_value_QL.setText(str(penaltyTotal))
        self.log_msg("Penalties total: {} pts".format(penaltyTotal))

        licenseTotal = int(self.license_total_value_QL.text())
        teamTotal = penaltyTotal + licenseTotal
        self.total_score_value_QL.setText(str(teamTotal))
        self.log_msg("Team total: {} pts".format(teamTotal))

        self.penalties_scores_QTW.blockSignals(False)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()

    sys.exit(app.exec_())
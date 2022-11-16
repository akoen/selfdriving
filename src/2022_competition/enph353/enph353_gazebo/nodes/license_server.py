#!/usr/bin/env python3

import rospy
import os
import csv
from random import shuffle, random
import datetime
from enph353_gazebo.srv import GetLegalPlates, GetLegalPlatesResponse, SubmitPlate, SubmitPlateResponse


class LicenseServer():

    def __init__(self):
        self.legal_plate_srv = rospy.Service('license_server', GetLegalPlates,
                                             self.serve_legal_plates)
        self.plate_check_srv = rospy.Service('license_verification',
                                             SubmitPlate, self.check_plates)
        self.file_path = os.path.dirname(os.path.realpath(__file__))
        self.legal_plates = []
        self.illegal_plates = []
        self.plate_location = []
        self.ids = {}
        with open(self.file_path+"/../scripts/plates.csv", "r") as plate_file:
            platereader = csv.reader(plate_file)
            for row in platereader:
                if random() > 0.4:
                    self.legal_plates.append(row[0])
                else:
                    self.illegal_plates.append(row[0])
                self.plate_location.append(row[0])

    def start_new_log_file(self, id):
        self.ids[id] = (self.file_path + "/" + id + "_" +
                        datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S") +
                        ".csv")
        with open(self.ids[id], "w") as log_file:
            reportwriter = csv.writer(log_file)
            reportwriter.writerow(["Time", "Location", "Plate", "Legal"])
            for i in range(len(self.plate_location)):
                plate = self.plate_location[i]
                reportwriter.writerow([0, i, plate, plate in self.legal_plates])

    def serve_legal_plates(self, req):
        if(req.id not in self.ids.keys()):
            self.start_new_log_file(req.id)
        shuffle(self.legal_plates)
        return GetLegalPlatesResponse(self.legal_plates)

    def check_plates(self, req):
        if(req.id not in self.ids.keys()):
            self.start_new_log_file(req.id)

        with open(self.ids[req.id], "a") as log_file:
            reportwriter = csv.writer(log_file)
            reportwriter.writerow([datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
                                   req.location, req.plate, req.legal])
        correct = False
        if (req.plate == self.plate_location[req.location]):
            if(req.legal):
                correct = req.plate in self.legal_plates
            else:
                correct = req.plate in self.illegal_plates
        return SubmitPlateResponse(correct)


if __name__ == '__main__':
    rospy.init_node('license_server')
    ls = LicenseServer()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

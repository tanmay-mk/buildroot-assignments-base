####################################################################################################
# @file name	: face_recog.py
# @Author	: Tanmay Kothale, Varun Mehta, Amey Dashaputre
# @Date	: 04/26/2022
# @References	: 1. https://github.com/ageitgey/face_recognition
#		  2. https://github.com/cu-ecen-aeld/final-project-saloni1307/blob/master/base_external/rootfs_overlay/etc/project/face_recog.py
# @Brief	: Facial Recognition Algorithm
####################################################################################################

##########################Importing libraries required for the program #############################
import face_recognition
import cv2
import numpy as np
from PIL import Image
#import serial
import os,time

#sim800l = serial.Serial('/dev/ttyAMA0', baudrate = 9600,timeout=1)

#previous ="Unknown"
#count=0

###########################code for face recognition started##############################

file = 'image.jpg'

# load the picture whose face has to be recognised.
elon_image = face_recognition.load_image_file("/etc/face-rec-sample/photos/elon.jpg")	#make changes here
elon_face_encoding = face_recognition.face_encodings(elon_image)[0]				#make changes here

bill_image = face_recognition.load_image_file("/etc/face-rec-sample/photos/bill.jpg")	#make changes here
bill_face_encoding = face_recognition.face_encodings(elon_image)[0]				#make changes here
	
# start your webcam
video_capture = cv2.VideoCapture(0)

# Create arrays of known face encodings and their names
known_face_encodings = [
    elon_face_encoding,
    bill_face_encoding
    ]
    
known_face_names = [
    "Elon Musk",
    "Bill Gates"
    ]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


################infinite loop to recognise face in the frame of the camera############
while True:
        
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    #Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            #if face encodings are matched with a known face, determine the face and store their name to a file
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_names.append(name)
                print(name)

    
video_capture.release()
cv2.destroyAllWindows()



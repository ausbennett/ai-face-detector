#Austin Bennett
#Followed tutorial: https://www.youtube.com/watch?v=XIrOM9oP3pA

#using opencv library
import cv2

#load pre-trained face data from opencv
face_data = cv2.CascadeClassifier('trained_data/haarcascade_frontalface_default.xml')

#Capture video from webcam
webcam = cv2.VideoCapture(0)

#Loop indefinetly 
while True:

    #read current frame
    successful_frame_read, frame = webcam.read()

    #convert image to gray scale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates = face_data.detectMultiScale(grayscaled_frame)

    #draw rectangles
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (176,243,224) , 2) 

    #output frame with rectangle drawn
    cv2.imshow('Face Detector', frame)

    #display frame for 1ms
    key = cv2.waitKey(1)
    
    ### QUIT using 'q' key
    if key==81 or key ==113:
        break

#release webcam
webcam.release()




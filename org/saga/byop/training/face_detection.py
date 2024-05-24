import dlib
import cv2
import matplotlib.pyplot as plt
from  blink_detection import  blink_detection
import config_manager
from org.saga.byop.training.face_landmarks_detection import face_landmarks_detection


class  face_detection:

    def face_detector(self,blink_detection=True,landmark_detection=True):
        # Open webcam
        cap = cv2.VideoCapture(0)
        detector = dlib.get_frontal_face_detector()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = detector(gray)
            if (blink_detection):
                blink = blink_detection(config_manager.get_face_detection_model_path());
            if (blink_detection):
                landmark = face_landmarks_detection(config_manager.get_face_detection_model_path());

            for face in faces:
                if (blink_detection):
                    blink.blink_detector(gray, face, frame)
                if(landmark_detection):
                    landmark.face_landmark_detector(gray,face,frame)

            # Display the frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # Break the loop if 'q' key is pressed
            if key == ord("q"):
                break


fd=face_detection()
fd.face_detector()
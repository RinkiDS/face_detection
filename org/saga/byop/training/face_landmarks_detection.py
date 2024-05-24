import cv2
import datetime
import dlib

class face_landmarks_detection:

    def __init__(self, face_detection_model_path):
        # Load face detector and predictor from dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(face_detection_model_path)

    def face_landmark_detector(self,gray,face,frame):
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # Draw a rectangle around the face (optional)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # Detect facial landmarks
        landmarks = self.predictor(gray, face)
        # Loop through each landmark point
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            # Draw a circle around each landmark point
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)


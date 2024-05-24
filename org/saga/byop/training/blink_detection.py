import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np


class blink_detection:

    def __init__(self,face_detection_model_path):
        # Load face detector and predictor from dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(face_detection_model_path)
        # Initialize counters for blink detection
        self.COUNTER = 0
        self.TOTAL = 0

    def blink_detector(self,gray,face,frame):
        landmarks = self.predictor(gray, face)
        points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 48)]
        left_eye = points[0:6]
        right_eye = points[6:12]

        # Compute eye aspect ratio for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the eye aspect ratio
        ear = (left_ear + right_ear) / 2.0

        # Check if the eye aspect ratio is below the threshold
        if ear < EYE_AR_THRESH:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>NO BLINK DETECTED>>>>>>>>>>>")
            self.COUNTER += 1
        else:
            if self.COUNTER >= EYE_AR_CONSEC_FRAMES:
                print(">>>>>>>>>>>>>>>> BLINK DETECTED>>>>>>>>>>>")
                self.TOTAL += 1
            self.COUNTER = 0

        # Draw eyes on the frame
        cv2.drawContours(frame, [cv2.convexHull(np.array(left_eye))], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(np.array(right_eye))], -1, (0, 255, 0), 1)


# Function to compute eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Constants for eye blink detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3


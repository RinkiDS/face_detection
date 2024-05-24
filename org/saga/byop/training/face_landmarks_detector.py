import dlib
import cv2
import matplotlib.pyplot as plt

# Open webcam
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

# Open webcam
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "D:\\San\\IIM\\SEM3\\GROUP-BYOP\\source\\esagav2\\eSAGA\\shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

        # Display the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Break the loop if 'q' key is pressed
        if key == ord("q"):
            break

        # Release the webcam and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
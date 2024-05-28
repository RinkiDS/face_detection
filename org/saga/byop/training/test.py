import cv2
import sys
from io import StringIO

# Function to redirect stdout to a variable
def redirect_stdout_to_variable():
    sys.stdout = StringIO()  # Redirect stdout to a StringIO object

# Function to display video frames and logs
def display_video_and_logs(video_path):
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow("Video and Logs", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video and Logs", 800, 600)  # Adjust window size as needed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Generate logs (example)
        logs = sys.stdout.getvalue()  # Get logs from redirected stdout
        sys.stdout.truncate(0)  # Clear the StringIO object

        # Concatenate video frame and logs vertically
        log_frame = cv2.putText(frame.copy(), logs, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display video frames and logs
        cv2.imshow("Video and Logs", log_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = "path_to_your_video.mp4"
redirect_stdout_to_variable()  # Redirect stdout to a variable
display_video_and_logs(video_path)

import cv2
import threading
import sys
import queue

# Function to capture frames from camera
def capture_frames(camera_index, frame_queue, stop_event):
    cap = cv2.VideoCapture(camera_index)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    cap.release()

# Function to capture system logs
def capture_logs(log_queue, stop_event):
    while not stop_event.is_set():
        logs = sys.stdout.readline()
        if logs:
            log_queue.put(logs)

# Function to display video frames and logs
def display_video_and_logs(camera_index=0):
    frame_queue = queue.Queue()
    log_queue = queue.Queue()
    stop_event = threading.Event()

    # Start threads for capturing frames and logs
    frame_thread = threading.Thread(target=capture_frames, args=(camera_index, frame_queue, stop_event))
    log_thread = threading.Thread(target=capture_logs, args=(log_queue, stop_event))
    frame_thread.start()
    log_thread.start()

    cv2.namedWindow("Video and Logs", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video and Logs", 800, 600)  # Adjust window size as needed

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            cv2.imshow("Video and Logs", frame)

        if not log_queue.empty():
            logs = log_queue.get()
            print(logs)  # Print logs to console
            # Optionally, overlay logs on the video frame using cv2.putText

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    # Join threads
    frame_thread.join()
    log_thread.join()
    cv2.destroyAllWindows()

# Example usage
display_video_and_logs()

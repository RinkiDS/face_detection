import cv2
import  os

class image_capturing:


    def start_capturing_images_from_vcam(image_path,f_count=10):
        time = 1000
        # def start_capturing_images_from_vcam(image_path,time):
        try:
            print("Starting The Video")
            video_capture = cv2.VideoCapture(0)
            print("Video Capture")
            frame_count = 0  # Counter to track the frames
            while True or frame_count <= f_count:
                ret, frame = video_capture.read()
                print(ret)
                print(frame)
                if not ret:
                    break

                # Save each frame to a file
                frame_count += 1
                # Generate a unique filename using timestamp and UUID
                # timestamp = int(time.time())
                # unique_id = str(uuid.uuid4())[:8]
                # Using the first 8 characters of a UUID
                # filename = f'frame_{timestamp}_{unique_id}.jpg'
                filename = f'frame_gaurav_{frame_count}.jpg'
                frame_name = os.path.join(image_path, filename)
                # frame_name = f'{save_path}/frame_{frame_count}.jpg'
                cv2.imwrite(frame_name, frame)

                # Display the captured frame (optional)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(time) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print("Error:", e)
        finally:
            video_capture.release()
            cv2.destroyAllWindows()

    def capture_images(num_images=5):
        # Initialize camera
        camera = cv2.VideoCapture(0)
        images = []
        for _ in range(num_images):
            ret, img = camera.read()
            if ret:
                images.append(img)
        camera.release()
        return images

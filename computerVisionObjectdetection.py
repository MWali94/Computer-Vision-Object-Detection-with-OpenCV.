import cv2
import sys
from cvlib.object_detection import detect_common_objects, draw_bbox

def main():
    # Video path
    video_path = "/Users/wali/Downloads/bazar3.mov"

    # Check if video file exists
    if not cv2.os.path.isfile(video_path):
        print(f"Error: The file '{video_path}' does not exist.")
        sys.exit()

    # Open video capture
    cap = cv2.VideoCapture(video_path)

    # Set to store all detected object labels
    detected_objects = set()

    # Main processing loop
    while True:
        # Read frame from video
        ret, frame = cap.read()

        # Break loop if end of video is reached
        if not ret:
            print("End of video reached. Exiting...")
            break

        # Resize frame
        frame = cv2.resize(frame, (800, 600))

        # Detect common objects using YOLOv3-tiny model
        bbox, label, conf = detect_common_objects(frame, model='yolov3-tiny')

        # Draw bounding boxes around detected objects
        frame = draw_bbox(frame, bbox, label, conf)

        # Update set with detected object labels
        detected_objects.update(label)

        # Display frame
        cv2.imshow("frame", frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Print all detected objects
    print("All objects detected in the video:")
    for obj in detected_objects:
        print(obj)

    # Release video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

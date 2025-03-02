import os
import cv2
import time
import numpy as np
from ultralytics import solutions
from yolo_detector import YoloDetector
from dpsort_tracker import DeepSortTracker

MODEL_PATH = r"C:\Users\bahaa\YOLO\yolo11x.pt"
VIDEOS_FOLDER = r"C:\DataSets\I5Videos\video"
OUTPUT_FOLDER = r"C:\Users\bahaa\YOLO\results"
Video1 = r"C:\DataSets\4K Road traffic video for object detection and tracking - free download now.mp4" # 4k Video (outside US)
Video2 = r"C:\DataSets\I5Videos\video\cctv052x2004080606x01827.avi" # Video from I5 dataset
Video3 = r"C:\DataSets\VecteezyTrafficFlow\vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4"

VIDEO_PATH = Video3

def main():

    roi_image = cv2.imread(r"C:\DataSets\RoadsCropped\Road_For_ROI_test.png")
    
    detector = YoloDetector(model_path = MODEL_PATH, confidence=0.7)
    tracker = DeepSortTracker()

    # Uncomment if looking at a single video instead of a file of videos
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Process each video in the file
    # for vid_file in os.listdir(VIDEOS_FOLDER):
    #     vid_path = os.path.join(VIDEOS_FOLDER, vid_file)
    #     cap = cv2.VideoCapture(vid_path)
    #     if not cap.isOpened():
    #         print("Error im Opening Video File")
    #         exit()
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format

    # output_file = os.path.join(OUTPUT_FOLDER, f"output_{os.path.splitext(vid_file)[0]}.mp4")
    output_file = os.path.join(OUTPUT_FOLDER, f"output_video3.mp4")
    
    #out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)

    # Initialize track history and speed calculator
    track_history = {}
    frame_count = 0
    prev_frame_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = time.perf_counter()
        frame_count += 1

        # Detection and tracking
        detections = detector.detect(frame)
        tracking_ids, boxes = tracker.track(detections, frame)

        # Process each detected object
        for tracking_id, box in zip(tracking_ids, boxes):
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Initialize new tracks
            if tracking_id not in track_history:
                track_history[tracking_id] = {
                    'positions': [],
                    'timestamps': [],
                    'speeds': []
                }

            # Store position and timestamp
            track_history[tracking_id]['positions'].append((center_x, center_y))
            track_history[tracking_id]['timestamps'].append(current_time)

            # Calculate instant speed (pixels/second)
            if len(track_history[tracking_id]['positions']) > 1:
                prev_pos = track_history[tracking_id]['positions'][-2]
                curr_pos = track_history[tracking_id]['positions'][-1]
                time_diff = current_time - track_history[tracking_id]['timestamps'][-2]
                
                if time_diff > 0:
                    dx = curr_pos[0] - prev_pos[0]
                    dy = curr_pos[1] - prev_pos[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    speed = distance / time_diff
                    track_history[tracking_id]['speeds'].append(speed)

                    # Display speed (last 5 average)
                    avg_speed = np.mean(track_history[tracking_id]['speeds'][-5:]) if len(track_history[tracking_id]['speeds']) > 0 else 0
                    cv2.putText(frame, f"{avg_speed:.1f} px/s", 
                               (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                               (0, 255, 0), 2)

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"ID: {tracking_id}", 
                       (x1, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 0, 255), 2)

        # Calculate processing FPS
        processing_fps = 1 / (current_time - prev_frame_time)
        prev_frame_time = current_time
        cv2.putText(frame, f"FPS: {processing_fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 255, 0), 2)

        # Display and output
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow("output", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



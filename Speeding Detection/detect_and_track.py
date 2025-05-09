import os
import cv2
import time
import numpy as np
from ultralytics import solutions
from yolo_detector import YoloDetector
from dpsort_tracker import DeepSortTracker
import calibration
from calibration import findROIUser
from calibration import CalculateHomography
from ultralytics import SAM

MODEL_PATH = r"C:\Users\bahaa\YOLO\yolo11x.pt"
VIDEOS_FOLDER = r"C:\DataSets\I5Videos\video"
OUTPUT_FOLDER = r"C:\Users\bahaa\YOLO\results"
Video1 = r"C:\DataSets\4K Road traffic video for object detection and tracking - free download now.mp4" # 4k Video (outside US)
Video2 = r"C:\DataSets\I5Videos\video\cctv052x2004080606x01827.avi" # Video from I5 dataset
Video3 = r"C:\DataSets\VecteezyTrafficFlow\vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4"
Video4 = r"C:\DataSets\SeattleStreetVideo.mp4" # Video from downtown seattle - car movement perpendicular to camera lens
Video5 = r"C:\DataSets\Seattle Crash Video Editted.mp4"

VIDEO_PATH = Video5 # Change this to the path of your video file
# H = np.array([ [-5.3872, -4.6759, 2848.4],
#                [0.0959, -5.69, 1378.9],
#                [0.00014702, -0.0098528, 1] ])

def apply_homography_point(point, H):
    px = np.array([[point]], dtype=np.float32)  # Shape: (1, 1, 2)
    transformed = cv2.perspectiveTransform(px, H)
    return tuple(transformed[0][0])

def main():
    
    # Uncomment if looking at a single video instead of a file of videos
    cap = cv2.VideoCapture(VIDEO_PATH)

    _ , roi_image = cap.read() # Extract a snapshot for road detection
    roi_image = cv2.resize(roi_image, (960, 540))
    roi_points, roi_mask = findROIUser(roi_image)
    H, _ = CalculateHomography(roi_image, roi_points)
    
    detector = YoloDetector(model_path = MODEL_PATH, confidence=0.7)
    tracker = DeepSortTracker()
    sam_model = SAM("sam2.1_b.pt")

    
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
        h, w = frame.shape[:2]

        if not ret:
            break
            
        current_time = time.perf_counter()
        frame_count += 1

        # Detection and tracking
        detections = detector.detect(frame)
        tracking_ids, boxes = tracker.track(detections, frame)

        # Process each detected vehicle
        for tracking_id, box in zip(tracking_ids, boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Check if bounding boxes are valid
            if x1 >= x2 or y1 >= y2:
                continue

            sam_result = sam_model.predict(frame, bboxes=[x1, y1, x2, y2])

            if not sam_result or not sam_result[0].masks:
                continue

            sam_mask = sam_result[0].masks.data[0].cpu().numpy().astype(bool)


            contour_groups = sam_result[0].masks.xy
            if contour_groups:
                for contour_group in contour_groups:
                    for contour in contour_group:
                        contour = np.array(contour, dtype=np.int32)
                        
                        # if contour.ndim != 2 or contour.shape[0] < 3:
                        #     continue

                        contour = contour.reshape((-1, 1, 2))
                        cv2.polylines(frame, [contour], isClosed=True, color=(0, 255, 0), thickness=2)

            ys, xs = np.where(sam_mask==1)
            points  = np.column_stack((xs, ys))

            botom_threshold = max(y1, y2) - int(1 * max(y1,y2)/frame.shape[1] * (max(y1,y2) - min(y1,y2)) )

            threshold_points = points[points[:, 1] > botom_threshold]
            
            if len(threshold_points) > 1:
                min_y = np.min(threshold_points[:, 1])
                max_y = np.max(threshold_points[:, 1])
                pixel_length = max_y - min_y

                rear_x = int(np.max(threshold_points[:, 0]))
                rear_y = min_y

                front_x = max(x1,x2)
                front_y = max(y1,y2)

                front_point = (front_x, front_y)  # (y, x) for OpenCV
                rear_point = (rear_x, rear_y)  # (y, x) for OpenCV

                rear_point_bev = apply_homography_point(rear_point, H)
                front_point_bev = apply_homography_point(front_point, H)

                px_vehicle_length = abs(rear_point_bev[1] - front_point_bev[1])

                # visualization
                cv2.circle(frame, rear_point, 5, (0, 0, 255), -1)
                cv2.circle(frame, front_point, 5, (0, 255, 0), -1)
                cv2.line(frame, rear_point, front_point, (255, 0, 0), 2)

                # Create text to overlay
                length_text = f"Len: {px_vehicle_length:.1f}px"

                # Draw the text slightly above the bounding box
                cv2.putText(
                    frame, 
                    length_text, 
                    (x1, y1 - 10),  # Just above the top-left corner of the box
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 0), 2
                )

            colored_mask = frame.copy()
            colored_mask[sam_mask] = [0, 0, 255]  # Red color for the mask

            # Draw sam mask, bounding box, and ID
            #frame[sam_mask] = cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)[sam_mask]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"ID: {tracking_id}", 
                       (x1, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 0, 255), 2)
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Initialize new tracks
            if tracking_id not in track_history:
                track_history[tracking_id] = {
                    'positions': [],
                    'timestamps': [],
                    'speeds': [],
                    'mask': [],
                    'pixel_length': [],
                }

            # Store position and timestamp
            track_history[tracking_id]['positions'].append((center_x, center_y))
            track_history[tracking_id]['timestamps'].append(current_time)
            track_history[tracking_id]['pixel_length'].append(px_vehicle_length)
            track_history[tracking_id]['mask'].append(sam_mask)


        # Calculate processing FPS
        processing_fps = 1 / (current_time - prev_frame_time)
        prev_frame_time = current_time
        cv2.putText(frame, f"FPS: {processing_fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 255, 0), 2)

        # Display and output
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow("output_normal", display_frame)
        display_frame = calibration.BEVTransform(display_frame,H)
        cv2.imshow("output_BEV", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



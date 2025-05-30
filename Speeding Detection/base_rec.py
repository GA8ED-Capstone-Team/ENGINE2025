import os
import sys
import cv2
import numpy as np
import torch
from ultralytics import SAM
from yolo_detector import YoloDetector
from dpsort_tracker import DeepSortTracker
from rear_point import get_vehicle_base_contour, get_vehicle_front_point
from test_atucalib import compute_pose_3d
from collections import defaultdict
import matplotlib.pyplot as plt

def moving_average_filter(speeds, window=5):
    if not speeds:
        return None
    speeds = np.array(speeds[-window:])
    return np.mean(speeds)

def filter_calibrations(calibs, orientation_thresh=0.7, z_range=(0, 700)):
    filtered = []
    for frame_idx, R, t in calibs:
        z_axis = R[:, 2]
        if abs(z_axis[1]) > orientation_thresh:
            continue
        if not (z_range[0] < t[2] < z_range[1]):
            continue
        filtered.append((frame_idx, R, t))
    return filtered

from sklearn.linear_model import RANSACRegressor

def ransac_filter_translations(calibs, min_samples=2, residual_threshold=1.0):
    if len(calibs) < min_samples:
        return []

    frames = np.array([f for f, _, _ in calibs])
    translations = np.array([t.ravel() for _, _, t in calibs])  # (N, 3)

    # Fit RANSAC model on each axis separately
    inlier_mask = np.ones(len(calibs), dtype=bool)
    for axis in [2]:
        ransac = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold)
        try:
            ransac.fit(frames.reshape(-1, 1), translations[:, axis])
            axis_inliers = ransac.inlier_mask_
            inlier_mask &= axis_inliers
        except Exception as e:
            print(f"RANSAC fitting error on axis {axis}: {e}")
            return []

    return [calibs[i] for i in range(len(calibs)) if inlier_mask[i]]


def main():
    MODEL_PATH = r"C:\Users\bahaa\YOLO\yolo11x.pt"
    VIDEO_PATH = r"C:\Users\bahaa\Downloads\Untitled video - Made with Clipchamp (3).mp4"
    intrinsics_file = r"C:\Users\bahaa\CapstoneProject\ENGINE2025\Speeding Detection\Camera_Calibration\camera_intrinsics.npz"
    intrinsics_data = np.load(intrinsics_file)
    intrinsic_matrix = intrinsics_data["mtx"]
    dist_coeffs = intrinsics_data["dist"]

    W, L = 1.82, 4.8
    world_coords = np.array([(0, 0, 0), (0, L, 0), (W, L, 0), (W, 0, 0)], dtype=np.float64)

    cap = cv2.VideoCapture(VIDEO_PATH)
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.7)
    tracker = DeepSortTracker()
    sam_model = SAM("sam2.1_s.pt")
    sam_model.model.to('cuda')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output_vehicle_detection.mp4", fourcc, fps, (width, height))

    N = 2  # window size
    B = 1   # min frame gap
    vehicle_history = defaultdict(list)
    vehicle_calibrations = defaultdict(list)
    vehicle_speeds = defaultdict(list)
    speed_time_series = defaultdict(list)

    frame_idx = 0
    s, b = 38, 0
    frame_number = (fps * s)  + b
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 3 != 0:
            frame_idx += 1
            continue

        detections = detector.detect(frame)
        tracking_ids, boxes = tracker.track(detections, frame)

        for tracking_id, box in zip(tracking_ids, boxes):
            x1, y1, x2, y2 = map(int, box)
            x1, y1, x2, y2 = max(0, x1-5), max(0, y1 - 10), min(width, x2+5), min(height, y2 + 10)
            if x1 >= x2 or y1 >= y2:
                continue

            sam_result = sam_model.predict(frame, bboxes=[x1, y1, x2, y2])
            if not sam_result or not sam_result[0].masks:
                continue
            mask = sam_result[0].masks.data[0].cpu().numpy().astype(bool)
            corners = get_vehicle_base_contour(mask, (x1, y1, x2, y2), max_iters=20, debug=False)
            if corners is None or len(corners) != 4:
                continue
            # Visualize rectangle corners and edges
            for i, pt in enumerate(corners):
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 7, (0, 255, 255), -1)
                cv2.putText(frame, f"V{i+1}", (int(pt[0])+5, int(pt[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            for i in range(4):
                pt1 = tuple(map(int, corners[i]))
                pt2 = tuple(map(int, corners[(i+1)%4]))
                cv2.line(frame, pt1, pt2, (255, 0, 255), 2)
            
            # front_point = get_vehicle_front_point(frame, corners, (x1, y1, x2, y2), debug=False)
            c1, c2 = corners[0], corners[3]  # Use the first two corners as front points
            front_points = [(c1[0], c1[1]-30), (c2[0], c2[1]-30)]
            if front_points is not None:
                for front_point in front_points:
                    cv2.circle(frame, (int(front_point[0]), int(front_point[1])), 7, (255, 0, 0), -1)
                world_coords_ext = np.array([
                    (0, 0, 0), 
                    (0, L, 0), 
                    (W, L, 0), 
                    (W, 0, 0), 
                    (0, 0, -1), # Use float division for correct type
                    (W, 0, -1)
                ], dtype=np.float64)
                corners_ext = np.vstack([corners, np.array(front_points, dtype=np.float64)])
            else:
                world_coords_ext = world_coords
                corners_ext = np.array(corners, dtype=np.float64)

            # world_coords_ext = np.array(world_coords, dtype=np.float64)
            # corners_ext = np.array(corners, dtype=np.float64)
            
            vehicle_history[tracking_id].append((frame_idx, corners_ext))
            try:
                R, t = compute_pose_3d(np.array(corners_ext, dtype=np.float64), world_coords_ext, intrinsic_matrix, dist_coeffs)
                vehicle_calibrations[tracking_id].append((frame_idx, R, t))
            except Exception as e:
                print(f"Pose estimation error: {e}")
                continue

            filtered = ransac_filter_translations(vehicle_calibrations[tracking_id][-N:])
            for i in range(len(filtered)):
                for j in range(i + 1, len(filtered)):
                    f1, _, t1 = filtered[i]
                    f2, _, t2 = filtered[j]
                    if f2 - f1 >= B:
                        delta = t2[:2].ravel() - t1[:2].ravel()
                        displacement = np.linalg.norm(t2 - t1)
                        time_elapsed = (f2 - f1) / fps
                        speed_mph = (displacement / time_elapsed) * 3.6 / 1.60934 if time_elapsed > 0 else 0
                        if 0 < speed_mph < 200:
                            vehicle_speeds[tracking_id].append(speed_mph)
                            speed_time_series[tracking_id].append((frame_idx, speed_mph))
                        break

            if vehicle_speeds[tracking_id]:
                filtered_speed = moving_average_filter(vehicle_speeds[tracking_id], window=3)
                if filtered_speed:
                    cv2.putText(
                        frame, f"Speed: {filtered_speed:.1f} mph", (x1, y2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"ID: {tracking_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        out.write(frame)
        cv2.imshow("Vehicle Detection + Speed", frame)
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
        # --- RANSAC filtering on all speeds for each ID and output a single speed ---
    print("\n=== FINAL VEHICLE SPEEDS (RANSAC filtered, one per ID) ===")
    from sklearn.linear_model import RANSACRegressor

    def ransac_single_speed(frames, speeds):
        if len(speeds) < 2:
            return np.mean(speeds) if speeds else None
        frames = np.array(frames).reshape(-1, 1)
        speeds = np.array(speeds)
        ransac = RANSACRegressor(min_samples=2, residual_threshold=5.0)
        try:
            ransac.fit(frames, speeds)
            inlier_mask = ransac.inlier_mask_
            inlier_speeds = speeds[inlier_mask]
            return np.mean(inlier_speeds)
        except Exception as e:
            print(f"RANSAC error: {e}")
            return np.mean(speeds)

    plt.figure(figsize=(12, 6))
    for tracking_id, series in speed_time_series.items():
        if len(series) < 2:
            continue
        series = sorted(series)
        frames, speeds = zip(*series)
        plt.plot(frames, speeds, label=f"ID {tracking_id}")

        # RANSAC filtering for a single speed value
        single_speed = ransac_single_speed(frames, speeds)
        print(f"ID {tracking_id}: RANSAC-filtered mean speed = {single_speed:.2f} mph")

    plt.title("Vehicle Speed vs Time (Frame Index)")
    plt.xlabel("Frame Index")
    plt.ylabel("Speed (mph)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("vehicle_speed_curves.png")
    plt.show()

if __name__ == "__main__":
    main()




# import os
# import sys
# import cv2
# import numpy as np
# import torch
# from ultralytics import SAM
# from yolo_detector import YoloDetector
# from dpsort_tracker import DeepSortTracker
# from rear_point import get_bottom_rectangle_corners, get_vehicle_base_contour
# from test_atucalib import compute_pose_3d
# from collections import defaultdict
# import matplotlib.pyplot as plt


# # --- Paths and Model Setup ---
# MODEL_PATH = r"C:\Users\bahaa\YOLO\yolo11x.pt"
# Video7 = r"C:\Users\bahaa\Downloads\Untitled video - Made with Clipchamp (3).mp4"
# VIDEO_PATH = Video7

# # --- Camera Intrinsics ---
# intrinsics_file = r"C:\Users\bahaa\CapstoneProject\ENGINE2025\Speeding Detection\Camera_Calibration\camera_intrinsics.npz"
# intrinsics_data = np.load(intrinsics_file)
# intrinsic_matrix = intrinsics_data["mtx"]
# dist_coeffs = intrinsics_data["dist"]
# # dist_coeffs = np.zeros((4, 1))  # Uncomment if needed

# # --- Vehicle dimensions (meters) ---
# W, L = 1.82, 4.0
# world_coords = np.array([(0, 0, 0), (0, L, 0), (W, L, 0), (W, 0, 0)], dtype=np.float64)

# def crop_car_image(frame, bbox):
#     x1, y1, x2, y2 = map(int, bbox)
#     return frame[y1:y2, x1:x2]

# def ransac_filter(speeds, threshold=5.0):
#     """Simple RANSAC-like filter: returns median of inliers within threshold of median."""
#     if not speeds:
#         return None
#     speeds = np.array(speeds)
#     median = np.median(speeds)
#     inliers = speeds[np.abs(speeds - median) < threshold]
#     if len(inliers) == 0:
#         return median
#     return np.median(inliers)

# def moving_average_filter(speeds, window=5):
#     """Returns the moving average of the last `window` speeds."""
#     if not speeds:
#         return None
#     speeds = np.array(speeds[-window:])
#     return np.mean(speeds)

# def main():
#     cap = cv2.VideoCapture(VIDEO_PATH)
#     detector = YoloDetector(model_path=MODEL_PATH, confidence=0.7)
#     tracker = DeepSortTracker()
#     sam_model = SAM("sam2.1_s.pt")
#     sam_model.model.to('cuda')

#     # --- Video Writer Setup ---
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out_path = "output_vehicle_detection.mp4"
#     fps = 30
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

#     # --- For speed estimation ---
#     frame_gap = 5
#     vehicle_history = defaultdict(list)  # tracking_id: list of (frame_idx, corners)
#     vehicle_speeds = defaultdict(list)   # tracking_id: list of speed estimates
#     speed_time_series = defaultdict(list)  # list of (frame_idx, speed)


#     frame_idx = 0
#     s, b = 0, 0
#     frame_number = (fps * s)  + b
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         h, w = frame.shape[:2]

#         detections = detector.detect(frame)
#         tracking_ids, boxes = tracker.track(detections, frame)

#         for tracking_id, box in zip(tracking_ids, boxes):
#             x1, y1, x2, y2 = map(int, box)
#             if x1 >= x2 or y1 >= y2:
#                 continue

#             # Run SAM segmentation for this bbox
#             sam_result = sam_model.predict(frame, bboxes=[x1, y1, x2, y2])
#             if not sam_result or not sam_result[0].masks:
#                 continue
#             mask = sam_result[0].masks.data[0].cpu().numpy().astype(bool)

#             # Get rectangle corners (global coordinates)
#             corners = get_bottom_rectangle_corners(frame, mask, (x1, y1, x2, y2), plot=False)
#             corners = get_vehicle_base_contour(mask, (x1, y1, x2, y2), max_iters=20, debug=True)
#             if corners is None or len(corners) != 4:
#                 continue

#             # Store corners for speed estimation
#             vehicle_history[tracking_id].append((frame_idx, corners))

#             # Draw bbox and ID
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             cv2.putText(frame, f"ID: {tracking_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#             # Visualize rectangle corners and edges
#             for i, pt in enumerate(corners):
#                 cv2.circle(frame, (int(pt[0]), int(pt[1])), 7, (0, 255, 255), -1)
#                 cv2.putText(frame, f"V{i+1}", (int(pt[0])+5, int(pt[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
#             for i in range(4):
#                 pt1 = tuple(map(int, corners[i]))
#                 pt2 = tuple(map(int, corners[(i+1)%4]))
#                 cv2.line(frame, pt1, pt2, (255, 0, 255), 2)

#             # --- Speed estimation every frame_gap frames ---
#             history = vehicle_history[tracking_id]
#             if len(history) >= frame_gap + 1:
#                 # Find two entries separated by at least frame_gap
#                 idx1, c1 = history[-frame_gap-1]
#                 idx2, c2 = history[-1]
#                 # if idx2 - idx1 == frame_gap:
#                 coords_1 = np.array(c1, dtype=np.float64)
#                 coords_2 = np.array(c2, dtype=np.float64)
#                 try:
#                     R1, t1 = compute_pose_3d(coords_1, world_coords, intrinsic_matrix, dist_coeffs)
#                     R2, t2 = compute_pose_3d(coords_2, world_coords, intrinsic_matrix, dist_coeffs)
#                     displacement = np.linalg.norm(t2 - t1)
#                     time_elapsed = (idx2 - idx1) / fps
#                     speed_m_s = displacement / time_elapsed if time_elapsed > 0 else 0
#                     speed_kmh = speed_m_s * 3.6
#                     speed_mph = speed_kmh / 1.60934
#                     # --- Reject outrageous displacements/speeds ---
#                     MAX_REASONABLE_SPEED_MPH = 200  # adjust as needed
#                     MAX_REASONABLE_DISPLACEMENT = 50  # meters, adjust as needed
#                     if (
#                         speed_mph < 0 or speed_mph > MAX_REASONABLE_SPEED_MPH
#                         or displacement > MAX_REASONABLE_DISPLACEMENT
#                     ):
#                         print(f"Rejected outlier speed/displacement for ID {tracking_id}: speed={speed_mph:.1f} mph, disp={displacement:.2f} m")
#                         continue
#                     vehicle_speeds[tracking_id].append(speed_mph)
#                     speed_time_series[tracking_id].append((frame_idx, speed_mph))
#                     print(f"ID {tracking_id} | Δframe={idx2 - idx1} | Δt={time_elapsed:.2f}s | speed={speed_mph:.1f} mph")
#                 except Exception as e:
#                     print(f"Speed estimation error for ID {tracking_id}: {e}")

#             # --- Display filtered speed ---
#             if vehicle_speeds[tracking_id]:
#                 filtered_speed = moving_average_filter(vehicle_speeds[tracking_id], window=3)
#                 if filtered_speed is not None:
#                     cv2.putText(
#                         frame,
#                         f"Speed: {filtered_speed:.1f} mph",
#                         (x1, y2 + 30),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.8,
#                         (0, 200, 0),
#                         2,
#                     )

#         out.write(frame)  # Save frame to video
#         cv2.imshow("Vehicle Detection + Bottom Rectangle + Speed", frame)
#         frame_idx += 1
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q") or key == 27:
#             break

#     cap.release()
#     out.release()  # Release the video writer
#     cv2.destroyAllWindows()
#     plt.figure(figsize=(12, 6))
#     for tracking_id, series in speed_time_series.items():
#         if len(series) < 2:
#             continue  # skip short tracks
#         series = sorted(series)  # sort by frame_idx
#         frames, speeds = zip(*series)
#         plt.plot(frames, speeds, label=f"ID {tracking_id}")

#     plt.title("Vehicle Speed vs Time (Frame Index)")
#     plt.xlabel("Frame Index")
#     plt.ylabel("Speed (mph)")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("vehicle_speed_curves.png")
#     plt.show()

# if __name__ == "__main__":
#     main()
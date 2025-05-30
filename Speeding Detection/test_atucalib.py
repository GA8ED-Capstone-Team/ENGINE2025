import cv2
import numpy as np
import os

# === CONFIGURATION ===
video_path = r"C:\Users\bahaa\Downloads\Untitled video - Made with Clipchamp (1).mp4"  # Path to your video
car_dimensions = {
    "length": 4.815,  # meters (front to back)
    "width": 1.48    # meters (left to right)
}
frame_gap = 10

# Load intrinsics from file
intrinsics_file = r"C:\Users\bahaa\CapstoneProject\ENGINE2025\Speeding Detection\Camera_Calibration\camera_intrinsics.npz"  # Path to your saved intrinsics
intrinsics_data = np.load(intrinsics_file)
intrinsic_matrix = intrinsics_data["mtx"]
dist_coeffs = intrinsics_data["dist"]

dist_coeffs = np.zeros((4, 1))  # Change if you have actual distortion

# === FUNCTIONS ===
def get_frame_at_index(video, idx):
    video.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = video.read()
    if not ret:
        raise ValueError(f"Could not read frame at index {idx}")
    return frame

def select_points(image, window_name="Select 4 points"):
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, image)

    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, click_event)
    print(f"Click on 4 vehicle corners in {window_name}")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.float64)

def compute_pose_3d(image_points, object_points, camera_matrix, dist_coeffs):
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        raise RuntimeError("SolvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", tvec)
    return R, tvec

if __name__ == "__main__":
    # === MAIN ===
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    mid_frame = frame_count // 2 + 30
    frame1 = get_frame_at_index(cap, mid_frame)
    frame2 = get_frame_at_index(cap, mid_frame + frame_gap)
    cap.release()

    cv2.imwrite("frame1.jpg", frame1)
    cv2.imwrite("frame2.jpg", frame2)

    # Let user select points
    img_pts1 = select_points(frame1.copy(), "Frame 1")
    img_pts2 = select_points(frame2.copy(), "Frame 2")

    # Define real-world 3D coordinates of the 4 keypoints (e.g., rear left, rear right, front right, front left)
    # Assume car lies on X (width) and Z (length) plane, Y is up (height ignored)
    obj_pts = np.array([
        [-car_dimensions["width"]/2, 0, 0],                      # Rear Left
        [1.55, 0.1, 3.02],                      # License plate center
        [ car_dimensions["width"]/2, 0, car_dimensions["length"]],  # Front Right
        [-car_dimensions["width"]/2, 0, car_dimensions["length"]]   # Front Left
    ], dtype=np.float64)

    # Solve for camera pose in both frames
    R1, t1 = compute_pose_3d(img_pts1, obj_pts, intrinsic_matrix, dist_coeffs)
    R2, t2 = compute_pose_3d(img_pts2, obj_pts, intrinsic_matrix, dist_coeffs)

    # Compute Euclidean displacement between translation vectors
    displacement = np.linalg.norm(t2 - t1)  # in meters
    time_elapsed = frame_gap / fps          # in seconds
    speed_m_s = displacement / time_elapsed
    speed_kmh = speed_m_s * 3.6
    speed_mph = speed_kmh / 1.60934

    # Output results
    print(f"Estimated vehicle displacement: {displacement:.2f} meters")
    print(f"Time elapsed: {time_elapsed:.2f} seconds")
    print(f"Estimated speed: {speed_m_s:.2f} m/s ({speed_mph:.2f} mph)")

# import cv2
# import numpy as np
# import os

# # --- Add YOLO and DeepSORT direct usage imports ---
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort

# # === CONFIGURATION ===
# video_path = r"C:\Users\bahaa\Downloads\Untitled video - Made with Clipchamp (1).mp4"  # Path to your video
# car_dimensions = {
#     "length": 4.815,  # meters (front to back)
#     "width": 1.48    # meters (left to right)
# }
# frame_gap = 10

# # Load intrinsics from file
# intrinsics_file = r"C:\Users\bahaa\CapstoneProject\ENGINE2025\Speeding Detection\Camera_Calibration\camera_intrinsics.npz"  # Path to your saved intrinsics
# intrinsics_data = np.load(intrinsics_file)
# intrinsic_matrix = intrinsics_data["mtx"]
# dist_coeffs = intrinsics_data["dist"]

# dist_coeffs = np.zeros((4, 1))  # Change if you have actual distortion

# def get_frame_at_index(video, idx):
#     video.set(cv2.CAP_PROP_POS_FRAMES, idx)
#     ret, frame = video.read()
#     if not ret:
#         raise ValueError(f"Could not read frame at index {idx}")
#     return frame

# def crop_bbox(image, bbox):
#     x1, y1, x2, y2 = map(int, bbox)
#     return image[y1:y2, x1:x2]

# def select_points(image, window_name="Select 4 points"):
#     points = []

#     def click_event(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
#             points.append((x, y))
#             cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
#             cv2.imshow(window_name, image)

#     cv2.imshow(window_name, image)
#     cv2.setMouseCallback(window_name, click_event)
#     print(f"Click on 4 vehicle corners in {window_name}")
#     cv2.waitKey(0)
#     cv2.destroyWindow(window_name)
#     return np.array(points, dtype=np.float64)

# def compute_pose_3d(image_points, object_points, camera_matrix, dist_coeffs):
#     success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
#     if not success:
#         raise RuntimeError("SolvePnP failed")
#     R, _ = cv2.Rodrigues(rvec)
#     return R, tvec

# def get_yolo_deepsort_bbox(frame, yolo_model, deepsort_tracker):
#     """
#     Detects vehicles in a frame using YOLO and DeepSORT directly (not via wrapper classes).
#     Returns the first detected bounding box and its tracking ID.
#     """
#     results = yolo_model(frame)
#     dets = results[0].boxes.xyxy.cpu().numpy()
#     confs = results[0].boxes.conf.cpu().numpy()
#     classes = results[0].boxes.cls.cpu().numpy()

#     # Filter for vehicles (COCO: car=2, truck=7, bus=5, motorcycle=3, etc.)
#     vehicle_classes = [2, 3, 5, 7]
#     mask = np.isin(classes, vehicle_classes) & (confs > 0.5)
#     filtered_dets = dets[mask]
#     filtered_confs = confs[mask]
#     filtered_classes = classes[mask]

#     if len(filtered_dets) == 0:
#         return None, None

#     # Prepare detections for deep_sort_realtime: [ [x1, y1, x2, y2], confidence, class_id ]
#     detections = []
#     for i, det in enumerate(filtered_dets):
#         x1, y1, x2, y2 = det[:4]
#         conf = float(filtered_confs[i])
#         class_id = int(filtered_classes[i])
#         detections.append([[x1, y1, x2, y2], conf, class_id])

#     # Update tracker
#     tracks = deepsort_tracker.update_tracks(detections, frame=frame)
#     # tracks is a list of Track objects

#     # Find the first confirmed track with a bbox
#     for track in tracks:
#         if not track.is_confirmed():
#             continue
#         track_id = track.track_id
#         ltrb = track.to_ltrb()  # [left, top, right, bottom]
#         return ltrb, track_id

#     return None, None

# # === MAIN ===
# # --- Load YOLO and DeepSORT directly ---
# yolo_model = YOLO(r"C:\Users\bahaa\YOLO\yolo11x.pt")
# deepsort_tracker = DeepSort(max_age=30)

# cap = cv2.VideoCapture(video_path)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = cap.get(cv2.CAP_PROP_FPS)

# mid_frame = 130
# frame1 = get_frame_at_index(cap, mid_frame)
# frame2 = get_frame_at_index(cap, mid_frame + frame_gap)

# # --- Detection and tracking on both frames using direct YOLO + DeepSORT ---
# bbox1, track_id1 = get_yolo_deepsort_bbox(frame1, yolo_model, deepsort_tracker)
# bbox2, track_id2 = get_yolo_deepsort_bbox(frame2, yolo_model, deepsort_tracker)

# cap.release()

# if bbox1 is None or bbox2 is None:
#     raise RuntimeError("No objects detected/tracked in one of the frames.")

# crop1 = crop_bbox(frame1, bbox1)
# crop2 = crop_bbox(frame2, bbox2)

# cv2.imwrite("crop1.jpg", crop1)
# cv2.imwrite("crop2.jpg", crop2)

# # Let user select points on the cropped images
# img_pts1 = select_points(crop1.copy(), "Crop 1")
# img_pts2 = select_points(crop2.copy(), "Crop 2")

# # Adjust points to original image coordinates
# x1a, y1a, x2a, y2a = map(int, bbox1)
# x1b, y1b, x2b, y2b = map(int, bbox2)
# img_pts1_full = img_pts1 + np.array([x1a, y1a])
# img_pts2_full = img_pts2 + np.array([x1b, y1b])

# # Define real-world 3D coordinates of the 4 keypoints (e.g., rear left, rear right, front right, front left)
# # Assume car lies on X (width) and Z (length) plane, Y is up (height ignored)
# obj_pts = np.array([
#     [-car_dimensions["width"]/2, 0, 0],                      # Rear Left
#     [1.55, 0.1, 3.02],                      # License plate center
#     [ car_dimensions["width"]/2, 0, car_dimensions["length"]],  # Front Right
#     [-car_dimensions["width"]/2, 0, car_dimensions["length"]]   # Front Left
# ], dtype=np.float64)

# # Solve for camera pose in both frames
# R1, t1 = compute_pose_3d(img_pts1_full, obj_pts, intrinsic_matrix, dist_coeffs)
# R2, t2 = compute_pose_3d(img_pts2_full, obj_pts, intrinsic_matrix, dist_coeffs)

# # Compute Euclidean displacement between translation vectors
# displacement = np.linalg.norm(t2 - t1)  # in meters
# time_elapsed = frame_gap / fps          # in seconds
# speed_m_s = displacement / time_elapsed
# speed_kmh = speed_m_s * 3.6
# speed_mph = speed_kmh / 1.60934

# # Output results
# print(f"Estimated vehicle displacement: {displacement:.2f} meters")
# print(f"Time elapsed: {time_elapsed:.2f} seconds")
# print(f"Estimated speed: {speed_m_s:.2f} m/s ({speed_mph:.2f} mph)")
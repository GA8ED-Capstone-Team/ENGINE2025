import cv2
import numpy as np
from ultralytics import SAM
from yolo_detector import YoloDetector
from dpsort_tracker import DeepSortTracker
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from scipy.spatial.distance import cdist
import cv2


def compute_pose_3d(image_points, object_points, camera_matrix, dist_coeffs):
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs,
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not success:
        raise RuntimeError("SolvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", tvec)
    return R, tvec

def order_corners_clockwise(corners):
    """
    Returns the corners ordered as:
    bottom-left, top-left, top-right, bottom-right
    """
    corners = np.array(corners, dtype=np.float32)
    center = np.mean(corners, axis=0)

    def angle(pt):
        return np.arctan2(pt[1] - center[1], pt[0] - center[0])

    sorted_corners = sorted(corners, key=angle)

    # Rearrange based on geometric assumptions
    # Assume vehicle is pointing up, sort accordingly
    top_two = sorted(sorted_corners[:2], key=lambda p: p[0])  # sort left-right
    bottom_two = sorted(sorted_corners[2:], key=lambda p: p[0])

    return np.array([bottom_two[0], top_two[0], top_two[1], bottom_two[1]], dtype=np.float32)

def get_vehicle_base_contour(mask, bbox, max_iters=20, debug=False):
    """
    Estimate vehicle base rectangle from mask using K-means-like bottom contour clustering.

    Args:
        mask (np.ndarray): Binary mask (H x W), dtype bool or uint8.
        bbox (tuple): (x1, y1, x2, y2) bounding box in global image coordinates.
        max_iters (int): Max iterations for clustering.
        debug (bool): If True, prints debug info and shows intermediate results.

    Returns:
        List of 4 points (x, y) in global image coordinates, or None if failed.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_img, w_img = mask.shape[:2]
    w = x2 - x1
    h = y2 - y1
    pad_w = int(0.025 * w)
    pad_h = int(0.025 * h)
    # Expand bbox by 5% on all sides, but keep within image bounds
    x1e = max(0, x1 - pad_w)
    y1e = max(0, y1 - pad_h)
    x2e = min(w_img, x2 + pad_w)
    y2e = min(h_img, y2 + pad_h)

    # Use expanded bbox for cropping
    mask_crop = mask[y1e:y2e, x1e:x2e].astype(np.uint8)
    if mask_crop.size == 0 or mask_crop.shape[0] < 2 or mask_crop.shape[1] < 2:
        if debug:
            print("Empty or invalid mask crop.")
        return None
    
    mask_uint8 = mask_crop.astype(np.uint8) * 255
    grad_y = cv2.Sobel(mask_uint8, cv2.CV_64F, 0, 1, ksize=3)
    bottom_edge = np.argwhere(grad_y < 0)
    if debug:
        print(f"Bottom edge points: {len(bottom_edge)}")
    if len(bottom_edge) < 4:
        return None

    # MeanShift clustering
    try:
        points = np.flip(bottom_edge, axis=1)  # (x, y)
        ms = MeanShift(bandwidth=20, bin_seeding=True)
        ms.fit(points)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
    except Exception as e:
        if debug:
            print(f"MeanShift error: {e}")
        return None

    threshold = 0.5 * h
    # --- Only filter clusters if there are more than 2 clusters ---
    if len(cluster_centers) > 2:
        dists = cdist(cluster_centers, cluster_centers)
        np.fill_diagonal(dists, np.inf)
        keep_mask = np.min(dists, axis=1) < threshold
        cluster_centers = cluster_centers[keep_mask]
        keep_labels = [i for i, keep in enumerate(keep_mask) if keep]
        mask_keep = np.isin(labels, keep_labels)
        points = points[mask_keep]
        labels = labels[mask_keep]
        unique_labels = {old: new for new, old in enumerate(sorted(set(labels)))}
        labels = np.array([unique_labels[l] for l in labels])
    else:
        keep_mask = np.ones(len(cluster_centers), dtype=bool)
    
    if debug:
        print(len(cluster_centers), "clusters found")

    # Filter outliers within each cluster
    filtered_points = []
    filtered_labels = []
    for cluster_id in np.unique(labels):
        cluster_points = points[labels == cluster_id]
        if len(cluster_points) < 3:
            filtered_points.append(cluster_points)
            filtered_labels.extend([cluster_id] * len(cluster_points))
            continue
        centroid = np.mean(cluster_points, axis=0)
        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        std = np.std(dists)
        keep = dists < (np.mean(dists) + 1.5 * std)
        filtered_points.append(cluster_points[keep])
        filtered_labels.extend([cluster_id] * np.sum(keep))
    if len(filtered_points) == 0:
        if debug:
            print("No filtered points after outlier removal.")
        return None
    filtered_points = np.vstack(filtered_points)
    filtered_labels = np.array(filtered_labels)

    # Enhanced Step 2: Initialize two lines using diagonals of the crop
    h, w = mask_crop.shape
    
    # Use V-shape initialization:
    mid_bottom = ((w-1)//2, h-1)
    left_top = (int(w*0.25), int(h*0.6))
    right_top = (int(w*0.75), int(h*0.6))

    line_left = np.polyfit([mid_bottom[0], left_top[0]], [mid_bottom[1], left_top[1]], 1)
    line_right = np.polyfit([mid_bottom[0], right_top[0]], [mid_bottom[1], right_top[1]], 1)
    lines = [line_left, line_right]

    for _ in range(max_iters):
        clusters = [[] for _ in range(2)]
        for pt in filtered_points:
            dists = [np.abs(l[0]*pt[0] - pt[1] + l[1]) / np.sqrt(l[0]**2 + 1) for l in lines]
            min_idx = np.argmin(dists)
            clusters[min_idx].append(pt)
        new_lines = []
        for pts in clusters:
            if len(pts) < 2:
                if debug:
                    print("Cluster too small:", [len(c) for c in clusters])
                return None
            pts = np.array(pts)
            try:
                line = np.polyfit(pts[:, 0], pts[:, 1], 1)
            except Exception as e:
                if debug:
                    print(f"Polyfit error: {e}")
                return None
            new_lines.append(line)
        lines = new_lines

    def line_intersection(m1, b1, m2, b2):
        if np.isinf(m1) and np.isinf(m2):
            return None
        elif np.isinf(m1):
            x = b1
            y = m2 * x + b2
            return (x, y)
        elif np.isinf(m2):
            x = b2
            y = m1 * x + b1
            return (x, y)
        elif m1 == m2:
            return None
        else:
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1
            return (x, y)

    def line_eq(p1, p2):
        x1_, y1_ = p1
        x2_, y2_ = p2
        if x2_ != x1_:
            m_ = (y2_ - y1_) / (x2_ - x1_)
            b_ = y1_ - m_ * x1_
        else:
            m_ = np.inf
            b_ = x1_
        return m_, b_

    # Ensure m1 is the width line and m2 is the length line
    if abs(lines[0][0]) <= abs(lines[1][0]):
        m1, b1 = lines[0]
        m2, b2 = lines[1]
    else:
        m1, b1 = lines[1]
        m2, b2 = lines[0]
    bl_pt = line_intersection(m1, b1, m2, b2)

    m5, b5 = line_eq((x2-x1, 0), (x2-x1, y2-y1))  # height line - right
    m6, b6 = line_eq((0, 0), (0, y2-y1))         # height line - left

    left_corner = line_intersection(m1, b1, m6, b6)
    right_corner = line_intersection(m2, b2, m5, b5)
    if left_corner is not None and right_corner is not None:
        m3, b3 = m1, right_corner[1] - m1 * right_corner[0]
        m4, b4 = m2, left_corner[1] - m2 * left_corner[0]
        inferred_pt = line_intersection(m3, b3, m4, b4)
    else:
        inferred_pt = None

    # Map all corners to int and check validity
    if all(pt is not None and np.all(np.isfinite(pt)) for pt in [bl_pt, left_corner, right_corner, inferred_pt]):
        bl_pt = tuple(map(int, bl_pt))
        left_corner = tuple(map(int, left_corner))
        right_corner = tuple(map(int, right_corner))
        inferred_pt = tuple(map(int, inferred_pt))
        corners_crop = [bl_pt, left_corner, inferred_pt, right_corner]
    else:
        if debug:
            print("Could not infer 4th corner or some corners are invalid.")
        return None

    # Convert to global coordinates
    corners_global = [(pt[0] + x1, pt[1] + y1) for pt in corners_crop]

    # Ensure clockwise order: top-right, bottom-right, bottom-left, top-left
    corners_global = order_corners_clockwise(corners_global)

    # --- REJECTION CRITERIA ---

    # Check area (not too small or too large)
    def polygon_area(pts):
        pts = np.array(pts)
        return 0.5 * np.abs(np.dot(pts[:,0], np.roll(pts[:,1], 1)) - np.dot(pts[:,1], np.roll(pts[:,0], 1)))
    area = polygon_area(corners_global)
    bbox_area = (x2-x1) * (y2-y1)
    if area < 0.05 * bbox_area or area > 1.5 * bbox_area:
        if debug:
            print(f"Unreasonable area: {area:.1f}, bbox_area={bbox_area:.1f}")
        return None
    
    # Check side lengths (not too small or too large)
    side_lengths = [np.linalg.norm(np.array(corners_global[i]) - np.array(corners_global[(i+1)%4]))
                    for i in range(4)]
    bbox_diag = np.linalg.norm([x2-x1, y2-y1])
    min_side = min(side_lengths)
    max_side = max(side_lengths)
    if min_side < 0.05 * bbox_diag or max_side > 2.0 * bbox_diag:
        if debug:
            print(f"Unreasonable side lengths: {side_lengths}, bbox_diag={bbox_diag:.1f}")
        return None
    
    
    return corners_global


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

    N = 10  # window size
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

        detections = detector.detect(frame)
        tracking_ids, boxes = tracker.track(detections, frame)

        for tracking_id, box in zip(tracking_ids, boxes):
            x1, y1, x2, y2 = map(int, box)
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
            
            vehicle_history[tracking_id].append((frame_idx, corners))
            try:
                R, t = compute_pose_3d(np.array(corners, dtype=np.float64), world_coords, intrinsic_matrix, dist_coeffs)
                vehicle_calibrations[tracking_id].append((frame_idx, R, t))
            except:
                continue

            # filtered = filter_calibrations(vehicle_calibrations[tracking_id][-N:])
            filtered = ransac_filter_translations(vehicle_calibrations[tracking_id][-N:])
            # filtered = vehicle_calibrations[tracking_id][-N:]
            for i in range(len(filtered)):
                for j in range(i + 1, len(filtered)):
                    f1, _, t1 = filtered[i]
                    f2, _, t2 = filtered[j]
                    if f2 - f1 >= B:
                        displacement = np.linalg.norm(t2 - t1)
                        time_elapsed = (f2 - f1) / fps
                        speed_mph = (displacement / time_elapsed) * 3.6 / 1.60934
                        if 0 < speed_mph < 200:
                            vehicle_speeds[tracking_id].append(speed_mph)
                            speed_time_series[tracking_id].append((frame_idx, speed_mph))
                        break

            if vehicle_speeds[tracking_id]:
                filtered_speed = moving_average_filter(vehicle_speeds[tracking_id], window=1)
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

    plt.figure(figsize=(12, 6))
    for tracking_id, series in speed_time_series.items():
        if len(series) < 2:
            continue
        series = sorted(series)
        frames, speeds = zip(*series)
        plt.plot(frames, speeds, label=f"ID {tracking_id}")

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
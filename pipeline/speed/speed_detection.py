import os
import json
import numpy as np
from collections import defaultdict
from sklearn.linear_model import RANSACRegressor
import cv2
import boto3
import psycopg2
import tempfile
from urllib.parse import urlparse
from scipy.spatial.distance import cdist
from sklearn.cluster import MeanShift
from ultralytics import SAM

# Constants
W, L = 1.82, 4.8  # Vehicle dimensions in meters TODO: put in moondream here
FPS = 30  # Assuming 30 FPS, adjust if different
MIN_TRACK_FRAMES = 10  # Minimum number of frames a track must persist
DB_SECRET_NAME = "ga8ed-db-userpass"
DB_NAME = "postgres"
DB_SCHEMA = "ga8ed"
DB_TABLE = "video_metadata_2"

# Initialize SAM model
sam_model = SAM("mobile_sam.pt")


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

    return np.array(
        [bottom_two[0], top_two[0], top_two[1], bottom_two[1]], dtype=np.float32
    )


def get_vehicle_base_contour(mask, bbox, max_iters=20, debug=False):
    """
    Estimate vehicle base rectangle from mask using K-means-like bottom contour clustering,
    with cluster filtering logic integrated from notebook.

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
    pad_w = int(0.0 * w)
    pad_h = int(0.0 * h)
    x1e = max(0, x1 - pad_w)
    y1e = max(0, y1 - pad_h)
    x2e = min(w_img, x2 + pad_w)
    y2e = min(h_img, y2 + pad_h)

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

    # --- Cluster filtering logic from notebook ---
    total_points = len(points)
    # Filter 1: Remove clusters with less than 5% of all points
    keep_cluster_ids = []
    for cluster_id in np.unique(labels):
        cluster_size = np.sum(labels == cluster_id)
        if cluster_size >= 0.03 * total_points:
            keep_cluster_ids.append(cluster_id)
    keep_mask = np.isin(labels, keep_cluster_ids)
    points = points[keep_mask]
    labels = labels[keep_mask]
    if len(keep_cluster_ids) > 0:
        cluster_centers = cluster_centers[keep_cluster_ids]
    else:
        if debug:
            print("No clusters left after 5% filtering.")
        return None

    # Only apply further filters if any cluster is less than 10% of all points
    apply_next_filters = False
    for cluster_id in np.unique(labels):
        cluster_size = np.sum(labels == cluster_id)
        if cluster_size < 0.06 * total_points:
            apply_next_filters = True
            break

    threshold = 0.65 * h  # 50% of the height of the cropped car image
    if apply_next_filters:
        # Filter clusters that are too far apart
        if len(cluster_centers) > 1:
            dists = cdist(cluster_centers, cluster_centers)
            np.fill_diagonal(dists, np.inf)
            keep_mask = np.min(dists, axis=1) < threshold
            cluster_centers = cluster_centers[keep_mask]
            keep_labels = [i for i, keep in enumerate(keep_mask) if keep]
            mask2 = np.isin(labels, keep_labels)
            points = points[mask2]
            labels = labels[mask2]
            # Remap labels to be consecutive for plotting
            unique_labels = {old: new for new, old in enumerate(sorted(set(labels)))}
            labels = np.array([unique_labels[l] for l in labels])
        else:
            keep_mask = np.ones(len(cluster_centers), dtype=bool)

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
            keep = dists < (np.mean(dists) + 1 * std)
            filtered_points.append(cluster_points[keep])
            filtered_labels.extend([cluster_id] * np.sum(keep))
        if len(filtered_points) == 0:
            if debug:
                print("No filtered points after outlier removal.")
            return None
        filtered_points = np.vstack(filtered_points)
        filtered_labels = np.array(filtered_labels)
    else:
        filtered_points = points
        filtered_labels = labels

    # --- Rectangle fitting as before ---
    h, w = mask_crop.shape
    mid_bottom = ((w - 1) // 2, h - 1)
    left_top = (int(w * 0.25), int(h * 0.6))
    right_top = (int(w - 1), int(h * 0.6))

    line_left = np.polyfit(
        [mid_bottom[0], left_top[0]], [mid_bottom[1], left_top[1]], 1
    )
    line_right = np.polyfit([w, right_top[0]], [mid_bottom[1], right_top[1]], 1)
    lines = [line_left, line_right]

    # # Line 1: along the bottom edge of the crop (horizontal)
    # line_bottom = np.polyfit([0, w-1], [h-1, h-1], 1)  # y = h-1

    # # Line 2: along the right edge of the crop (vertical)
    # # For a vertical line, polyfit will fail, so handle separately
    # m_right = np.inf
    # b_right = w-1

    # lines = [line_bottom, (m_right, b_right)]

    for _ in range(max_iters):
        clusters = [[] for _ in range(2)]
        for pt in filtered_points:
            dists = [
                np.abs(l[0] * pt[0] - pt[1] + l[1]) / np.sqrt(l[0] ** 2 + 1)
                for l in lines
            ]
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

    if abs(lines[0][0]) <= abs(lines[1][0]):
        m1, b1 = lines[0]
        m2, b2 = lines[1]
    else:
        m1, b1 = lines[1]
        m2, b2 = lines[0]
    bl_pt = line_intersection(m1, b1, m2, b2)

    m5, b5 = line_eq((x2 - x1, 0), (x2 - x1, y2 - y1))  # height line - right
    m6, b6 = line_eq((0, 0), (0, y2 - y1))  # height line - left

    left_corner = line_intersection(m1, b1, m6, b6)
    right_corner = line_intersection(m2, b2, m5, b5)
    if left_corner is not None and right_corner is not None:
        m3, b3 = m1, right_corner[1] - m1 * right_corner[0]
        m4, b4 = m2, left_corner[1] - m2 * left_corner[0]
        inferred_pt = line_intersection(m3, b3, m4, b4)
    else:
        inferred_pt = None

    # Map all corners to int and check validity
    if all(
        pt is not None and np.all(np.isfinite(pt))
        for pt in [bl_pt, left_corner, right_corner, inferred_pt]
    ):
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
    def polygon_area(pts):
        pts = np.array(pts)
        return 0.5 * np.abs(
            np.dot(pts[:, 0], np.roll(pts[:, 1], 1))
            - np.dot(pts[:, 1], np.roll(pts[:, 0], 1))
        )

    area = polygon_area(corners_global)
    bbox_area = (x2 - x1) * (y2 - y1)
    if area < 0.05 * bbox_area or area > 1.5 * bbox_area:
        if debug:
            print(f"Unreasonable area: {area:.1f}, bbox_area={bbox_area:.1f}")
        return None

    side_lengths = [
        np.linalg.norm(
            np.array(corners_global[i]) - np.array(corners_global[(i + 1) % 4])
        )
        for i in range(4)
    ]
    bbox_diag = np.linalg.norm([x2 - x1, y2 - y1])
    min_side = min(side_lengths)
    max_side = max(side_lengths)
    if min_side < 0.05 * bbox_diag or max_side > 2.0 * bbox_diag:
        if debug:
            print(
                f"Unreasonable side lengths: {side_lengths}, bbox_diag={bbox_diag:.1f}"
            )
        return None

    return corners_global


def get_db_userpass():
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=DB_SECRET_NAME)
    secret = json.loads(response["SecretString"])
    return secret


def update_video_record(video_id, max_speed, speed_alert):
    secrets = get_db_userpass()
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=secrets["username"],
        password=secrets["password"],
        host=secrets["host"],
        port=secrets["port"],
    )
    cur = conn.cursor()

    cur.execute(
        f"""
        UPDATE {DB_SCHEMA}.{DB_TABLE}
        SET max_speed = %s,
            speed_alert = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE video_id = %s
        """,
        (max_speed, speed_alert, video_id),
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Updated record for Video ID: {video_id}")


def upload_json_to_s3(json_data, bucket, prefix, filename="speeds.json"):
    output_path = os.path.join("/tmp", filename)
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)
    s3 = boto3.client("s3")
    s3.upload_file(output_path, bucket, f"{prefix}/{filename}")
    print(f"✅ Uploaded speeds to s3://{bucket}/{prefix}/{filename}")


def get_video_id_from_s3_path(s3_path):
    parsed = urlparse(s3_path)
    path_parts = parsed.path.lstrip("/").split("/")
    return path_parts[-2]


def moving_average_filter(speeds, window=5):
    if not speeds:
        return None
    speeds = np.array(speeds[-window:])
    return np.mean(speeds)


def ransac_filter_translations(calibs, min_samples=2, residual_threshold=1.0):
    if len(calibs) < min_samples:
        return []

    frames = np.array([f for f, _, _ in calibs])
    translations = np.array([t.ravel() for _, _, t in calibs])

    inlier_mask = np.ones(len(calibs), dtype=bool)
    for axis in [2]:
        ransac = RANSACRegressor(
            min_samples=min_samples, residual_threshold=residual_threshold
        )
        try:
            ransac.fit(frames.reshape(-1, 1), translations[:, axis])
            axis_inliers = ransac.inlier_mask_
            inlier_mask &= axis_inliers
        except Exception as e:
            print(f"RANSAC fitting error on axis {axis}: {e}")
            return []

    return [calibs[i] for i in range(len(calibs)) if inlier_mask[i]]


def compute_pose_3d(corners, world_coords, intrinsic_matrix, dist_coeffs):
    """Compute 3D pose using PnP"""
    success, rvec, tvec = cv2.solvePnP(
        world_coords, corners, intrinsic_matrix, dist_coeffs
    )
    if not success:
        raise ValueError("PnP failed")
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec


def download_video_from_s3(s3_uri):
    """Download video from S3 to a temporary file"""
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    return local_path


def download_predictions_from_s3(s3_uri):
    parsed = urlparse(s3_uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    return local_path, bucket, key


def process_tracks(
    tracked_predictions_path: str,
    video_path: str,
    output_path: str,
    camera_intrinsics_path: str,
    min_track_length: int = 10,
    min_speed_threshold: float = 5.0,
    max_speed_threshold: float = 100.0,
    speed_alert_threshold: float = 80.0,
    min_frame_persistence: int = 10,
    window_size: int = 100,
    raise_alerts: bool = True,
):
    """
    Process tracked predictions to compute speeds and generate alerts.
    """
    # Download predictions from S3 if it's an S3 URI
    if tracked_predictions_path.startswith("s3://"):
        local_predictions_path, bucket, key = download_predictions_from_s3(tracked_predictions_path)
        tracked_predictions_path = local_predictions_path

    # Load tracked predictions
    with open(tracked_predictions_path, "r") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    window = frames[-window_size:] if len(frames) > window_size else frames
    total_tracked = len(window)

    # Load camera intrinsics
    camera_intrinsics = np.load(camera_intrinsics_path)
    camera_matrix = camera_intrinsics["camera_matrix"]
    dist_coeffs = camera_intrinsics["dist_coeffs"]

    # Process tracks and compute speeds
    speeds = {}
    speed_alerts = {}
    for frame in window:
        for tr in frame.get("tracks", []):
            track_id = tr["track_id"]
            if track_id not in speeds:
                speeds[track_id] = {
                    "speeds": [],
                    "max_speed": 0.0,
                    "class_id": tr["class_id"],
                    "class_name": tr["class_name"],
                    "persistence": 0,
                }
            speeds[track_id]["persistence"] += 1

    # Calculate speeds for each track
    for track_id, track_data in speeds.items():
        if track_data["persistence"] >= min_track_length:
            # Calculate speed using the track's bounding boxes
            # This is a placeholder - you'll need to implement the actual speed calculation
            # using the track's bounding boxes and camera intrinsics
            speed = calculate_speed(track_id, frames, camera_matrix, dist_coeffs)
            if min_speed_threshold <= speed <= max_speed_threshold:
                track_data["speeds"].append(speed)
                track_data["max_speed"] = max(track_data["max_speed"], speed)

    # Generate alerts for tracks exceeding speed threshold
    for track_id, track_data in speeds.items():
        if track_data["max_speed"] > speed_alert_threshold:
            speed_alerts[track_id] = {
                "max_speed": track_data["max_speed"],
                "class_name": track_data["class_name"],
            }
            if raise_alerts:
                print(
                    f"ALERT: {track_data['class_name']} (ID: {track_id}) exceeded speed threshold "
                    f"(speed={track_data['max_speed']:.2f} km/h)"
                )

    # Write results to output file
    with open(output_path, "w") as f:
        json.dump(
            {
                "video": data.get("video"),
                "total_tracked_frames": total_tracked,
                "speeds": speeds,
                "alerts": speed_alerts,
            },
            f,
            indent=2,
        )
    print(f"Speed analysis written to {output_path}")

    return speeds, speed_alerts


def get_video_path_from_db(video_id):
    secrets = get_db_userpass()
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=secrets["username"],
        password=secrets["password"],
        host=secrets["host"],
        port=secrets["port"],
    )
    cur = conn.cursor()

    cur.execute(
        f"""
        SELECT video_uri
        FROM {DB_SCHEMA}.{DB_TABLE}
        WHERE video_id = %s
        """,
        (video_id,),
    )
    result = cur.fetchone()
    cur.close()
    conn.close()

    if result is None:
        raise ValueError(f"No video path found for video_id: {video_id}")
    return result[0]


def main():
    # Get input paths from environment variables
    tracked_predictions_path = os.environ.get("TRACKED_PREDICTIONS")
    if not tracked_predictions_path:
        raise ValueError("TRACKED_PREDICTIONS environment variable not set")

    # Get video ID and video URI from database
    video_id = get_video_id_from_s3_path(tracked_predictions_path)
    video_s3_uri = get_video_path_from_db(video_id)

    # Set up paths
    camera_intrinsics_path = "/app/camera_intrinsics.npz"  # Built-in path
    speed_alert_threshold = float(
        os.environ.get("SPEED_ALERT_THRESHOLD", "30.0")
    )  # Default 30 mph

    # Process tracks and get speeds
    speeds, speed_alerts = process_tracks(
        tracked_predictions_path, video_s3_uri, camera_intrinsics_path
    )
    print(f"Speeds: {speeds}")

    # Calculate max speed and check alert threshold
    max_speed = max(speeds.values(), key=lambda x: x["max_speed"])["max_speed"]
    speed_alert = max_speed > speed_alert_threshold

    # Update database
    update_video_record(video_id, max_speed, speed_alert)

    # Upload speeds to S3
    parsed = urlparse(tracked_predictions_path)
    bucket = parsed.netloc
    output_prefix = f"speed_results/{video_id}"
    upload_json_to_s3(speeds, bucket, output_prefix, filename="speeds.json")


if __name__ == "__main__":
    main()

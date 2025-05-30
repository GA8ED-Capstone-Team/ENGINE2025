import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from scipy.spatial.distance import cdist

def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / e_x.sum()
def crop_car_image(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return frame[y1:y2, x1:x2]

# def find_rear_point_from_mask_and_depth(
#     frame, sam_mask, bbox, depth_model,
#     depth_percentile=10, depth_z=2.0, cluster_z=1.5,
#     depth_weight=2.0, dist_weight=1.0, n_clusters=3
# ):
#     """
#     Finds the rear (deepest & central) point inside a SAM mask using DAM2 depth prediction.

#     Args:
#         frame (np.ndarray): Full RGB frame.
#         sam_mask (np.ndarray): Binary mask for the object (bool, full-frame).
#         bbox (tuple): (x1, y1, x2, y2) bounding box.
#         depth_model (torch.nn.Module): DAM2 depth model.
#         depth_percentile (float): Percentile for deepest region (default 10).
#         depth_z (float): Z-score threshold for depth outlier removal.
#         cluster_z (float): Z-score threshold for cluster inlier filtering.
#         depth_weight (float): Weight for depth in scoring.
#         dist_weight (float): Weight for distance in scoring.
#         n_clusters (int): Number of clusters for KMeans.

#     Returns:
#         rear_point (tuple): (x, y) coordinates of the rear point in original frame.
#         depth_pred (np.ndarray): Predicted depth map (cropped).
#         masked_depth (np.ndarray): Masked depth map used for rear point extraction.
#     """
#     x1, y1, x2, y2 = bbox

#     # 1. Crop region
#     bbox_crop = frame[y1:y2, x1:x2]
#     if bbox_crop.size == 0:
#         print("[WARNING] Empty bounding box crop.")
#         return None, None, None

#     # 2. Run DAM2 depth prediction
#     with torch.no_grad():
#         depth_pred = depth_model.infer_image(bbox_crop)

#     # 3. Resize and erode SAM mask to match depth resolution
#     sam_mask_crop = sam_mask[y1:y2, x1:x2]
#     if sam_mask_crop.size == 0:
#         print("[WARNING] Empty mask crop.")
#         return None, depth_pred, None

#     kernel = np.ones((3, 3), np.uint8)
#     sam_mask_crop = cv2.erode(sam_mask_crop.astype(np.uint8), kernel, iterations=1).astype(bool)
#     sam_mask_resized = cv2.resize(
#         sam_mask_crop.astype(np.uint8),
#         (depth_pred.shape[1], depth_pred.shape[0]),
#         interpolation=cv2.INTER_NEAREST
#     ).astype(bool)

#     # 4. Apply mask to depth
#     masked_depth = np.where(sam_mask_resized, depth_pred, np.nan)

#     # 5. Get valid (non-NaN) points
#     valid_y, valid_x = np.where(~np.isnan(masked_depth))
#     if len(valid_y) == 0:
#         print("[WARNING] No valid depth points inside SAM mask.")
#         return None, depth_pred, masked_depth

#     depths = masked_depth[valid_y, valid_x]
#     depth_threshold = np.percentile(depths, depth_percentile)
#     deep_mask = depths <= depth_threshold

#     deep_depths = depths[deep_mask]
#     if deep_depths.size == 0:
#         print("[WARNING] No deep points found.")
#         return None, depth_pred, masked_depth

#     # 6. Z-score filter
#     mean_depth = np.mean(deep_depths)
#     std_depth = np.std(deep_depths)
#     z_scores = (deep_depths - mean_depth) / (std_depth + 1e-8)
#     inlier_mask = np.abs(z_scores) < depth_z

#     xs_deep = valid_x[deep_mask][inlier_mask]
#     ys_deep = valid_y[deep_mask][inlier_mask]
#     if xs_deep.size == 0:
#         print("[WARNING] No inlier deep points after z-score filtering.")
#         return None, depth_pred, masked_depth

#     xy = np.stack([xs_deep, ys_deep], axis=1)
#     depth = masked_depth[ys_deep, xs_deep][:, None]
#     features = np.concatenate([xy, depth], axis=1)

#     # 7. KMeans clustering
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0)
#     labels = kmeans.fit_predict(features)

#     # 8. Cluster statistics
#     scores = []
#     for i in range(n_clusters):
#         cluster_points = features[labels == i]
#         if cluster_points.shape[0] < 2:
#             scores.append(-np.inf)
#             continue

#         mean_depth = np.mean(cluster_points[:, 2])
#         cov = np.cov(cluster_points.T) + np.eye(3) * 1e-6
#         spread = np.linalg.det(cov)

#         n_points = len(cluster_points)
#         score = (
#             depth_weight * mean_depth
#             + 0.5 * n_points
#             - 2.0 * spread
#         )
#         scores.append(score)

#     probs = softmax(np.array(scores))
#     best_cluster = np.argmax(probs)

#     # 9. Refine points from best cluster
#     cluster_mask = (labels == best_cluster)
#     xs_best = xs_deep[cluster_mask]
#     ys_best = ys_deep[cluster_mask]
#     depths_best = masked_depth[ys_best, xs_best]

#     if depths_best.size == 0:
#         print("[WARNING] Best cluster has no depth values.")
#         return None, depth_pred, masked_depth

#     mean_depth = np.mean(depths_best)
#     std_depth = np.std(depths_best)
#     z_scores = (depths_best - mean_depth) / (std_depth + 1e-8)
#     inlier_mask = np.abs(z_scores) < cluster_z

#     xs_inlier = xs_best[inlier_mask]
#     ys_inlier = ys_best[inlier_mask]
#     depths_inlier = depths_best[inlier_mask]

#     if xs_inlier.size == 0:
#         print("[WARNING] No inlier points in best cluster.")
#         return None, depth_pred, masked_depth

#     coords = np.stack([xs_inlier, ys_inlier], axis=1)
#     # mean_xy = np.mean(coords, axis=0)
#     corner_bbox = np.array([x2, y1]) # top-right corner of the bbox
#     dists = np.linalg.norm(coords - corner_bbox, axis=1)

#     # 10. Score each point by depth & centrality
#     depths_norm = (depths_inlier - depths_inlier.min()) / (depths_inlier.ptp() + 1e-8)

#     dists_norm = (dists - dists.min()) / (dists.ptp() + 1e-8)
#     point_scores = 1 /  (depths_norm+ 0.1) - 1/ (0.5*dists_norm+1e-8)

#     if point_scores.size == 0:
#         print("[WARNING] No point scores computed.")
#         return None, depth_pred, masked_depth

#     idx_best = np.argmax(point_scores)
#     x_best, y_best = xs_inlier[idx_best], ys_inlier[idx_best]

#     print(f"[INFO] Best rear point (local): (x={x_best}, y={y_best}), depth={depths_inlier[idx_best]:.3f}")

#     # 11. Rescale point to original frame coordinates
#     scale_x = (x2 - x1) / depth_pred.shape[1]
#     scale_y = (y2 - y1) / depth_pred.shape[0]
#     x_best = int(x1 + x_best * scale_x)
#     y_best = int(y1 + y_best * scale_y)

#     return (x_best, y_best), depth_pred, masked_depth

def find_rear_point_from_mask_and_depth(
    frame, sam_mask, bbox, depth_model,
    depth_percentile=10, depth_z=1.0, cluster_z=1.0,
    depth_weight=0.6, dist_weight=0.4
):
    x1, y1, x2, y2 = bbox
    print(f"\n[INFO] Processing BBox: {bbox} ({x2 - x1}x{y2 - y1})")

    # 1. Crop region
    bbox_crop = frame[y1:y2, x1:x2]
    if bbox_crop.size == 0:
        print("[WARNING] Empty bounding box crop.")
        return None, None, None, []

    # 2. Run DAM2 depth prediction
    with torch.no_grad():
        depth_pred = depth_model.infer_image(bbox_crop)


    print(f"[DEBUG] depth_pred: min={depth_pred.min():.3f}, max={depth_pred.max():.3f}, mean={depth_pred.mean():.3f}")

    # 3. Resize and erode SAM mask
    sam_mask_crop = sam_mask[y1:y2, x1:x2]
    if sam_mask_crop.size == 0:
        print("[WARNING] Empty mask crop.")
        return None, depth_pred, None, []

    kernel = np.ones((3, 3), np.uint8)
    sam_mask_crop = cv2.erode(sam_mask_crop.astype(np.uint8), kernel, iterations=1).astype(bool)
    sam_mask_resized = cv2.resize(
        sam_mask_crop.astype(np.uint8),
        (depth_pred.shape[1], depth_pred.shape[0]),
        interpolation=cv2.INTER_AREA
    ).astype(bool)

    # 4. Apply mask to depth
    masked_depth = np.where(sam_mask_resized, depth_pred, np.nan)
    valid_y, valid_x = np.where(~np.isnan(masked_depth))
    if len(valid_y) == 0:
        print("[WARNING] No valid depth points inside SAM mask.")
        return None, depth_pred, masked_depth, []

    depths = masked_depth[valid_y, valid_x]
    print(f"[DEBUG] All depths: min={np.nanmin(depths):.3f}, max={np.nanmax(depths):.3f}, mean={np.nanmean(depths):.3f}")

    # 5. Select deepest X% (higher = deeper after inversion)
    depth_threshold = np.percentile(depths, depth_percentile)
    deep_mask = depths <= depth_threshold
    deep_depths = depths[deep_mask]
    print(f"[DEBUG] Deep threshold = {depth_threshold:.3f}, num_deep_points = {deep_depths.size}")

    if deep_depths.size == 0:
        print("[WARNING] No deep points found.")
        return None, depth_pred, masked_depth, []

    # 6. Z-score filtering
    mean_depth = np.mean(deep_depths)
    std_depth = np.std(deep_depths)
    z_scores = (deep_depths - mean_depth) / (std_depth + 1e-6)
    inlier_mask = np.abs(z_scores) < depth_z
    xs_deep = valid_x[deep_mask][inlier_mask]
    ys_deep = valid_y[deep_mask][inlier_mask]
    if xs_deep.size == 0:
        print("[WARNING] No inlier deep points after z-score filtering.")
        return None, depth_pred, masked_depth, []

    xy = np.stack([xs_deep, ys_deep], axis=1)
    depth_vals = masked_depth[ys_deep, xs_deep][:, None]
    features = np.concatenate([xy, depth_vals], axis=1)

    # 7. Mean Shift clustering
    bandwidth = estimate_bandwidth(features, quantile=0.2, n_samples=500)
    if bandwidth <= 0:
        bandwidth = 1  # fallback in case estimate fails

    # Run MeanShift clustering
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = meanshift.fit_predict(features)
    centers = meanshift.cluster_centers_
    n_clusters = len(centers)

    # 8. Score clusters (favor high depth + compactness)
    scores = []
    n_points_list = []
    mean_depth_list = []
    spread_list = []
    for i in range(n_clusters):
        cluster_points = features[labels == i]
        if cluster_points.shape[0] < 2:
            scores.append(-np.inf)
            continue
        n_points = len(cluster_points)
        mean_d = np.mean(cluster_points[:, 2])
        spread = np.linalg.norm(cluster_points[:, :2] - np.mean(cluster_points[:, :2], axis=0), axis=1).mean()
        n_points_list.append(n_points)
        mean_depth_list.append(mean_d)
        spread_list.append(spread)
        # print(f"[DEBUG] Cluster {i}: mean_depth={mean_d:.3f}, spread={spread:.2f}, score={score:.4f}")
    
    # Normalize features
    n_points_arr = np.array(n_points_list, dtype=np.float32)
    mean_depth_arr = np.array(mean_depth_list, dtype=np.float32)
    spread_arr = np.array(spread_list, dtype=np.float32)

    # Avoid division by zero
    spread_arr[spread_arr == 0] = 1e-6

    # Normalize to [0, 1]
    n_points_norm = (n_points_arr - n_points_arr.min()) / (n_points_arr.ptp() + 1e-8)
    mean_depth_norm = (mean_depth_arr - mean_depth_arr.min()) / (mean_depth_arr.ptp() + 1e-8)
    spread_norm = (spread_arr - spread_arr.min()) / (spread_arr.ptp() + 1e-8)

    # Compute score: more points, deeper, less scatter
    for i, (n_points, mean_d, spread) in enumerate(zip(n_points_norm, mean_depth_norm, spread_norm)):
        # score = ( mean_depth_norm[i]) / (spread_norm[i] + 1e-8)
        score  = 6 * mean_depth_norm[i] + 9 * n_points_norm[i] - 1 * (spread_norm[i])
        print(f"Cluster {i}: n_points={n_points_norm[i]}, mean_depth={mean_depth_norm[i]:.2f}, spread={spread_norm[i]:.2f}, score={score:.2f}")
        scores.append(score)
    
    scores = np.array(scores)
    probs = softmax(scores)
    
    best_cluster = np.argmax(probs)
    cluster_mask = (labels == best_cluster)

    xs_best = xs_deep[cluster_mask]
    ys_best = ys_deep[cluster_mask]
    depths_best = masked_depth[ys_best, xs_best]
    if depths_best.size == 0:
        print("[WARNING] Best cluster has no depth values.")
        return None, depth_pred, masked_depth, []

    # Refine within best cluster
    mean_depth = np.mean(depths_best)
    std_depth = np.std(depths_best)
    z_scores = (depths_best - mean_depth) / (std_depth + 1e-6)
    inlier_mask = np.abs(z_scores) < cluster_z
    xs_inlier = xs_best[inlier_mask]
    ys_inlier = ys_best[inlier_mask]
    depths_inlier = depths_best[inlier_mask]

    if xs_inlier.size == 0:
        print("[WARNING] No inlier points in best cluster.")
        return None, depth_pred, masked_depth, []

    coords = np.stack([xs_inlier, ys_inlier], axis=1)
    corner_bbox = np.array([x2-x1, 0])  # bottom-left, assume rear-left target
    dists = np.linalg.norm(coords - corner_bbox, axis=1)

    depths_norm = (depths_inlier - np.nanmin(depths)) / (np.nanmax(depths) - np.nanmin(depths) + 1e-6)
    dists_norm = (dists - dists.min()) / (dists.ptp() + 1e-6)
    point_scores = (1 - depths_norm) - 1.0*dists_norm
    idx_best = np.argmax(point_scores)

    x_best, y_best = xs_inlier[idx_best], ys_inlier[idx_best]
    best_depth = depths_inlier[idx_best]

    print(f"[INFO] Final rear point (local): x={x_best}, y={y_best}, depth={best_depth:.3f}")

    # Convert to global coords
    scale_x = (x2 - x1) / depth_pred.shape[1]
    scale_y = (y2 - y1) / depth_pred.shape[0]
    x_best = int(x1 + x_best * scale_x)
    y_best = int(y1 + y_best * scale_y)

    # For visualization
    cluster_coords_global = [
        (int(x1 + x * scale_x), int(y1 + y * scale_y)) for x, y in coords
    ]

    return (x_best, y_best), depth_pred, masked_depth, cluster_coords_global





##############################Depends on the output of the above function###############################


def find_front_point_1(frame, sam_mask, bbox, depth_model, rear_point):
    """
    Estimate the front (headlight) point of a car using SAM mask, bounding box,
    and depth information from Depth Anything V2.

    Args:
        frame (np.ndarray): Full RGB frame.
        sam_mask (np.ndarray): Full-frame binary mask of the object (bool).
        bbox (tuple): (x1, y1, x2, y2) of the object.
        depth_model (torch.nn.Module): Depth Anything V2 model (torch, on cuda).

    Returns:
        front_point (tuple): (x, y) point on headlight in full-frame coordinates.
    """
    x1, y1, x2, y2 = map(int, bbox)
    cropped_car = crop_car_image(frame, bbox)
    car_depth = depth_model.infer_image(cropped_car)

    cropped_mask = crop_car_image(sam_mask, bbox).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    cropped_mask = cv2.erode(cropped_mask, kernel, iterations=1).astype(bool)

    ys, xs = np.nonzero(cropped_mask)
    if len(xs) < 10:
        return None  # Not enough data

    depth_vals = car_depth[ys, xs]
    depth_thresh = np.percentile(depth_vals, 80)
    deep_mask = depth_vals >= depth_thresh

    ys_deep = ys[deep_mask]
    xs_deep = xs[deep_mask]
    depths_deep = depth_vals[deep_mask]

    mean_depth = np.mean(depths_deep)
    std_depth = np.std(depths_deep)
    z_scores = (depths_deep - mean_depth) / (std_depth + 1e-8)
    inliers = np.abs(z_scores) < 2
    ys_deep = ys_deep[inliers]
    xs_deep = xs_deep[inliers]

    if len(xs_deep) < 10:
        return None

    gray = cv2.cvtColor(cropped_car, cv2.COLOR_BGR2GRAY)
    grad = np.sqrt(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)**2 +
                   cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)**2)
    grad_vals = grad[ys_deep, xs_deep]
    grad_thresh = np.percentile(grad_vals, 60)
    strong_grad = grad_vals >= grad_thresh
    xs_deep = xs_deep[strong_grad]
    ys_deep = ys_deep[strong_grad]

    if len(xs_deep) < 3:
        return None

    # Prepare features for KMeans
    depths = car_depth[ys_deep, xs_deep]
    xy = np.stack([xs_deep, ys_deep], axis=1)
    features = np.concatenate([xy, depths[:, None]], axis=1)

    bandwidth = estimate_bandwidth(features, quantile=0.2, n_samples=500)
    if bandwidth <= 0:
        bandwidth = 1  # fallback in case estimate fails

    # Run MeanShift clustering
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = meanshift.fit_predict(features)
    centers = meanshift.cluster_centers_
    n_clusters = len(centers)

    rear_crop = (rear_point[0] - x1, rear_point[1] - y1)
    dists = np.linalg.norm(centers[:, :2] - np.array(rear_crop), axis=1)
    closest_idx = np.argmin(dists)

    cluster_points = xy[labels == closest_idx]
    cluster_depths = car_depth[cluster_points[:, 1], cluster_points[:, 0]]
    norm_depths = (cluster_depths - np.min(cluster_depths)) / (np.ptp(cluster_depths) + 1e-8)

    shallow_mask = norm_depths >= np.percentile(norm_depths, 90)
    shallow_points = cluster_points[shallow_mask]

    if len(shallow_points) == 0:
        return None

    mean_shallow_point = np.mean(shallow_points, axis=0)
    front_point = (int(mean_shallow_point[0] + x1), int(mean_shallow_point[1] + y1))
    return front_point

################################# Finds the closer to the camera side mirror center ####################################
def find_side_mirror_center(frame, sam_mask, bbox, depth_model, rear_point):
    """
    Detects a single side mirror in the cropped vehicle using depth edges.
    Returns the global (x, y) center coordinates of the detected mirror.
    """
    x1, y1, x2, y2 = bbox
    cropped_car = frame[y1:y2, x1:x2]
    cropped_mask = sam_mask[y1:y2, x1:x2].astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(cropped_mask, kernel, iterations=1)

    mask_border = cropped_mask - eroded_mask
    mask_border_bin = (mask_border > 0).astype(np.uint8)


    # Step 1: Get Depth Map
    with torch.no_grad():
        depth_crop = depth_model.infer_image(cropped_car)

    # Step 2: Normalize Depth
    depth_norm = (depth_crop - np.nanmin(depth_crop)) / (np.nanmax(depth_crop) - np.nanmin(depth_crop) + 1e-8)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    # Step 3: Canny + Morphological Closing
    edges = cv2.Canny(depth_uint8, 0, 150)
    depth_only = cv2.bitwise_and((edges > 0).astype(np.uint8), cv2.bitwise_not(mask_border_bin))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_edges = cv2.morphologyEx(depth_only, cv2.MORPH_CLOSE, kernel)

    # Step 4: Find Contours
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Filter Contours for Mirrors
    mirror_candidates = []
    h, w = depth_uint8.shape
    min_area = 10
    max_area = 800
    print(f"Found {len(contours)} contours.")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(f"Contour area: {area}")
        if min_area < area < max_area:
            print("yes")
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bw / float(bh + 1e-5)
            if h * 0.3 < y < h * 0.7 and 0.5 < aspect_ratio < 2.5:
                mirror_candidates.append(cnt)

    # Step 6: Pick Best Candidate (Largest Area)
    if not mirror_candidates:
        return None  # No mirror found

    best_cnt = max(mirror_candidates, key=cv2.contourArea)
    M = cv2.moments(best_cnt)
    if M["m00"] == 0:
        return None

    cx_local = int(M["m10"] / M["m00"])
    cy_local = int(M["m01"] / M["m00"])

    # Convert to global coordinates
    mirror_cx = x1 + cx_local
    mirror_cy = y1 + cy_local

    return (mirror_cx, mirror_cy)


def get_bottom_rectangle_corners(frame, mask, bbox, plot=False):
    """
    Approximates the bottom half of the mask as a polygon,
    finds the bottom left corner and the two lines intersecting at that corner,
    closes the rectangle, and returns the 4 corners in global image coordinates.

    Args:
        frame (np.ndarray): The original image.
        mask (np.ndarray): The binary mask (same size as frame).
        bbox (tuple): (x1, y1, x2, y2) bounding box.
        plot (bool): If True, plot the result on the cropped image.

    Returns:
        corners_global (list of tuple): List of 4 (x, y) points in global image coordinates.
    """
    x1, y1, x2, y2 = map(int, bbox)
    mask_crop = mask[y1:y2, x1:x2].astype(np.uint8)

    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found in mask.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    h_crop = mask_crop.shape[0]
    bottom_half_points = largest_contour[largest_contour[:, 0, 1] >  h_crop // 2]

    if len(bottom_half_points) >= 3:
        epsilon = 0.05 * cv2.arcLength(bottom_half_points, True)
        approx_poly = cv2.approxPolyDP(bottom_half_points, epsilon, True)
    else:
        approx_poly = bottom_half_points

    pts = approx_poly[:, 0, :] if approx_poly.ndim == 3 else approx_poly[:, 0, :]
    if len(pts) < 3:
        print("Not enough points to form a polygon.")
        return None

    idx_bl = np.lexsort((pts[:, 0], -pts[:, 1]))[0]
    bl_pt = tuple(pts[idx_bl])
    n = len(pts)
    prev_idx = (idx_bl - 1) % n
    next_idx = (idx_bl + 1) % n
    prev_pt = tuple(pts[prev_idx])
    next_pt = tuple(pts[next_idx])

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

    def line_intersection(m1, b1, m2, b2):
        if np.isinf(m1) and np.isinf(m2):
            return None  # Both lines vertical, no intersection
        elif np.isinf(m1):
            x = b1
            y = m2 * x + b2
            return (x, y)
        elif np.isinf(m2):
            x = b2
            y = m1 * x + b1
            return (x, y)
        elif m1 == m2:
            return None  # Parallel lines, no intersection
        else:
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1
            return (x, y)

    # Rectangle construction logic (matches test.ipynb)
    m1, b1 = line_eq(bl_pt, prev_pt)  # width line - front
    m2, b2 = line_eq(bl_pt, next_pt)  # length line - right
    m3, b3 = m1, next_pt[1] - m1 * next_pt[0]  # width line - back
    m4, b4 = m2, prev_pt[1] - m2 * prev_pt[0]  # length line - left
    m5, b5 = line_eq((x2-x1, 0), (x2-x1, y2-y1))  # height line - right
    m6, b6 = line_eq((0, 0), (0, y2-y1))         # height line - left

    left_corner = line_intersection(m1, b1, m6, b6)
    right_corner = line_intersection(m2, b2, m5, b5)

    if left_corner is not None and right_corner is not None:
        m3, b3 = m1, right_corner[1] - m1 * right_corner[0]  # width line - back
        m4, b4 = m2, left_corner[1] - m2 * left_corner[0]    # length line - left
        inferred_pt = line_intersection(m3, b3, m4, b4)
    else:
        inferred_pt = None

    # Map all corners to int and check validity
    if inferred_pt and left_corner and right_corner:
        inferred_pt = tuple(map(int, inferred_pt))
        corners_crop = [bl_pt, left_corner, inferred_pt, right_corner]
        corners_crop = [tuple(map(int, pt)) for pt in corners_crop]
    else:
        print("Could not infer 4th corner.")
        return None

    # Convert to global coordinates
    corners_global = [(pt[0] + x1, pt[1] + y1) for pt in corners_crop]

    if plot:
        car_crop = frame[y1:y2, x1:x2].copy()
        for i, pt in enumerate(corners_crop):
            cv2.circle(car_crop, pt, 7, (0, 255, 255), -1)
            cv2.putText(car_crop, f"V{i+1}", (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        for i in range(len(corners_crop)):
            pt1 = corners_crop[i]
            pt2 = corners_crop[(i+1)%len(corners_crop)]
            cv2.line(car_crop, pt1, pt2, (255, 0, 255), 2)
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB))
        plt.title("Completed Rectangle with 4 Corners")
        plt.axis('off')
        plt.show()

    return corners_global

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
    from scipy.spatial.distance import cdist

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
    mid_bottom = ((w-1)//2, h-1)
    left_top = (int(w*0.25), int(h*0.6))
    right_top = (int(w-1), int(h*0.6))

    line_left = np.polyfit([mid_bottom[0], left_top[0]], [mid_bottom[1], left_top[1]], 1)
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
    def polygon_area(pts):
        pts = np.array(pts)
        return 0.5 * np.abs(np.dot(pts[:,0], np.roll(pts[:,1], 1)) - np.dot(pts[:,1], np.roll(pts[:,0], 1)))
    area = polygon_area(corners_global)
    bbox_area = (x2-x1) * (y2-y1)
    if area < 0.05 * bbox_area or area > 1.5 * bbox_area:
        if debug:
            print(f"Unreasonable area: {area:.1f}, bbox_area={bbox_area:.1f}")
        return None

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

def get_vehicle_front_point(frame, corners, bbox, debug=False):
    """
    Given a frame, 4 corners (global coordinates), and bbox (x1, y1, x2, y2),
    returns the center (x, y) of the smallest detected blob in the vertical cut
    between the two lowest corners in the bbox region.
    """
    import cv2
    import numpy as np

    # Convert corners to local bbox coordinates
    x1, y1, x2, y2 = bbox
    local_corners = [(int(c[0] - x1), int(c[1] - y1)) for c in corners]
    # Sort by y (descending), so the largest y are at the front
    sorted_by_y = sorted(local_corners, key=lambda pt: pt[1], reverse=True)
    if len(sorted_by_y) < 2:
        return None

    # Compute vertical cut region between the two lowest corners
    dist = np.linalg.norm(np.array(sorted_by_y[0]) - np.array(sorted_by_y[1]))
    midpoint = np.mean(sorted_by_y[:2], axis=0)
    mx, my = int(midpoint[0]), int(midpoint[1])

    cropped_car = frame[y1:y2, x1:x2]
    h, w = cropped_car.shape[:2]
    # Ensure the cut is within image bounds
    x_start = max(mx - int(0.3 * dist), 0)
    x_end = min(mx + int(0.3 * dist), w)
    y_start = h // 2
    y_end = h
    vertical_cut = cropped_car[y_start:y_end, x_start:x_end]

    # Blob detection on the vertical cut
    if vertical_cut.size == 0:
        return None

    gray = cv2.cvtColor(vertical_cut, cv2.COLOR_BGR2GRAY) if vertical_cut.ndim == 3 else vertical_cut.copy()
    grey = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 5000
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(grey)

    # Find the smallest blob (by size)
    smallest_blob = None
    min_size = float('inf')
    for kp in keypoints:
        if kp.size < min_size:
            min_size = kp.size
            smallest_blob = kp

    if smallest_blob is not None:
        # Convert blob center back to full image coordinates
        bx = int(smallest_blob.pt[0]) + x_start + x1
        by = int(smallest_blob.pt[1]) + y_start + y1

        if debug:
            blob_vis = vertical_cut.copy()
            cv2.circle(blob_vis, (int(smallest_blob.pt[0]), int(smallest_blob.pt[1])), 6, (0, 255, 0), -1)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(blob_vis, cv2.COLOR_BGR2RGB))
            plt.title("Center of Smallest Blob (Green)")
            plt.axis('off')
            plt.show()
        return (bx, by)
    else:
        return None

# def get_vehicle_base_contour(mask, bbox, max_iters=20, debug=False):
#     """
#     Estimate vehicle base rectangle from mask using K-means-like bottom contour clustering.

#     Args:
#         mask (np.ndarray): Binary mask (H x W), dtype bool or uint8.
#         max_iters (int): Max iterations for clustering.
#         debug (bool): If True, prints debug info and shows intermediate results.

#     Returns:
#         List of 4 points (x, y) corresponding to bottom rectangle corners or None if failed.
#     """
#     x1, y1, x2, y2 = map(int, bbox)
#     # Step 1: Get bottom edge contour points using vertical gradient
#     mask_uint8 = mask.astype(np.uint8) * 255
#     grad_y = cv2.Sobel(mask_uint8, cv2.CV_64F, 0, 1, ksize=3)
#     bottom_edge = np.argwhere(grad_y > 0)
#     if debug:
#         print(f"Bottom edge points: {len(bottom_edge)}")
#     if len(bottom_edge) < 4:
#         return None
    
#     # Filter the bottom edge points - Uses MeanShift clustering to filter outliers
#     if len(bottom_edge) > 0:
#         # bottom_edge is (row, col) = (y, x)
#         points = np.flip(bottom_edge, axis=1)  # (x, y)
#         ms = MeanShift(bandwidth=20, bin_seeding=True)
#         ms.fit(points)
#         labels = ms.labels_
#         cluster_centers = ms.cluster_centers_

#     threshold = max(60, min(0.5 * (y2 - y1), 20))
#     threshold = 60

#     if len(cluster_centers) > 1:
#         dists = cdist(cluster_centers, cluster_centers)
#         np.fill_diagonal(dists, np.inf)
#         keep_mask = np.min(dists, axis=1) < threshold
#         cluster_centers = cluster_centers[keep_mask]
#         keep_labels = [i for i, keep in enumerate(keep_mask) if keep]
#         mask = np.isin(labels, keep_labels)
#         points = points[mask]
#         labels = labels[mask]
#         # Remap labels to be consecutive for plotting
#         unique_labels = {old: new for new, old in enumerate(sorted(set(labels)))}
#         labels = np.array([unique_labels[l] for l in labels])
#     else:
#         keep_mask = np.ones(len(cluster_centers), dtype=bool)

#     # --- Filter outliers within each cluster ---
#     filtered_points = []
#     filtered_labels = []
#     for cluster_id in np.unique(labels):
#         cluster_points = points[labels == cluster_id]
#         if len(cluster_points) < 3:
#             filtered_points.append(cluster_points)
#             filtered_labels.extend([cluster_id] * len(cluster_points))
#             continue
#         # Compute centroid
#         centroid = np.mean(cluster_points, axis=0)
#         dists = np.linalg.norm(cluster_points - centroid, axis=1)
#         std = np.std(dists)
#         # Keep points within 2 std of the mean distance
#         keep = dists < (np.mean(dists) + 1 * std)
#         filtered_points.append(cluster_points[keep])
#         filtered_labels.extend([cluster_id] * np.sum(keep))
#     filtered_points = np.vstack(filtered_points)
#     filtered_labels = np.array(filtered_labels)

#     # --- Enhanced Step 2: Initialize two lines using diagonals of the crop ---
#     h, w = mask.shape[:2]
#     diag1_pts = np.array([[0, 0], [w-1, h-1]], dtype=np.float32)
#     diag2_pts = np.array([[w-1, 0], [0, h-1]], dtype=np.float32)
#     line1 = np.polyfit(diag1_pts[:, 0], diag1_pts[:, 1], 1)  # y = a1*x + b1
#     line2 = np.polyfit(diag2_pts[:, 0], diag2_pts[:, 1], 1)  # y = a2*x + b2
#     lines = [line1, line2]

#     for _ in range(max_iters):
#         # Step 3: Assign points to nearest line (cluster)
#         clusters = [[] for _ in range(2)]
#         for pt in filtered_points:
#             dists = [np.abs(l[0]*pt[0] - pt[1] + l[1]) / np.sqrt(l[0]**2 + 1) for l in lines]
#             min_idx = np.argmin(dists)
#             clusters[min_idx].append(pt)

#         if debug:
#             print(f"Iteration {_}: cluster sizes = {[len(c) for c in clusters]}")

#         # Step 4: Refit lines from new clusters
#         new_lines = []
#         for pts in clusters:
#             if len(pts) < 2:
#                 if debug:
#                     print("Cluster too small:", [len(c) for c in clusters])
#                 return None
#             pts = np.array(pts)
#             line = np.polyfit(pts[:, 0], pts[:, 1], 1)
#             new_lines.append(line)
#         lines = new_lines

#     def line_intersection(m1, b1, m2, b2):
#         if np.isinf(m1) and np.isinf(m2):
#             return None  # Both lines vertical, no intersection
#         elif np.isinf(m1):
#             x = b1
#             y = m2 * x + b2
#             return (x, y)
#         elif np.isinf(m2):
#             x = b2
#             y = m1 * x + b1
#             return (x, y)
#         elif m1 == m2:
#             return None  # Parallel lines, no intersection
#         else:
#             x = (b2 - b1) / (m1 - m2)
#             y = m1 * x + b1
#             return (x, y)
        
#     def line_eq(p1, p2):
#             x1_, y1_ = p1
#             x2_, y2_ = p2
#             if x2_ != x1_:
#                 m_ = (y2_ - y1_) / (x2_ - x1_)
#                 b_ = y1_ - m_ * x1_
#             else:
#                 m_ = np.inf
#                 b_ = x1_
#             return m_, b_


#     m1, b1 = lines[1]
#     m2, b2 = lines[0]
#     bl_pt = line_intersection(m1, b1, m2, b2)

#     m5, b5 = line_eq((w-1, 0), (w-1, h-1))  # height line - right
#     m6, b6 = line_eq((0, 0), (0, h-1))         # height line - left

#     left_corner = line_intersection(m1, b1, m6, b6)
#     right_corner = line_intersection(m2, b2, m5, b5)
#     if left_corner is not None and right_corner is not None:
#         m3, b3 = m1, right_corner[1] - m1 * right_corner[0]  # width line - back
#         m4, b4 = m2, left_corner[1] - m2 * left_corner[0]    # length line - left
#         inferred_pt = line_intersection(m3, b3, m4, b4)
#     else:
#         inferred_pt = None

#     # Map all corners to int and check validity
#     if inferred_pt and left_corner and right_corner:
#         inferred_pt = tuple(map(int, inferred_pt))
#         corners_crop = [bl_pt, left_corner, inferred_pt, right_corner]
#         corners_crop = [tuple(map(int, pt)) for pt in corners_crop]
#     else:
#         print("Could not infer 4th corner.")

#     return corners_crop



# def get_bottom_rectangle_corners(frame, mask, bbox, plot=False):
#     """
#     Approximates the bottom half of the SAM2 mask outline as a polygon,
#     finds the bottom left corner and the two lines intersecting at that corner,
#     closes the rectangle, and returns the 4 corners in global image coordinates.

#     Args:
#         frame (np.ndarray): The original image.
#         mask (np.ndarray): The binary mask (same size as frame).
#         bbox (tuple): (x1, y1, x2, y2) bounding box.
#         plot (bool): If True, plot the result on the cropped image.

#     Returns:
#         corners_global (list of tuple): List of 4 (x, y) points in global image coordinates.
#     """
#     x1, y1, x2, y2 = bbox
#     mask_crop = mask[y1:y2, x1:x2].astype(np.uint8)

#     contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         print("No contours found in mask.")
#         return None

#     largest_contour = max(contours, key=cv2.contourArea)
#     h_crop = mask_crop.shape[0]
#     bottom_half_points = largest_contour[largest_contour[:, 0, 1] > h_crop // 3]

#     if len(bottom_half_points) >= 3:
#         epsilon = 0.02 * cv2.arcLength(bottom_half_points, True)
#         approx_poly = cv2.approxPolyDP(bottom_half_points, epsilon, True)
#     else:
#         approx_poly = bottom_half_points

#     pts = approx_poly[:, 0, :] if approx_poly.ndim == 3 else approx_poly[:, 0, :]
#     if len(pts) < 3:
#         print("Not enough points to form a polygon.")
#         return None

#     idx_bl = np.lexsort((pts[:, 0], -pts[:, 1]))[0]
#     bl_pt = tuple(pts[idx_bl])
#     n = len(pts)
#     prev_idx = (idx_bl - 1) % n
#     next_idx = (idx_bl + 1) % n
#     prev_pt = tuple(pts[prev_idx])
#     next_pt = tuple(pts[next_idx])

#     def line_eq(p1, p2):
#         x1_, y1_ = p1
#         x2_, y2_ = p2
#         if x2_ != x1_:
#             m_ = (y2_ - y1_) / (x2_ - x1_)
#             b_ = y1_ - m_ * x1_
#         else:
#             m_ = np.inf
#             b_ = x1_
#         return m_, b_

#     m1, b1 = line_eq(bl_pt, prev_pt)
#     m2, b2 = line_eq(bl_pt, next_pt)
#     m3 = m1
#     m4 = m2

#     b3 = next_pt[1] - m3 * next_pt[0]
#     b4 = prev_pt[1] - m4 * prev_pt[0]

#     # Check for parallel or invalid lines
#     if not np.isfinite(m3) or not np.isfinite(m4) or abs(m3 - m4) < 1e-8:
#         print("Lines are parallel or vertical, cannot find intersection.")
#         return None

#     x = (b4 - b3) / (m3 - m4)
#     y = m3 * x + b3

#     if not np.isfinite(x) or not np.isfinite(y):
#         print("Intersection is not finite.")
#         return None

#     inferred_pt = (x, y)

#     # Corners in cropped coordinates
#     corners_crop = [bl_pt, prev_pt, tuple(map(int, inferred_pt)), next_pt]
#     # Convert to global coordinates
#     corners_global = [(pt[0] + x1, pt[1] + y1) for pt in corners_crop]

#     if plot:
#         car_crop = frame[y1:y2, x1:x2].copy()
#         for i, pt in enumerate(corners_crop):
#             cv2.circle(car_crop, tuple(map(int, pt)), 7, (0, 0, 255), -1)
#             cv2.putText(car_crop, f"V{i+1}", (int(pt[0])+5, int(pt[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
#         for i in range(4):
#             pt1 = tuple(map(int, corners_crop[i]))
#             pt2 = tuple(map(int, corners_crop[(i+1)%4]))
#             cv2.line(car_crop, pt1, pt2, (255, 0, 255), 2)
#         plt.figure(figsize=(8, 8))
#         plt.imshow(cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB))
#         plt.title("Bottom Rectangle from Polygon: All 4 Vertices")
#         plt.axis('off')
#         plt.show()

#     return corners_global


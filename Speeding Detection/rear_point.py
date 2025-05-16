import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

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

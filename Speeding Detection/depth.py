import cv2
import torch
import numpy as np
from torch.cuda.amp import autocast  # optional for speedup

def find_front_rear_points(frame, sam_mask, bbox, depth_model):
    """
    Finds the front (closest) and rear (farthest) points inside a SAM mask
    using the DAM2 depth prediction.

    Args:
        frame (np.ndarray): The full RGB frame.
        sam_mask (np.ndarray): The binary mask for the object (bool type).
        bbox (tuple): (x1, y1, x2, y2) bounding box coordinates.
        depth_model (torch.nn.Module): The DAM2 depth model.

    Returns:
        front_point (tuple): (x, y) coordinates of the front point (closest to camera).
        rear_point (tuple): (x, y) coordinates of the rear point (farthest from camera).
    """
    x1, y1, x2, y2 = bbox

    # 1. Crop bbox region
    bbox_crop = frame[y1:y2, x1:x2]
    if bbox_crop is None or bbox_crop.size == 0:
        return None, None, None


    # 3. Run DAM2 depth prediction
    with torch.no_grad():
        depth_pred = depth_model.infer_image(bbox_crop)

    # 4. Resize SAM2 mask to DAM2 input size
    sam_mask_crop = sam_mask[y1:y2, x1:x2]
    sam_mask_resized = cv2.resize(sam_mask_crop.astype(np.uint8), (depth_pred.shape[1], depth_pred.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 5. Apply mask to depth
    masked_depth = np.where(sam_mask_resized, depth_pred, np.nan)

    # 6. Find valid (non-NaN) points
    valid_y, valid_x = np.where(~np.isnan(masked_depth))
    print(f"[DEBUG] Valid masked depth points: {len(valid_y)}")

    if len(valid_y) == 0:
        print("[WARNING] No valid depth points inside SAM2 mask!")
        return None, None, None

    if len(valid_x) == 0:
        return None, None  # No valid points

    depths = masked_depth[valid_y, valid_x]

    # Normalize depth
    norm_depths = (depths - np.nanmin(depths)) / (np.nanmax(depths) - np.nanmin(depths))

    # Compute distances to bbox center
    cx = depth_pred.shape[1] / 2
    cy = depth_pred.shape[0] / 2
    distances = np.sqrt((valid_x - cx) ** 2 + (valid_y - cy) ** 2)
    norm_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

    # Heuristic score (adjust alpha/beta to taste) Currently only looks at depth
    alpha = 1
    beta = 0
    rear_scores = alpha * norm_depths + beta * (1 - norm_distances)
    front_scores = (1 - alpha) * (1 - norm_depths) + beta * (1 - norm_distances)

    # Select best points
    rear_idx = np.argmax(rear_scores)
    front_idx = np.argmax(front_scores)

    rear_px, rear_py = valid_x[rear_idx], valid_y[rear_idx]
    front_px, front_py = valid_x[front_idx], valid_y[front_idx]


    # 8. Map back to original bbox/frame
    scale_x = (x2 - x1) / depth_pred.shape[1]
    scale_y = (y2 - y1) / depth_pred.shape[0]

    front_point = (int(x1 + front_px * scale_x), int(y1 + front_py * scale_y))
    rear_point  = (int(x1 + rear_px * scale_x), int(y1 + rear_py * scale_y))

    depth_norm = cv2.normalize(depth_pred, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)


    return front_point, rear_point, depth_colored
import os
import sys
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from ultralytics import solutions
from yolo_detector import YoloDetector
from dpsort_tracker import DeepSortTracker
import calibration
from calibration import findROIUser
from calibration import CalculateHomography
from ultralytics import SAM
from depth import find_front_rear_points
from rear_point import find_rear_point_from_mask_and_depth, find_front_point_1

dam2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Depth-Anything-V2'))
sys.path.append(dam2_path)
from depth_anything_v2.dpt import DepthAnythingV2


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
depth_model = DepthAnythingV2(**model_configs[encoder])
depth_model.load_state_dict(torch.load(f'C:/Users/bahaa/CapstoneProject/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_model = depth_model.to('cuda').eval()

MODEL_PATH = r"C:\Users\bahaa\YOLO\yolo11x.pt"
VIDEOS_FOLDER = r"C:\DataSets\I5Videos\video"
OUTPUT_FOLDER = r"C:\Users\bahaa\YOLO\results"
Video1 = r"C:\DataSets\4K Road traffic video for object detection and tracking - free download now.mp4" # 4k Video (outside US)
Video2 = r"C:\DataSets\I5Videos\video\cctv052x2004080606x01827.avi" # Video from I5 dataset
Video3 = r"C:\DataSets\VecteezyTrafficFlow\vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4"
Video4 = r"C:\DataSets\SeattleStreetVideo.mp4" # Video from downtown seattle - car movement perpendicular to camera lens
Video5 = r"C:\DataSets\Seattle Crash Video Editted.mp4"
Video6 = r"C:\Users\bahaa\Downloads\IMG_1662 (1).MOV"
Video7 = r"C:\Users\bahaa\Downloads\Untitled video - Made with Clipchamp (3).mp4"

VIDEO_PATH = Video7 # Change this to the path of your video file

def apply_homography_point(point, H):
    px = np.array([[point]], dtype=np.float32)  # Shape: (1, 1, 2)
    transformed = cv2.perspectiveTransform(px, H)
    return tuple(transformed[0][0])

def get_stable_car(track_history):
    """Select car with lowest variance in pixel length and enough samples."""
    best_id = None
    best_score = float('inf')
    best_lengths = []
    for tid, hist in track_history.items():
        lengths = np.array(hist['pixel_length'])
        n = len(lengths)
        if n < 10:  # Require enough samples
            continue
        var = np.var(lengths)
        score = var/n
        if score < best_score:
            best_score = score
            best_id = tid
            best_lengths = lengths
    return best_id, best_lengths

def filter_outliers(lengths, thresh=2.0):
    """Remove outliers using z-score."""
    mean = np.mean(lengths)
    std = np.std(lengths)
    filtered = lengths[np.abs((lengths - mean) / (std + 1e-8)) < thresh]
    return filtered

def crop_car_image(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return frame[y1:y2, x1:x2]

def query_vlm_for_length(image_path):
    # Placeholder: Replace with actual VLM call
    # For example, return 4.8 for a sedan in meters
    return 4.8

# ---- Stitch depth maps into a grid (auto fit) ----
def tile_images(images, max_width=1280):
    if len(images) == 0:
        return None

    # Resize all to same size (e.g., 256x256)
    resized = [cv2.resize(img, (256, 256)) for img in images]

    # Compute grid size (square-ish)
    n = len(resized)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # Fill grid
    blank = np.zeros_like(resized[0])
    padded = resized + [blank] * (rows * cols - n)
    grid_rows = [cv2.hconcat(padded[i*cols:(i+1)*cols]) for i in range(rows)]
    grid_image = cv2.vconcat(grid_rows)

    return grid_image


def slice_mask_along_pca(sam_mask, bbox, above_ratio=0.3):
    """
    Slices a SAM mask along its PCA direction such that `above_ratio` of points
    lie above the slicing line (shifted along PCA normal).

    Args:
        sam_mask (np.ndarray): Full-frame binary mask.
        bbox (tuple): (x1, y1, x2, y2) bounding box.
        above_ratio (float): Fraction of mask points to lie above the slicing line.

    Returns:
        sliced_mask (np.ndarray): Boolean mask with 1s for kept (lower) region.
    """
    x1, y1, x2, y2 = bbox
    cropped = sam_mask[y1:y2, x1:x2].astype(np.uint8)

    ys, xs = np.nonzero(cropped)
    if len(xs) < 2:
        return np.zeros_like(sam_mask, dtype=bool), None, None

    points = np.column_stack((xs, ys)).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(points, mean=None, maxComponents=1)

    # PCA direction and normal
    dx, dy = eigenvectors[0]
    direction = np.array([dx, dy])
    normal = np.array([-dy, dx])
    normal /= np.linalg.norm(normal)

    # if normal[1] < 0:  # Y is increasing downward, so this points down
    #     normal = -normal

    # Project all points onto the normal
    rel_points = points - mean  # shift origin to PCA center
    projections = rel_points @ normal  # dot product along normal

    # Find cutoff to retain 70% below (i.e., 30% above)
    threshold = np.percentile(projections, above_ratio * 100)

    # Keep points below the threshold
    keep = projections >= threshold
    kept_points = points[keep]
    kept_xs, kept_ys = kept_points[:, 0].astype(int), kept_points[:, 1].astype(int)

    # Reconstruct sliced mask
    new_cropped = np.zeros_like(cropped, dtype=bool)
    new_cropped[kept_ys, kept_xs] = True

    # Embed back in full-frame mask
    sliced_mask = np.zeros_like(sam_mask, dtype=bool)
    sliced_mask[y1:y2, x1:x2] = new_cropped

    # Project slicing line back to image space
    shift_vec = normal * threshold
    shifted_center = mean[0] + shift_vec  # (x, y)

    # Get line endpoints along the PCA direction
    line_len = 1000  # length to draw
    p1 = shifted_center - direction * line_len / 2
    p2 = shifted_center + direction * line_len / 2

    # Convert to integer and shift back to frame coordinates
    p1 = (int(p1[0] + x1), int(p1[1] + y1))
    p2 = (int(p2[0] + x1), int(p2[1] + y1))

    return sliced_mask, p1, p2

# def fit_car_direction_line(sam_mask, bbox):
#     """
#     Fit a best-fit line through the SAM2 mask within a bounding box.

#     Args:
#         sam_mask (np.ndarray): binary mask from SAM2.
#         bbox (tuple): (x1, y1, x2, y2) bounding box coordinates.

#     Returns:
#         origin (tuple): bottom-left corner of the bbox (x1, y2).
#         direction (tuple): unit vector (dx, dy) representing car orientation.
#     """
#     x1, y1, x2, y2 = bbox
#     cropped_mask = sam_mask[y1:y2, x1:x2].astype(np.uint8)
#     ys, xs = np.nonzero(cropped_mask)

#     if len(xs) < 2:
#         return None, None

#     points = np.column_stack((xs, ys)).astype(np.float32)

#     mean, eigenvectors = cv2.PCACompute(points, mean=None, maxComponents=1)
#     direction = eigenvectors[0]
#     direction /= np.linalg.norm(direction)

#     origin = (x1, int(y2 + 0.6*(y1-y2)))  # bottom-left of bbox

#     return origin, direction


def main():
    
    # Uncomment if looking at a single video instead of a file of videos
    cap = cv2.VideoCapture(VIDEO_PATH)

    _ , roi_image = cap.read() # Extract a snapshot for road detection
    # roi_image = cv2.resize(roi_image, (960, 540))
    roi_points, roi_mask = findROIUser(roi_image)
    H, _ = CalculateHomography(roi_image, roi_points)
    
    
    detector = YoloDetector(model_path = MODEL_PATH, confidence=0.7)
    tracker = DeepSortTracker()
    sam_model = SAM("sam2.1_b.pt")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format

    # Output video: 3x stacked (960 each)
    output_video_path = os.path.join(OUTPUT_FOLDER, "combined_output_PCA_v7.mp4")
    combined_size = (960 * 3, 540)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, combined_size)

    # Initialize track history and speed calculator
    track_history = {}
    frame_count = 0
    prev_frame_time = time.perf_counter()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 40 * 30 + 5)

    while True:
        ret, frame = cap.read()
        h, w = frame.shape[:2]

        if not ret:
            break
        
        depth_visuals = []
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
            sliced_sam_mask, p1, p2 = slice_mask_along_pca(sam_mask, [x1, y1, x2, y2], above_ratio=0.9)
            # Visualize the sliced mask
            sliced_vis = frame.copy()
            sliced_vis[sliced_sam_mask] = [255, 0, 255]
            cv2.imshow("Sliced SAM Mask", sliced_vis)
            # cv2.line(frame, p1, p2, (255, 0, 255), 2)


            contour_groups = sam_result[0].masks.xy
            if contour_groups:
                for contour_group in contour_groups:
                    for contour in contour_group:
                        contour = np.array(contour, dtype=np.int32)
                        
                        # if contour.ndim != 2 or contour.shape[0] < 3:
                        #     continue

                        contour = contour.reshape((-1, 1, 2))
                        cv2.polylines(frame, [contour], isClosed=True, color=(0, 255, 0), thickness=2)

            # front_point, rear_point, depth_viz = find_front_rear_points(frame, sliced_sam_mask, (x1, y1, x2, y2), depth_model)
            rear_point,_,_, rear_cluster_points = find_rear_point_from_mask_and_depth(frame, sam_mask, (x1, y1, x2, y2), depth_model)
            front_point_1 = find_front_point_1(frame, sam_mask, (x1, y1, x2, y2), depth_model, rear_point)
            # origin, direction = fit_car_direction_line(sam_mask, (x1, y1, x2, y2))
            # if origin and direction is not None:
            #     ox, oy = origin
            #     dx, dy = direction
            #     scale = 100  # length of line to draw
            #     end_x = int(ox + dx * scale)
            #     end_y = int(oy + dy * scale)
            #     cv2.arrowedLine(frame, (ox, oy), (end_x, end_y), (255, 255, 0), 2, tipLength=0.2)
            
            for pt in rear_cluster_points:
                cv2.circle(frame, pt, 2, (0, 255, 255), -1)  # yellow for cluster

            if front_point_1 is not None and rear_point is not None:
                # depth_visuals.append(depth_viz)
                rear_point_bev = apply_homography_point(rear_point, H)
                front_point_bev = apply_homography_point(front_point_1, H)

                px_vehicle_length = abs(rear_point_bev[1] - front_point_bev[1])

                # Visualization
                cv2.circle(frame, rear_point, 5, (0, 0, 255), -1)
                cv2.circle(frame, front_point_1, 5, (0, 255, 0), -1)
                cv2.line(frame, rear_point, front_point_1, (255, 0, 0), 2)

                length_text = f"Len: {px_vehicle_length:.1f}px"
                cv2.putText(
                    frame, 
                    length_text, 
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0), 2
                )

                # Track history update
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if tracking_id not in track_history:
                    track_history[tracking_id] = {
                        'positions': [],
                        'timestamps': [],
                        'speeds': [],
                        'mask': [],
                        'pixel_length': [],
                    }

                track_history[tracking_id]['positions'].append((frame_count, (x1, y1, x2, y2)))
                track_history[tracking_id]['timestamps'].append(current_time)
                track_history[tracking_id]['pixel_length'].append(px_vehicle_length)
                # track_history[tracking_id]['mask'].append(sam_mask)

            colored_mask = frame.copy()
            colored_mask[sam_mask] = [0, 0, 255]  # Red color for the mask

            # Draw sam mask, bounding box, and ID
            #frame[sam_mask] = cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)[sam_mask]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"ID: {tracking_id}", 
                       (x1, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 0, 255), 2)
            
        depth_frame = tile_images(depth_visuals)

        # if depth_frame is not None:
            # cv2.imshow("Per-Car Depth Maps", depth_frame)

        # Calculate processing FPS
        processing_fps = 1 / (current_time - prev_frame_time)
        prev_frame_time = current_time
        cv2.putText(frame, f"FPS: {processing_fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 255, 0), 2)

        # # Display and output
        # display_frame = cv2.resize(frame, (960, 540))
        # cv2.imshow("output_normal", display_frame)
        # display_frame = calibration.BEVTransform(display_frame,H)
        # cv2.imshow("output_BEV", display_frame)
        
        # Resize everything to same height
        frame_resized = cv2.resize(frame, (960, 540))
        bev_resized = calibration.BEVTransform(frame_resized.copy(), H)
        depth_resized = cv2.resize(depth_frame, (960, 540)) if depth_frame is not None else np.zeros_like(frame_resized)

        # Horizontally stack
        combined_frame = cv2.hconcat([frame_resized, bev_resized, depth_resized])

        # Show preview (optional)
        cv2.imshow("Combined Output", combined_frame)

        video_writer.write(combined_frame)  # Write to output video

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()
    if track_history:
        # 1. Find the most stable car
        best_id, best_lengths = get_stable_car(track_history)
        if best_id is not None:
            print(f"Selected car ID: {best_id} with {len(best_lengths)} samples")
            # 2. Get the frame index and bbox for the best sample (e.g., the median sample)
            positions = track_history[best_id]['positions']
            # Use the median sample for robustness
            idx = len(positions) // 2
            idx = min(0, len(positions) - 20)  # Ensure idx is within bounds
            frame_idx, bbox = positions[idx]
            x1, y1, x2, y2 = bbox

            # 3. Reopen the video and seek to the frame
            cap2 = cv2.VideoCapture(VIDEO_PATH)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, selected_frame = cap2.read()
            cap2.release()
            if not ret:
                print(f"Failed to read frame {frame_idx} for cropping.")
            else:
                # 4. Crop and save image
                cropped = crop_car_image(selected_frame, (x1, y1, x2, y2))
                crop_path = os.path.join(OUTPUT_FOLDER, f"car_{best_id}_crop.jpg")
                cv2.imwrite(crop_path, cropped)
                # 5. Query VLM for real-world length
                real_length_m = 1.0  # Placeholder: Replace with actual VLM call
                # 6. Filter outliers and average
                filtered_lengths = filter_outliers(best_lengths)
                avg_pixel_length = np.mean(filtered_lengths)
                pixel_to_meter = avg_pixel_length / real_length_m
                # 7. Save to file
                ratio_path = os.path.join(OUTPUT_FOLDER, "pixel_to_meter_ratio.json")
                with open(ratio_path, "w") as f:
                    json.dump({
                        "pixel_to_meter": pixel_to_meter,
                        "homography": H.tolist(),
                        "roi_points": roi_points,
                        "homography": H.tolist(),
                        "pixel_to_meter": pixel_to_meter
                    }, f, indent=2)
                print(f"Pixel-to-meter ratio saved: {pixel_to_meter:.4f}")
        else:
            print("No stable car found for calibration.")
    else:
        print("No cars tracked.")


if __name__ == "__main__":
    main()



import cv2
import os
from ultralytics import solutions
import numpy as np
import matplotlib.pyplot as plt
from edge import canny
from edge import hough_transform
import filters
#from torchvision.models.segmentation import deeplabv3_resnet
from yolo_detector import YoloDetector
from dpsort_tracker import DeepSortTracker
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor



#IMG_PATH = r"C:\DataSets\SeattleStreet.png"

DATASET_PATH = "C:\DataSets"
ROADS_FOLDER = os.path.join(DATASET_PATH, "RoadsCropped")

# Initialize SegFormer-b5 (higher accuracy, larger model)
model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
processor = SegformerImageProcessor(
    do_resize=True,
    do_normalize=True,
    do_random_crop=False,
    do_pad=False
)
model.eval()  # Disable dropout layers

# cap = cv2.VideoCapture(Video1)
# assert cap.isOpened(), "Error reading video file"
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
#video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# In case you want to apply object counting + heatmaps, you can pass region points.
# region_points = [(20, 400), (1080, 400)]  # Define line points
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # Define region points
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]  # Define polygon points

# # Init heatmap
# heatmap = solutions.Heatmap(
#     show=True,  # Display the output
#     model=MODEL_PATH,  # Path to the YOLO11 model file
#     colormap=cv2.COLORMAP_PARULA,  # Colormap of heatmap
#     # region=region_points,  # If you want to do object counting with heatmaps, you can pass region_points
#     # classes=[0, 2],  # If you want to generate heatmap for specific classes i.e person and car.
#     # show_in=True,  # Display in counts
#     # show_out=True,  # Display out counts
#     # line_width=2,  # Adjust the line width for bounding boxes and text display
# )

# Process video
# while cap.isOpened():
#     success, im0 = cap.read()
#     if not success:
#         print("Video frame is empty or video processing has been successfully completed.")
#         break
#     # im0 = heatmap.generate_heatmap(im0)
#     # ideo_writer.write(im0)

# cap.release()
# #video_writer.release()
# cv2.destroyAllWindows()

# Function to find the region of interest in a video assuming the camera is stationary and is looking at a road.
# The region of interest is defined as part of the road plane where most of the car detections (Using YOLOv11) happen 
# and car trackings (Using DeepSort) are stable. The function should return the rectangular region in the image that covers part of the road
# Overall Initial Approach:
# Run Road segmentation Algorithm:
        # 1. Otsu's Thresholding to make road plane pop out in the image
        # 2. Run Canny Edge Detector to detect edges in the frame
        # 3. Run Hough Transform to find edges corrosponding to road edges
        # 4. Find Region where Detections are maximum and Tracking_IDs are mostly stable

# def FindROI(images, Detections, Tracking_IDs):
#     for image in images:
#         # Run Road segmentation Algorithm:
#         # 1. Otsu's Thresholding to make road plane pop out in the image
#         # 2. Run Canny Edge Detector to detect edges in the frame
#         # 3. Run Hough Transform to find edges corrosponding to road edges
#         # 4. Find Region where Detections are maximum and Tracking_IDs are mostly stable

def filter_short_lines(lines, min_length=50):
    """ Removes lines that are too short to be meaningful road edges. """
    filtered_lines = []
    for slope, intercept in lines:
        x1, y1 = 0, int(intercept)
        x2, y2 = 800, int(slope * 800 + intercept)  # Assume width = 800 pixels

        # Compute Euclidean distance of the line
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if length > min_length:
            filtered_lines.append((slope, intercept))

    return filtered_lines

def filter_invalid_slopes(lines, min_slope=0.3, max_slope=3.5):
    """ Filters out lines that are too steep (|slope| > max) or too flat (|slope| < min). """
    return [(slope, intercept) for slope, intercept in lines if min_slope < abs(slope) < max_slope]

def merge_lines(lines, max_gap=30, max_angle=5):
    """Merge similar lines using DBSCAN clustering"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    
    # Convert lines to angle-distance representation
    line_params = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        angle = np.arctan2(dy, dx)  # Range: (-π/2, π/2)
        length = np.sqrt(dx**2 + dy**2)
        line_params.append([angle, length])  # Use angle + length features
    
    # Normalize features
    scaler = StandardScaler()
    scaled_params = scaler.fit_transform(line_params)
    
    # Cluster lines with single eps value
    db = DBSCAN(eps=0.5, min_samples=1).fit(scaled_params)  # Tune eps between 0.3-1.0
    
    # Merge clusters
    merged = []
    for label in set(db.labels_):
        if label == -1: continue
        cluster = np.array(lines)[np.where(db.labels_ == label)]
        x_points = np.concatenate(cluster[:,:,[0,2]])
        y_points = np.concatenate(cluster[:,:,[1,3]])
        merged.append([
            int(x_points.min()), int(y_points.min()),
            int(x_points.max()), int(y_points.max())
        ])
    
    return merged

def get_driveway_mask(img):
    """
    Lets users draw polygons around driveways using mouse clicks.
    Returns a binary mask where driveways are marked (255 = driveway).
    """
    driveway_mask = np.zeros_like(img[:, :, 0])
    polygons = []
    current_poly = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_poly
        if event == cv2.EVENT_LBUTTONDOWN:
            current_poly.append((x, y))
            # Draw the point
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Mark Driveways", clone)
            
        elif event == cv2.EVENT_RBUTTONDOWN and current_poly:
            # Close the polygon
            current_poly.append(current_poly[0])
            polygons.append(np.array(current_poly, dtype=np.int32))
            # Draw the polygon
            cv2.polylines(clone, [np.array(current_poly)], True, (0, 255, 0), 2)
            cv2.imshow("Mark Driveways", clone)
            current_poly = []

    clone = img.copy()
    cv2.namedWindow("Mark Driveways")
    cv2.setMouseCallback("Mark Driveways", mouse_callback)

    while True:
        cv2.imshow("Mark Driveways", clone)
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'c' to close current polygon
        if key == ord('c') and len(current_poly) > 2:
            current_poly.append(current_poly[0])
            polygons.append(np.array(current_poly, dtype=np.int32))
            cv2.polylines(clone, [np.array(current_poly)], True, (0, 255, 0), 2)
            current_poly = []
            
        # Press 'q' to finish
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    
    # Create mask from polygons
    for poly in polygons:
        cv2.fillPoly(driveway_mask, [poly], 255)
        
    return driveway_mask

def findRoadPlane(img, kernelSize=3, Sigma=5.0, High=0.6, Low=0.4, numLines=6):
    """
    Detects road edges using Canny edge detection and Hough Transform.
    Displays exactly 6 detected straight lines on the image.
    """
    # Apply Canny edge detection
    edges = canny(img, kernel_size=kernelSize, sigma=Sigma, high=High, low=Low)

    # Perform Hough Transform
    acc, rhos, thetas = hough_transform(edges)

    # Store detected lines
    detected_lines = []

    # Extract 6 strongest peaks from the Hough accumulator
    num_lines = numLines
    peak_threshold = 0.5 * np.max(acc)  

    for _ in range(num_lines):
        idx = np.argmax(acc)  # Find peak index
        if acc[idx // acc.shape[1], idx % acc.shape[1]] < peak_threshold:
            break  # Stop if remaining peaks are too weak
        r_idx, t_idx = divmod(idx, acc.shape[1])  # Convert to 2D indices
        acc[r_idx, t_idx] = 0  # Suppress this peak to avoid duplicate detection
        acc[r_idx, max(t_idx-5, 0):t_idx+5] = 0  # Suppress nearby peaks

        rho = rhos[r_idx]
        theta = thetas[t_idx]

        # Convert Hough parameters to Cartesian line representation
        if np.sin(theta) != 0:  # Avoid division by zero
            slope = - (np.cos(theta) / np.sin(theta))  # Line slope
            intercept = (rho / np.sin(theta))  # y-intercept
            detected_lines.append((slope, intercept))
    
    detected_lines = filter_invalid_slopes(detected_lines, min_slope=0.001, max_slope=np.tan(np.deg2rad(80)))
    detected_lines = filter_short_lines(detected_lines, min_length=50)

    # Convert grayscale image to color for visualization
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Function to draw detected lines
    def draw_lines(lines, color):
        for slope, intercept in lines:
            x1, y1 = 0, int(intercept)
            x2, y2 = img.shape[1], int(slope * img.shape[1] + intercept)
            cv2.line(color_img, (x1, y1), (x2, y2), color, 3)

    # Draw 6 detected lines in different colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]  # 6 distinct colors

    for i, line in enumerate(detected_lines):
        draw_lines([line], colors[i % len(colors)])  

    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    edges = (edges * 255).astype("uint8")

    return color_img, edges  # Return color_img and edges for debugging

def findROIBoundingBoxes(VIDEO_PATH, YOLO_PATH, threshold_ratio=0.1):
    """ 
    Detects road edges by masking out car regions and analyzing accumulated road pixels.
    """
    # Initialize models and video capture
    detector = YoloDetector(model_path=YOLO_PATH, confidence=0.7)
    tracker = DeepSortTracker()
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Initialize road probability map
    ret, target_img = cap.read()
    if not ret:
        raise ValueError("Failed to read video.")
    H, W = target_img.shape[:2]
    road_probability = np.zeros((H, W), dtype=np.float32)

    # Accumulate road regions over frames
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and track cars
        detections = detector.detect(frame)
        tracking_ids, boxes = tracker.track(detections, frame)
        
        # Create a mask for cars in this frame
        car_mask = np.zeros((H, W), dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(car_mask, (x1, y1), (x2, y2), 255, cv2.FILLED)
        
        # Update road probability: areas WITHOUT cars get higher weights
        road_probability += (1 - car_mask / 255.0)  # Inverse logic
        
        frame_count += 1

    cap.release()

    # Normalize and threshold road probability
    road_probability = (road_probability / frame_count) * 255
    road_probability = road_probability.astype(np.uint8)
    _, road_mask = cv2.threshold(road_probability, int(threshold_ratio * 255), 255, cv2.THRESH_BINARY)

    # Cleanup mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Edge detection ON ROAD REGIONS ONLY
    gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=road_mask)
    
    # Adaptive Canny thresholds
    median = np.median(masked_gray[masked_gray > 0])
    lower = int(max(0, 0.6 * median))
    upper = int(min(255, 1.4 * median))
    edges = cv2.Canny(masked_gray, lower, upper)

    # Probabilistic Hough with adaptive params
    lines = cv2.HoughLinesP(
        edges, 
        rho=2, 
        theta=np.pi/180, 
        threshold=50, 
        minLineLength=100, 
        maxLineGap=30
    )

    # Visualize results
    output_img = target_img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return output_img, lines

def findROIDeepLearning(img):
    # Get refined road mask using SegFormer
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(**inputs.to(model.device))
    
    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=img.shape[:2],
        mode='bilinear',
        align_corners=False
    )
    
    mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    road_mask = (mask == 0).astype(np.uint8) * 255

    # 1. Mask refinement using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Get user-defined driveway mask
    driveway_mask = get_driveway_mask(img)
    
    # Exclude driveways from road mask
    road_mask = cv2.bitwise_and(road_mask, cv2.bitwise_not(driveway_mask))

    # 2. Edge detection with noise reduction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), sigmaX=2, sigmaY=2)
    masked_img = cv2.bitwise_and(blurred, blurred, mask=road_mask)

    # 3. Adaptive Canny with stricter thresholds
    median = np.median(masked_img[masked_img > 0])
    lower = int(max(50, 0.7 * median))  # Higher minimum threshold
    upper = int(min(200, 1.3 * median))
    edges = cv2.Canny(masked_img, lower, upper)

    # 4. Road-aligned Hough transform filtering
    lines = cv2.HoughLinesP(
        edges, 
        rho=2, 
        theta=np.pi/180, 
        threshold=75,  # Increased threshold
        minLineLength=150,  # Longer minimum length
        maxLineGap=20      # Smaller gap tolerance
    )

    # 5. Geometric filtering of detected lines
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Filter by line orientation (keep horizontal-ish lines)
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180/np.pi)
            if not (160 > angle > 20):  # Adjust based on road geometry
                continue
                
            # Verify line lies within road mask
            points = np.linspace([x1,y1], [x2,y2], num=10)
            road_points = sum(road_mask[int(p[1]), int(p[0])] > 0 for p in points)
            if road_points >= 7:  # At least 70% on road
                filtered_lines.append(line)

    # 6. Line merging and visualization
    output_img = img.copy()
    if filtered_lines:
        # Merge nearby lines (optional)
        merged_lines = merge_lines(filtered_lines, max_gap=30, max_angle=10)
        
        for line in merged_lines:
            x1, y1, x2, y2 = line
            cv2.line(output_img, (x1,y1), (x2,y2), (0,255,0), 2)

    return road_mask, output_img


def findROIUser(img):
    """
    Lets users draw a polygon to define the road region of interest (ROI).
    Returns the polygon points and a mask of the selected region.
    """
    roi_points = []
    current_point = []
    mask = np.zeros_like(img[:, :, 0])
    clone = img.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_points, current_point, clone
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(roi_points) < 4:  # We need exactly 4 points for homography
                roi_points.append((x, y))
                current_point = [(x, y)]
                
                # Draw the point and lines
                if len(roi_points) > 1:
                    cv2.line(clone, roi_points[-2], roi_points[-1], (0,255,0), 2)
                cv2.circle(clone, (x,y), 5, (0,0,255), -1)
                cv2.imshow("Define Road ROI", clone)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Reset selection
            roi_points = []
            current_point = []
            clone = img.copy()
            cv2.imshow("Define Road ROI", clone)

    cv2.namedWindow("Define Road ROI")
    cv2.setMouseCallback("Define Road ROI", mouse_callback)

    while True:
        cv2.imshow("Define Road ROI", clone)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or len(roi_points) == 4:
            break

    cv2.destroyAllWindows()
    
    # Create mask from polygon
    if len(roi_points) == 4:
        cv2.fillPoly(mask, [np.array(roi_points, dtype=np.int32)], 255)
    
    return roi_points, mask

def CalculateHomography(img, roi_points):
    """
    Calculates the homography matrix for BEV transformation based on user-defined points.
    User will click 4 points in this order: top-left, top-right, bottom-right, bottom-left
    """
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Define destination points (BEV rectangle)
    side_margin = int(w * 0.1)  # 10% margin
    bev_width = w - 2*side_margin
    bev_height = int(bev_width * 0.6)  # Aspect ratio 1:0.6
    
    dst_points = np.float32([
        [side_margin, 0],           # Top-left
        [w-side_margin, 0],         # Top-right
        [w-side_margin, bev_height],# Bottom-right
        [side_margin, bev_height]   # Bottom-left
    ])
    
    # Convert source points to numpy array
    src_points = np.float32(roi_points)
    
    # Calculate homography matrix
    H, _ = cv2.findHomography(src_points, dst_points)
    return H, dst_points

def BEVTransform(img, H, output_size=None):
    """
    Applies the BEV transformation using the calculated homography matrix
    """
    if output_size is None:
        output_size = (img.shape[1], img.shape[0])  # Same size as input
        
    return cv2.warpPerspective(img, H, output_size, flags=cv2.INTER_LINEAR)



def main():
    """
    Main function to run Canny edge detection on an image and save the result.
    """
    # # Create output directories
    # output_dir = os.path.join("results", "LineDetection")
    # Edges_output_dir = os.path.join(output_dir, "Edges")
    # Lines_output_dir = os.path.join(output_dir, "Lines")
    # os.makedirs(Edges_output_dir, exist_ok=True)
    # os.makedirs(Lines_output_dir, exist_ok=True)
    
    # # Load image
    # for img_file in os.listdir(ROADS_FOLDER):
    #     img_path = os.path.join(ROADS_FOLDER, img_file)
    #     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #     if img is None:
    #         print("Error: Could not load image.")
    #         return
    
    #     # Apply Canny edge detection
    #     lines, edges = findRoadPlane(img)

    #     output_path = os.path.join(Edges_output_dir, img_file)

    #     # Save result
    #     cv2.imwrite(output_path, edges)
    #     print(f"Canny edge detection result saved to {output_path}")

    #     output_path = os.path.join(Lines_output_dir, img_file)

    #     # Save result
    #     cv2.imwrite(output_path, lines)
    #     print(f"Canny edge detection result saved to {output_path}")
    video = r"C:\DataSets\ShaiwalVids\20250120_225133000_iOS.MP4"
    yolo = r"C:\Users\bahaa\YOLO\yolo11x.pt"

    cap2 = cv2.VideoCapture(video)
    _, img = cap2.read()
    
    # Step 1: Let user define ROI
    roi_points, roi_mask = findROIUser(img)
    
    if len(roi_points) != 4:
        print("Error: Exactly 4 points required for homography")
        return
    
    # Step 2: Calculate homography matrix
    H, dst_points = CalculateHomography(img, roi_points)
    
    # Step 3: Apply BEV transformation
    bev_img = BEVTransform(img, H)
    
    # Visualization
    cv2.imshow("Original ROI", cv2.bitwise_and(img, img, mask=roi_mask))
    cv2.imshow("Bird's Eye View", bev_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save results
    cv2.imwrite("bev_transform.jpg", bev_img)


    """this code is for findROIBoundingBoxes"""
    # road = findROIBoundingBoxes(video, yolo)
    # cv2.imshow("Road Plane", road)
    # cv2.waitKey(0)  # Wait for key press
    # cv2.destroyAllWindows()

    # output_dir = "results"
    # os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    # output_path = os.path.join(output_dir, "road_plane.png")  # Add filename
    # cv2.imwrite(output_path, road)
    # print(f"Image saved at: {output_path}")


if __name__ == "__main__":
    main()

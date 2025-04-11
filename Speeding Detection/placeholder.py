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

# Function to draw detected lines
def draw_lines(color_img, empty_img,line, color):
    slope, intercept = line[0], line[1]
    x1, y1 = 0, int(intercept)
    x2, y2 = color_img.shape[1], int(slope * color_img.shape[1] + intercept)
    cv2.line(color_img, (x1, y1), (x2, y2), color, 3)
    cv2.line(empty_img, (x1, y1), (x2, y2), color, 3)

def find_outermost_lines(lines):
    """Takes in a bunch of lines in the format (m,b) and finds the 
    two outermost lines in the image"""

    min_line = None
    max_line = None
    min_x = float('inf')
    max_x = float('-inf')

    for line in lines:
        m, b = line
        x = -b / m # Calculate x-intercept: x = -b / m

        # Update leftmost line
        if x < min_x:
            min_x = x
            min_line = line

        # Update rightmost line
        if x > max_x:
            max_x = x
            max_line = line

    return min_line, max_line


def findCarEdges(img, kernelSize=3, Sigma=5.0, High=0.6, Low=0.4, numLines=6):
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
    peak_threshold = 0.5 * np.max(acc)  # Relative threshold for strong peaks (filters out irrelevent lines)

    # for _ in range(num_lines):
    while (len(detected_lines) < num_lines): 
        idx = np.argmax(acc)  # Find peak index (idx is flattened - needs to be converted back to (row,colomn))
        r_idx, t_idx = divmod(idx, acc.shape[1])  # Convert to 2D indices
        if acc[r_idx, t_idx] < peak_threshold: 
            break  # Stop if remaining peaks are too weak

        rho = rhos[r_idx]
        theta = thetas[t_idx]
        
        acc[r_idx, t_idx] = 0  # Suppress this peak to avoid duplicate detection
        acc[max(r_idx-5, 0):r_idx+5, max(t_idx-5, 0):t_idx+5] = 0  # Suppress nearby peaks to avoid crowding one region


        # Convert Hough parameters to Cartesian line representation
        if np.sin(theta) != 0:  # Avoid division by zero
            slope = - (np.cos(theta) / np.sin(theta))  # Line slope
            intercept = (rho / np.sin(theta))  # y-intercept
            detected_lines.append((slope, intercept))
    
    # Convert grayscale image back to color
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Function to draw detected lines
    def draw_lines(line, color):
        slope, intercept = line[0], line[1]
        x1, y1 = 0, int(intercept)
        x2, y2 = img.shape[1], int(slope * img.shape[1] + intercept)
        cv2.line(color_img, (x1, y1), (x2, y2), color, 3)

    # Draw detected lines in 6 different colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]  # 6 distinct colors

    for i, line in enumerate(detected_lines):
        draw_lines(line, colors[i % len(colors)])  

    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    edges = (edges * 255).astype("uint8")

    return detected_lines, color_img, edges  # Return color_img and edges for debugging

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

def findRoadAlignedEdge(img, kernelSize=3, Sigma=5.0, High=0.6, Low=0.4, numLines=2):
    """
    Detects road edges using Canny edge detection and Hough Transform.
    Displays exactly 6 detected straight lines on the image.
    """

    empty_img = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)) # empty image to draw lines on

    # Apply Canny edge detection
    edges = canny(img, kernel_size=kernelSize, sigma=Sigma, high=High, low=Low)

    # Perform Hough Transform
    acc, rhos, thetas = hough_transform(edges)

    # Store detected lines
    detected_lines = []

    # Extract 6 strongest peaks from the Hough accumulator
    num_lines = numLines
    peak_threshold = 0.5 * np.max(acc)  # Relative threshold for strong peaks (filters out irrelevent lines)

    # for _ in range(num_lines):
    while (len(detected_lines) < num_lines): 
        idx = np.argmax(acc)  # Find peak index (idx is flattened - needs to be converted back to (row,colomn))
        r_idx, t_idx = divmod(idx, acc.shape[1])  # Convert to 2D indices
        if acc[r_idx, t_idx] < peak_threshold: 
            break  # Stop if remaining peaks are too weak

        rho = rhos[r_idx]
        theta = thetas[t_idx]
        
        acc[r_idx, t_idx] = 0  # Suppress this peak to avoid duplicate detection
        acc[max(r_idx-100, 0):r_idx+100, max(t_idx-5, 0):t_idx+5] = 0  # Suppress nearby peaks to avoid crowding one region


        # Convert Hough parameters to Cartesian line representation
        if np.sin(theta) != 0:  # Avoid division by zero
            slope = - (np.cos(theta) / np.sin(theta))  # Line slope
            intercept = (rho / np.sin(theta))  # y-intercept
            detected_lines.append((slope, intercept))
    
    # Convert grayscale image back to color
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw detected lines in 6 different colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]  # 6 distinct colors

    for i, line in enumerate(detected_lines):
        draw_lines(color_img, empty_img, line, colors[i % len(colors)])  

    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    edges = (edges * 255).astype("uint8")

    return detected_lines, color_img, edges  # Return color_img and edges for debugging


def main():
    """
    Main function to run Canny edge detection on an image and save the result.
    """
    img_path = r"C:\DataSets\RoadsCropped\SeattleStreet.png"

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    #lines, linesCar, edges = findCarEdges(img, kernelSize=3, Sigma=1.0, High=2, Low=1.5, numLines=1)
    lines, linedImage, edges = findRoadAlignedEdge(img, kernelSize=3, Sigma=1.0, High=2, Low=1.5, numLines=5)
    print(lines)
    lines = find_outermost_lines(lines)
    black_img = np.zeros_like(img)
    for line in lines:
        draw_lines(img, black_img, line, (255,0,0))
    
    y1, y2 = 0 , img.shape[0]
    x1,x2 =(y1 - lines[0][1]) / lines[0][0], (y2 - lines[0][1]) / lines[0][0] 
    x3, x4 = (y1 - lines[1][1]) / lines[1][0], (y2 - lines[1][1]) / lines[1][0]
    corners = [(y1,x2),(y1,x4), (y2,x3),(y2,x1)]
    H,_ = CalculateHomography(img, corners)
    transformed = BEVTransform(img, H)

    cv2.imshow("BEV", transformed)    
    cv2.imshow('outermost edge lines', black_img)
    cv2.imshow('edges', edges)
    cv2.imshow('Lines on Road', linedImage)
    cv2.imwrite('output.PNG', linedImage)
    cv2.imwrite('output2.PNG', black_img)
    # cv2.imshow('Lines', linesCar)
    # cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

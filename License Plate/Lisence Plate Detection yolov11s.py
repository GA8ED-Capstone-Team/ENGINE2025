import cv2
import torch
import numpy as np
from ultralytics import YOLO  # 替换为ultralytics库
from PIL import Image
import matplotlib.pyplot as plt
import os
from glob import glob
from collections import deque
import pandas as pd
from scipy.signal import savgol_filter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class LicensePlateDetector:
    def __init__(self, model_path="runs/train/license_plate_detection/weights/best.pt"):
        """Initialize the license plate detector"""
        # Check if CUDA is available
        self.device = 0 if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("Warning: CUDA is not available, using CPU instead")
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        # Load the YOLO model
        print(f"Loading license plate detection model from {model_path}...")
        self.detector = YOLO(model_path)

        # Standard license plate dimensions (in millimeters)
        self.PLATE_STANDARDS = {
            'US': {'width': 305, 'height': 152, 'ratio': 305 / 152},
            'EU': {'width': 520, 'height': 110, 'ratio': 520 / 110},
            'DEFAULT': {'width': 440, 'height': 120, 'ratio': 440 / 120}
        }

    def detect_plate_type(self, width, height):
        """
        Determine the license plate type based on the aspect ratio, without relying on historical records
        Args:
            width: Width of the license plate
            height: Height of the license plate
        Returns:
            plate_type: Type of the license plate
        """
        aspect_ratio = width / height

        # Calculate the difference in ratio compared to standard types
        differences = {
            plate_type: abs(specs['ratio'] - aspect_ratio)
            for plate_type, specs in self.PLATE_STANDARDS.items()
            if plate_type != 'DEFAULT'  # Exclude the default type
        }

        # If the difference is too large compared to any standard type, return DEFAULT
        min_diff = min(differences.values())
        if min_diff > 0.5:  # Set a threshold
            return 'DEFAULT'

        # Return the most matching type
        return min(differences.items(), key=lambda x: x[1])[0]

    def calculate_distance_and_scale(self, image):
        """Calculate the distance and scale factor"""
        bbox, conf, _ = self.detect_license_plate(image)
        if bbox is None:
            return None, None, None, None, image

        # Calculate the width and height of the license plate
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # Determine the license plate type
        plate_type = self.detect_plate_type(width, height)

        # Get the standard width and calculate the scale factor
        standard_width = self.PLATE_STANDARDS[plate_type]['width']
        scale_factor = standard_width / width

        # Calculate the distance
        image_center_x = image.shape[1] / 2
        plate_center_x = (bbox[0] + bbox[2]) / 2
        pixel_distance = abs(image_center_x - plate_center_x)
        distance = pixel_distance * scale_factor / 1000  # Convert to meters

        # Add annotations
        image_with_annotation = image.copy()
        cv2.rectangle(image_with_annotation,
                      (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])),
                      (0, 255, 0), 2)

        text = f'Dist: {distance:.2f}m, Scale: {scale_factor:.2f}mm/px ({plate_type})'
        cv2.putText(image_with_annotation,
                    text,
                    (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

        return distance, scale_factor, bbox, plate_type, image_with_annotation

    def detect_license_plate(self, image):
        """Detect the license plate in the image using YOLO"""
        if not isinstance(image, np.ndarray):
            # Convert PIL Image to numpy array if needed
            image = np.array(image)

        # Run detection with the YOLO model
        results = self.detector(image, conf=0.25)

        # No detections found
        if len(results[0].boxes) == 0:
            return None, 0, None

        # Get the box with highest confidence
        boxes = results[0].boxes
        confidence_scores = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidence_scores)

        # Extract the bounding box coordinates
        xyxy = boxes.xyxy.cpu().numpy()
        best_box = xyxy[best_idx]
        best_conf = confidence_scores[best_idx]

        bbox = [
            best_box[0],  # xmin
            best_box[1],  # ymin
            best_box[2],  # xmax
            best_box[3]  # ymax
        ]

        return bbox, best_conf, None

    def process_and_display(self, image_folder, display_frames=20):
        """
        Process all frames and display the detection results for the specified number of frames
        Args:
            image_folder: Path to the image folder
            display_frames: Number of frames to display
        """
        # Get all images and sort them
        image_files = sorted(glob(os.path.join(image_folder, '*.jpg')))

        # Store results
        results = []
        distances = []
        scales = []
        frame_numbers = []

        # Process all images
        for i, img_path in enumerate(image_files):
            print(f"\nProcessing: {os.path.basename(img_path)}")
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Process a single frame
            distance, scale_factor, bbox, plate_type, annotated_image = self.calculate_distance_and_scale(image)

            if distance is not None:
                results.append({
                    'frame': i,
                    'distance': distance,
                    'scale_factor': scale_factor,
                    'plate_type': plate_type,
                    'annotated_image': annotated_image if i < display_frames else None
                })
                distances.append(distance)
                scales.append(scale_factor)
                frame_numbers.append(i)

        # Smooth the data
        if len(distances) > 5:
            smoothed_distances = savgol_filter(distances, min(5, len(distances)), 2)
            smoothed_scales = savgol_filter(scales, min(5, len(scales)), 2)
        else:
            smoothed_distances = distances
            smoothed_scales = scales

        # 1. Plot the analysis for all frames
        plt.figure(figsize=(15, 10))

        # Distance change plot
        plt.subplot(2, 1, 1)
        plt.plot(frame_numbers, distances, 'b.', label='Raw Distance')
        plt.plot(frame_numbers, smoothed_distances, 'r-', label='Smoothed Distance')
        plt.xlabel('Frame Number')
        plt.ylabel('Distance (m)')
        plt.title('Distance Change Over All Frames')
        plt.legend()
        plt.grid(True)

        # Scale factor change plot
        plt.subplot(2, 1, 2)
        plt.plot(frame_numbers, scales, 'b.', label='Raw Scale Factor')
        plt.plot(frame_numbers, smoothed_scales, 'r-', label='Smoothed Scale Factor')
        plt.xlabel('Frame Number')
        plt.ylabel('Scale Factor (mm/pixel)')
        plt.title('Scale Factor Change Over All Frames')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # 2. Display the detection results for the first 40 frames
        early_results = [r for r in results if r['frame'] < display_frames]
        rows = 8
        cols = 5

        if early_results:
            # Increase the figure size to display larger images
            plt.figure(figsize=(25, 32))  # Adjust the overall size

            for i, result in enumerate(early_results):
                if i >= rows * cols:  # Limit to displaying 40 images
                    break

                plt.subplot(rows, cols, i + 1)
                plt.imshow(cv2.cvtColor(result['annotated_image'], cv2.COLOR_BGR2RGB))

                # Adjust the title text size and content format
                plt.title(
                    f"Frame {result['frame']}\nDistance: {result['distance']:.2f}m\nScale: {result['scale_factor']:.2f}",
                    fontsize=10, pad=10)
                plt.axis('off')

            # Adjust the spacing between subplots
            plt.subplots_adjust(wspace=0.3, hspace=0.4)
            plt.show()

        # Print the analysis results
        print("\nAnalysis Results:")
        print(f"Total processed frames: {len(results)}")
        print(f"\nDetailed data for the first {display_frames} frames:")
        for result in early_results:
            print(f"\nFrame {result['frame']}:")
            print(f"- Distance: {result['distance']:.2f}m")
            print(f"- Scale factor: {result['scale_factor']:.2f}mm/pixel")
            print(f"- License plate type: {result['plate_type']}")

        return results, smoothed_distances, smoothed_scales


if __name__ == "__main__":
    detector = LicensePlateDetector()  # Using best.pt model

    test_folder = "test_images"
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
        print(f"Created '{test_folder}' directory. Please add test images.")
    else:
        # Process all frames in test_images folder, but only display the first 40 frames
        results, smoothed_distances, smoothed_scales = detector.process_and_display(test_folder, display_frames=40)
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from ultralytics import YOLO
import warnings
from sklearn.metrics import roc_auc_score, roc_curve
warnings.filterwarnings("ignore", category=FutureWarning)



# Define paths to dataset
DATASET_PATH = "C:\DataSets"
CARS_FOLDER = os.path.join(DATASET_PATH, "Selection_Cars")
BACKGROUND_FOLDER = os.path.join(DATASET_PATH, r"Backgrounds\negative\img")

# Load YOLO models (Ensure you have the correct weight files)
models = {
    "YOLOv11": YOLO(r"C:\Users\bahaa\YOLO\yolo11x.pt"),
    "YOLOv10": YOLO(r"C:\Users\bahaa\yolov10\weights\yolov10x.pt")
}

# Accuracy tracking
results_summary = {}


def evaluate_model(model, model_name):
    """Run inference and calculate accuracy metrics."""
    TP, FN, FP = 0, 0, 0 # Trure Positive, False Negative, False Positive
    TP_confidence_arr = [] # Confidence for true classifications (car when car)
    FP_confidence_arr = [] # Confidence for false classifications (car when no car)

    ground_truth = [] # Ground truth labels (1 = Car , 0 = No Car)
    classification_confidence = [] # Model's Confidence Scores (Higher, More Confident)

    # Create output directories
    output_dir = os.path.join("results", model_name)
    TP_output_dir = os.path.join(output_dir, "TP")
    FN_output_dir = os.path.join(output_dir, "FN")
    FP_output_dir = os.path.join(output_dir, "FP")
    os.makedirs(TP_output_dir, exist_ok=True)
    os.makedirs(FN_output_dir, exist_ok=True)
    os.makedirs(FP_output_dir, exist_ok=True)

    # Test on "cars" folder (Should detect cars) - Most Pics have one cars, few have 2
    for img_file in os.listdir(CARS_FOLDER):
        img_path = os.path.join(CARS_FOLDER, img_file)
        img = cv2.imread(img_path)

        results = model.predict(img, device=0) # Run Inferenece

        detected = False

        # Check if at least one car is detected
        for result in results:

            labels = result.boxes.cls.int().tolist()
            confidences = result.boxes.conf.tolist()
            # Check if detected object is a car (assuming 'car' class index is 2) and save the confidence of classification
            for i, lbl in enumerate(labels): 
                if lbl == 2:
                    detected = True
                    TP_confidence_arr.append(confidences[i])
                    classification_confidence.append(confidences[i]) # Append the confidence score corrosponding to the detection
                    ground_truth.append(1) # All images in this file include cars 

        if detected:
            TP += 1  # True Positive

            # Save annotated image
            annotated_img = result.plot()
            output_path = os.path.join(TP_output_dir, img_file)
            cv2.imwrite(output_path, annotated_img)
        else:
            classification_confidence.append(0) # Confidence score of 0 in case of no detection
            ground_truth.append(1) # All images in this file include cars 
            FN += 1  # False Negative (missed detection)
            # Save annotated image
            annotated_img = result.plot()
            output_path = os.path.join(FN_output_dir, img_file)
            cv2.imwrite(output_path, annotated_img)

    # Test on "background" folder (Should NOT detect cars)
    for img_file in os.listdir(BACKGROUND_FOLDER):
        img_path = os.path.join(BACKGROUND_FOLDER, img_file)
        img = cv2.imread(img_path)
        results = model(img)

        # Check if a car is wrongly detected
        # If results is a list, extract the first element (the detection for the first image)
        # Check if at least one car is detected

        detected = False
        for result in results:
            labels = result.boxes.cls.int().tolist()
            confidences = result.boxes.conf.tolist()
            for i, lbl in enumerate(labels): 
                if lbl == 2:
                    detected = True
                    FP_confidence_arr.append(confidences[i])
                    classification_confidence.append(confidences[i]) # Append the confidence score corrosponding to the detection
                    ground_truth.append(0) # All images in this file DO NOT include cars

        if detected:
            FP += 1  # False Positive (wrong detection)
            # Save annotated image
            annotated_img = result.plot()
            output_path = os.path.join(FP_output_dir, img_file)
            cv2.imwrite(output_path, annotated_img)
        else:
            classification_confidence.append(0) # Confidence score of 0 in case of no detection
            ground_truth.append(0) # All images in this file DO NOT include cars

    # Compute Precision, Recall, F1-score, AuC score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    auc_score = roc_auc_score(ground_truth, classification_confidence)


    # Store results
    results_summary[model_name] = {
        "True Positives": TP,
        "False Negatives": FN,
        "False Positives": FP,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1_score, 4),
        "AuC Score" : auc_score
    }

    print(f"\n{model_name} Results:")
    print(f"TP: {TP}, FN: {FN}, FP: {FP}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}\n")

    plt.figure()
    plt.hist(TP_confidence_arr, bins=20, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel("Confidence Score of Correctly Identified Car Classifications")
    plt.ylabel("Frequency")
    plt.ylim(0,400)
    plt.title(f"{model_name} True Positives Confidence Score Distribution")
    plt.savefig(f"{model_name}_TP_confidence_histogram.png")
    plt.close()

    plt.figure()
    plt.hist(FP_confidence_arr, bins=20, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel("Confidence Score of Incorrectly Identified Car Classifications")
    plt.ylabel("Frequency")
    plt.ylim(0,30)
    plt.title(f"{model_name} False Positives Confidence Score Distribution")
    plt.savefig(f"{model_name}_FP_confidence_histogram.png")
    plt.close()

# Run evaluation for all models
for model_name, model in models.items():
    evaluate_model(model, model_name)

# Print final comparison
metrics_file = "metrics.txt"
with open(metrics_file, "w") as f:
    for model, scores in results_summary.items():
        f.write(f"{model}:\n")
        for metric, value in scores.items():
            f.write(f" {metric}: {value}")
        f.write("\n")

print(f"Metrics saved to {metrics_file}")

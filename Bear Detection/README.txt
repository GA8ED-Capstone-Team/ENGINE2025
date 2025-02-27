Wildlife Detection and Alert System
===================================

This Python file implements a real-time wildlife detection and tracking system using YOLO v11x for object detection and DeepSORT for multi-object tracking. The system is optimized to detect and track specific wild animals (e.g., bears, wolves, deer) and trigger alerts based on a stability score-based filtering mechanism.

Key Features
------------

- **Object Detection with YOLO v11x**: Detects and classifies multiple objects per frame with bounding boxes and confidence scores.
- **Multi-Object Tracking with DeepSORT**: Assigns unique IDs to detected animals, ensuring identity persistence across frames.
- **Adaptive Confidence Thresholding**: Dynamically filters weak detections (default threshold: 0.7 for improved low-light sensitivity).
- **Stability Score-Based Filtering**:
  - Measures detection consistency over frames using an exponential moving average of confidence scores.
  - Balances sensitivity and robustness against flickering detections.
- **Rolling Window Optimization (100 Frames)**:
  - Computes stability scores over the most recent 100 frames instead of all frames, making detection more responsive to real-time changes.
- **Enhanced Alerting Mechanism**:
  - Triggers an alert if:
    - Stability Score > 0.2, OR
    - Object appears in ≥3 frames with an average confidence > 0.5
- **Bounding Box Enhancements**:
  - Displays class labels and confidence scores directly on the detected objects.
  - Adjusted color/thickness for clearer visualization.

How It Works
------------

1. **Video Processing**: Reads input video frame-by-frame while dynamically skipping frames for efficiency.
2. **Object Detection**: YOLO v11x processes each frame, extracting bounding boxes, class IDs, and confidence scores.
3. **Tracking with DeepSORT**: Maintains object identities across frames to reduce flickering and misdetections.
4. **Stability Score Computation**:
   - Uses a rolling window to compute:

     Stability Score = (Mean Confidence * Persistence Count) / (Tracked Frames, max 100)

   - Ensures that short-lived detections don’t dominate long-term stability.
5. **Alert Triggering**: Checks if an object meets either stability threshold or persistence-based confidence threshold.

Known Limitations & Future Enhancements
---------------------------------------

- **YOLO Misclassifications**: Wolves occasionally get misclassified as sheep, requiring additional fine-tuning on wildlife datasets.
- **Low-Light Detection Challenges**:
  - Some bears are missed in nighttime or shadowed conditions.
  - **Solution**: Fine-tune YOLO on infrared/night-vision datasets.
- **Fast-Moving Object Handling**:
  - Objects appearing only for 1–2 frames may not accumulate enough stability to trigger an alert.
  - **Current Fix**: Reduced frame persistence threshold to 3 frames.


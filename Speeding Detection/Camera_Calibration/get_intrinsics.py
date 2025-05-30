import cv2
import numpy as np
import os

# Settings
chessboard_size = (7, 5)  # Number of inner corners per chessboard row and column
square_size = 15          # Set this to your chessboard square size
video_path = r"C:\Users\bahaa\CapstoneProject\ENGINE2025\Speeding Detection\Camera_Calibration\IMG_1774.MOV"

# Prepare object points
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if frame_count < 30:
    print("Video too short for 30 samples.")
    cap.release()
    exit()

# Randomly sample 30 unique frames from the midsection
import random
start = frame_count // 3
end = 2 * frame_count // 3
indices = sorted(random.sample(range(start, end), 30))

for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, img = cap.read()
    if not ret:
        print(f"Failed to read frame {idx}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret_cb:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret_cb)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)
    else:
        print(f"Chessboard not found in frame {idx}")

cap.release()
cv2.destroyAllWindows()

if len(objpoints) < 1:
    print("Not enough valid frames for calibration.")
    exit()

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix (intrinsics):")
print(mtx)
print("\nDistortion coefficients:")
print(dist.ravel())

# Optionally, save to file
np.savez('camera_intrinsics.npz', mtx=mtx, dist=dist)
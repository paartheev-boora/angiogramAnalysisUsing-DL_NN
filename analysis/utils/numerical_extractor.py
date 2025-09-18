# numerical_extractor.py

import cv2
import os
import numpy as np
# from skimage import measure

def compute_optical_flow(prev_frame, next_frame):
    flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude

def compute_contrast(frame):
    return frame.std()

def compute_vessel_area(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = sum(cv2.contourArea(c) for c in contours)
    return area

def extract_numerical_features(folder):
    frames = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    if len(frames) < 2:
        return {}

    flow_vals, contrast_vals, area_vals = [], [], []

    for i in range(len(frames) - 1):
        f1 = cv2.imread(os.path.join(folder, frames[i]), cv2.IMREAD_GRAYSCALE)
        f2 = cv2.imread(os.path.join(folder, frames[i + 1]), cv2.IMREAD_GRAYSCALE)

        flow_mag = compute_optical_flow(f1, f2)
        flow_vals.append(np.mean(flow_mag))

        contrast_vals.append(compute_contrast(f2))
        area_vals.append(compute_vessel_area(f2))

    return {
        "avg_flow": float(np.mean(flow_vals)),
        "avg_contrast": float(np.mean(contrast_vals)),
        "avg_area": float(np.mean(area_vals))
    }

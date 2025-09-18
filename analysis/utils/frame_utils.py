# frame_utils.py

import cv2
import os

def extract_frames_from_video(video_path, output_folder, frame_size=(224, 224)):
    """
    Extracts frames from a video and saves them to a folder.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Folder where frames will be saved.
        frame_size (tuple): Desired size (width, height) for resized frames.

    Returns:
        int: Total number of frames extracted.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, frame_size)
        filename = os.path.join(output_folder, f"frame_{count:04d}.png")
        cv2.imwrite(filename, resized)
        count += 1

    cap.release()
    return count

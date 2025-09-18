# lesion_detector.py

import cv2
import os
from roboflow import Roboflow

# Initialize Roboflow model (replace with your own API key and project/version)
rf = Roboflow(api_key="aZ1z1VnnSG13LcdBx0WG")
print(" Your Workspaces:")
print(rf.workspace())
project = rf.workspace("stenosisdetection").project("coronary_lesion_detection-tstzc")  # replace with your Roboflow project name
model = project.version("5").model
#model = rf.model("stenosisdetection/coronary_lesion_detection-tstzc-instant-1")  # replace 1 with your version number

def detect_lesions_on_frame(image_path, output_path=None, conf_threshold=0.1):
    """
    Detect lesions in a single image using Roboflow Instant Model and annotate the frame.
    """
    prediction = model.predict(image_path, confidence=conf_threshold).json()
    image = cv2.imread(image_path)

    if not prediction['predictions']:
        print(f"[INFO] No lesions detected in {image_path}")
    else:
        for obj in prediction['predictions']:
            x, y, w, h = int(obj['x']), int(obj['y']), int(obj['width']), int(obj['height'])
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2
            label = f"{obj['class']} {obj['confidence']:.2f}"

            # Draw bounding box (yellow, thicker line)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Background rectangle for text
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - baseline),
                          (x1 + tw, y1), (0, 255, 255), -1)

            # Put label text (black for contrast)
            cv2.putText(image, label, (x1, y1 - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Save output if requested (always writes image, even if no boxes)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)

    return prediction['predictions']



def detect_lesions_in_folder(input_folder, output_folder):
    """
    Annotate all images in a folder using the trained Roboflow model.

    Args:
        input_folder (str): Path to the folder containing image frames.
        output_folder (str): Path to save the annotated images.
    """
    for fname in sorted(os.listdir(input_folder)):
        if fname.endswith(".png"):
            in_path = os.path.join(input_folder, fname)
            out_path = os.path.join(output_folder, fname)
            detect_lesions_on_frame(in_path, out_path)

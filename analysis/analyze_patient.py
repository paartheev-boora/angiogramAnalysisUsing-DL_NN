import os
import sys
import shutil
import json
from PIL import Image   # for Image.open()
from torchvision import transforms

# Torch
import torch
import torch.nn as nn
import streamlit as st  

# -------------------------
# Now safe to import from models, utils, data
# -------------------------
from data.dataset_loader import AngioClipDataset
from models.cnn_lstm_nested import NestedCNNLSTM
from utils.frame_utils import extract_frames_from_video
from utils.lesion_detector import detect_lesions_in_folder
from utils.numerical_extractor import extract_numerical_features
from utils.fusion_module import fuse_decisions


# ---- Initialize session_state ----
if "results" not in st.session_state:
    st.session_state["results"] = {}

# CONFIG

VIDEO_DIR = "new_patient"
OUTPUT_DIR = "output_dataset"
FRAME_SIZE = (224, 224)
FRAMES_PER_CLIP = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
print(f" Using device: {DEVICE}")

# Load model

print(" Loading CNN+LSTM model...")
model = NestedCNNLSTM().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Helper to form batches

def create_clips(frames, clip_length):
    return [frames[i:i + clip_length] for i in range(0, len(frames) - clip_length + 1, clip_length)]

# Inference loop
def run_analysis():
    # Create output folders

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "visual"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "numerical"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "fusion"), exist_ok=True)
    
    for video_file in os.listdir(VIDEO_DIR):
        if not video_file.endswith(".mp4"):
            continue

        name = os.path.splitext(video_file)[0]
        st.write(f"🔄 Processing: {name}")

        # Step 1: Extract frames
        video_path = os.path.join(VIDEO_DIR, video_file)
        frame_dir = os.path.join("temp_frames", name)
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir, exist_ok=True)
        extract_frames_from_video(video_path, frame_dir, FRAME_SIZE)

        # Step 2: Lesion Detection
        print(" Detecting lesions using Roboflow...")
        detect_lesions_in_folder(frame_dir, os.path.join(OUTPUT_DIR, "visual", name))

        # Step 3: Video Prediction using CNN+LSTM
        print(" Predicting PCI requirement...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load and transform frames
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
        clips = create_clips(frame_files, FRAMES_PER_CLIP)
        predictions = []

        for clip in clips:
            imgs = []
            for frame_name in clip:
                img = Image.open(os.path.join(frame_dir, frame_name)).convert('RGB')
                imgs.append(transform(img))
            video_tensor = torch.stack(imgs).unsqueeze(0).to(DEVICE)  # shape: (1, T, C, H, W)
            with torch.no_grad():
                output = model(video_tensor.to(DEVICE))
                prob = torch.softmax(output, dim=1)[0][1].item()
                predictions.append(prob)

        avg_prob = sum(predictions) / len(predictions) if predictions else 0

        # Step 4: Numerical Feature Extraction
        print(" Extracting numerical features...")
        numerical_features = extract_numerical_features(frame_dir)
        with open(os.path.join(OUTPUT_DIR, "numerical", f"{name}.json"), "w") as f:
            json.dump(numerical_features, f, indent=2)

        # Step 5: Fusion
        print(" Fusing predictions...")
        final_decision = fuse_decisions(avg_prob, numerical_features)
        with open(os.path.join(OUTPUT_DIR, "fusion", f"{name}_decision.txt"), "w") as f:
            f.write(f"Final Decision: {final_decision}\n")
            f.write(f"Visual PCI Prob: {avg_prob:.2f}\n")
            f.write(json.dumps(numerical_features, indent=2))
        print(f" {name}: {final_decision}")
        
        # ---- Save results in session_state ----
        st.session_state["results"][name] = {
            "lesions_text": f"Lesion detection completed for {name}",
            "lesion_images": [
                os.path.join(OUTPUT_DIR, "visual", name, f)
                for f in os.listdir(os.path.join(OUTPUT_DIR, "visual", name))
            ],
            "numerical_data": numerical_features,
            "pci_prediction": final_decision,
            "confidence": round(avg_prob * 100, 2),
        }
    print("\n All patient videos processed.")
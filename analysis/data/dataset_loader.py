import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class AngioClipDataset(Dataset):
    def __init__(self, root_dir, label_map={"PCI": 1, "No_PCI": 0}, clip_length=10, transform=None):
        self.samples = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.clip_length = clip_length

        for label_name in label_map.keys():
            label_dir = os.path.join(root_dir, label_name)
            for patient_id in os.listdir(label_dir):
                patient_dir = os.path.join(label_dir, patient_id)
                for clip_name in os.listdir(patient_dir):
                    clip_path = os.path.join(patient_dir, clip_name)
                    frame_paths = sorted([
                        os.path.join(clip_path, f) for f in os.listdir(clip_path) if f.endswith('.png')
                    ])
                    if len(frame_paths) == clip_length:
                        self.samples.append((frame_paths, label_map[label_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]

        frames = []
        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            frames.append(img)

        clip_tensor = torch.stack(frames)  # shape: (T, C, H, W)
        return clip_tensor, label

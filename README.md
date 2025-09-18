# 🫀 AI-Powered Coronary Angiogram Analysis

## 📌 Overview  
This project develops an **AI-driven system** for analyzing **coronary angiograms** to assist in **Percutaneous Coronary Intervention (PCI) decision-making**.  
It combines **deep learning** and **numerical analysis** to provide **transparent, patient-friendly insights**, reducing the chances of **miscommunication and medical fraud**.  

The system:  
- Uses a **Nested CNN+LSTM** model (ResNet-18 backbone) to classify **PCI requirement** from angiogram video clips.  
- Extracts **numerical features** (vessel diameter, flow estimates, lesion severity, TIMI frame count).  
- (Optional) Detects **lesions with bounding boxes** using a YOLOv5 model trained on **Roboflow**.  
- Provides an **interactive Streamlit frontend** with separate pages for Input, Lesion Detection, Numerical Analysis, and PCI Prediction.  

---

## ✨ Features  
- 📂 Upload angiogram videos for automated processing.  
- 🧠 **CNN+LSTM model** trained on **CADICA dataset** for PCI classification.  
- 📊 **Numerical Analysis Module** (vessel diameter, flow, TIMI flow).  
- 🔎 (Optional) **Lesion Detection** with YOLOv5 from Roboflow dataset.  
- 📈 Accuracy of **91.3%** achieved on PCI classification.  
- 💻 Interactive **Streamlit web app** with user-friendly navigation.  
- 🔒 Designed to improve **trust, transparency, and patient empowerment**.  

---

## 📊 Datasets  
1. **CADICA Dataset** – Used for training CNN+LSTM model on 10-frame angiogram clips.
   https://www.kaggle.com/datasets/abszgbert/invasive-coronary-angiography-cadica-dataset
2. **Stenosis Detection Dataset (Mendeley)** – Uploaded to **Roboflow** for training YOLOv5 lesion detection model.
   https://data.mendeley.com/datasets/ydrm75xywg/1

---

## 🏗️ Architecture  
- **Model:** ResNet-18 backbone → Feature Encoding → LSTM → PCI Prediction  
- **Numerical Path:** Vessel measurements, TIMI flow, lesion severity  
- **Decision Module:** Fusion of CNN+LSTM prediction and numerical features  
- **Frontend:** Streamlit multi-page app  

---

## 📈 Performance Metrics  
- **Accuracy (PCI Classification):** 91.3%  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam (lr=1e-4)  
- **Evaluation Metrics:** Accuracy, Loss, TIMI analysis values  

---

## 🖥️ Software & Hardware Requirements  

### Software  
- Python 3.10+  
- PyTorch (GPU-enabled, CUDA 11.8)  
- Torchvision  
- OpenCV  
- Streamlit  
- Roboflow SDK (for lesion detection)  
- Matplotlib, Pandas, Numpy  

### Hardware  
- NVIDIA GPU (tested on GeForce GTX 1650, 4GB VRAM)  
- 16GB RAM recommended for preprocessing  
- Windows/Linux environment  

---

## 🚀 Installation  

```bash
# Clone the repo
git clone https://github.com/yourusername/angiogram-analysis.git
cd angiogram-analysis

# Create virtual environment
python -m venv ang_env
source ang_env/bin/activate  # Linux/Mac
ang_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage  

### Train Model  
```bash
python train_model.py
```

### Run Analysis (Backend Only)  
```bash
python pages/analyze_patient.py
```

### Run Streamlit Frontend  
```bash
streamlit run app.py
```

---

## 📂 Project Structure  

```
angiogram_project/
│── data/
│   └── dataset_loader.py
│── models/
│   └── cnn_lstm_nested.py
│── utils/
│   ├── numerical_extractor.py
│   └── lesion_detector.py
│── pages/
│   ├── Input_uploader.py
│   ├── analyze_patient.py
│   ├── Detected_lesions.py
│   ├── Numerical_analysis.py
│   └── PCI_prediction.py
│── app.py
│── train_model.py
│── requirements.txt
```
keep the datasets in the same hierarchy as angiogram_project
(i) Cadica Dataset
(ii) Stenosis detection dataset
(iii) extracted modified version of CADICA dataset using dataset loader. 
---

## 🎯 Use Case  
- Assists **cardiologists** in decision-making by providing transparent and visual predictions.  
- Offers **patients** with limited medical literacy **easy-to-understand results** to reduce risk of fraud.  
- Could be deployed in hospitals under **licensed cardiologist supervision**.  

---

## 🔮 Future Development  
- Incorporation of **Grad-CAM/SHAP** for explainability.  
- Larger datasets with multi-center validation.  
- Integration into **clinical decision support systems (CDSS)**.  
- Multi-class classification for **different coronary artery diseases**.  

---

## 📜 License  
Open-source for **research and educational use**. For clinical use, deployment under **medical expert supervision** is required.  

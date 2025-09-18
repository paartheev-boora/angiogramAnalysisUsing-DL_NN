import streamlit as st
import os
from PIL import Image
import math

def detected_lesions():
    st.title("Detected Lesions")

    # ---- INIT SESSION STATE ----
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []
    if "results" not in st.session_state:
        st.session_state["results"] = {}
    # Guard clause: stop execution if no files uploaded
    if not st.session_state["uploaded_files"]:
        st.warning("No files uploaded yet. Please go to the Input Uploader page and upload angiogram data.")
        st.stop()
    if not st.session_state["results"]:   # Check if results are empty
        st.warning("Please upload and run analysis first from Input page.")
    else:
        for video, report in st.session_state["results"].items():
            st.subheader(f"{video}")
            st.write(report["lesions_text"])

            images = report.get("lesion_images", [])
            if images:
                cols_per_row = 5   # 👈 Number of images per row (adjustable)
                rows = math.ceil(len(images) / cols_per_row)

                for row in range(rows):
                    cols = st.columns(cols_per_row)
                    for i, col in enumerate(cols):
                        idx = row * cols_per_row + i
                        if idx < len(images) and os.path.exists(images[idx]):
                            col.image(images[idx], caption=f"Lesion {idx+1}", width=150)

        st.markdown("---")
        st.subheader("Artery Reference Map")
        artery_img_path = "C:/Users/arunj/Documents/Project/angiogram_project/assets/artery_map.png"
        if os.path.exists(artery_img_path):
            artery_img = Image.open(artery_img_path)
            st.image(artery_img, caption="Major Coronary Arteries", width="stretch")
        else:
            st.error("Artery map image not found. Please place it in `assets/artery_map.png`.")

if __name__=="__main__":
    detected_lesions()
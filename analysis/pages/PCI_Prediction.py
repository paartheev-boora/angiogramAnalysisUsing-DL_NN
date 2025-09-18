import streamlit as st
import os
def pci_prediction():
    st.title(" PCI Prediction")
    # ---- INIT SESSION STATE ----
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []
    if "results" not in st.session_state:
        st.session_state["results"] = {}
    # Guard clause: stop execution if no files uploaded
    if not st.session_state["uploaded_files"]:
        st.warning("No files uploaded yet. Please go to the Input Uploader page and upload angiogram data.")
        st.stop()
    if not st.session_state["results"]:   # <-- check if empty
        st.warning(" Please upload and run analysis first from Input page.")
    else:
        for video, report in st.session_state["results"].items():
            st.subheader(f" {video}")
            st.write(f"**Prediction:** {report['pci_prediction']}")
            st.write(f"**Confidence:** {report['confidence']}%")
            artery_img_path = "C:/Users/arunj/Documents/Project/angiogram_project/assets/heart_arteries.png"
            if os.path.exists(artery_img_path):
                st.image(artery_img_path, caption="Major Coronary Arteries", width="stretch")

if __name__=="__main__":
    pci_prediction()
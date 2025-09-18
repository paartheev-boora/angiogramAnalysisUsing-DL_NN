import streamlit as st
import os
from analyze_patient import run_analysis  # <-- call function instead of subprocess
def input_uploader():
    st.title(" Upload Angiogram Videos")

    UPLOAD_DIR = "new_patient"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # ---- Init session_state ----
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "results" not in st.session_state:
        st.session_state.results = {}

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload angiogram videos", accept_multiple_files=True, type=["mp4", "avi"]
    )

    if uploaded_files:
        # 🔹 Reset results ONLY when new files are uploaded
        if uploaded_files != [f["name"] for f in st.session_state.uploaded_files]:
            st.session_state.uploaded_files = []
            st.session_state.results = {}

            for uploaded_file in uploaded_files:
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.uploaded_files.append({"name": uploaded_file.name, "path": file_path})
                st.success(f" Saved: {uploaded_file.name}")


    # Show already uploaded files
    if st.session_state.uploaded_files:
        st.subheader("Uploaded Files")
        for file_info in st.session_state.uploaded_files:
            st.write(f"- {file_info['name']}")

    # Prediction button
    if st.button(" Run Prediction"):
        if not st.session_state.uploaded_files:
            st.warning(" Please upload videos first!")
        else:
            run_analysis()

if __name__=="__main__":
    input_uploader()
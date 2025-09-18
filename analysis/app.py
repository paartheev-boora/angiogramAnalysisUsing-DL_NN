import streamlit as st

st.set_page_config(
    page_title=" Coronary Angiogram Analysis",
    layout="wide"
)

# ---- INIT SESSION STATE ----
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "results" not in st.session_state:
    st.session_state.results = {}

st.title(" Coronary Angiogram Analysis Dashboard")
st.write("Navigate using the sidebar to upload angiogram videos, view detected lesions, PCI predictions, and numerical analysis.")

st.sidebar.success("Select a page above ")

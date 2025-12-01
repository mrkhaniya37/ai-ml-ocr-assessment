import os
import sys

# Make src/ importable when running "python -m streamlit run app.py"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import streamlit as st
import numpy as np
import cv2

from ocr_engine import OCREngine
from text_extraction import extract_target_line
from preprocessing import preprocess_image

st.set_page_config(page_title="Shipping Label OCR", layout="wide")

st.title("📦 Shipping Label OCR – `_1_` Line Extractor")

st.markdown(
    """
Upload a shipping label / waybill image and this app will:

1. Auto-rotate the image if needed  
2. Run OCR using EasyOCR (open-source)  
3. Extract the **complete line** containing the pattern `_1_`  
4. Show the result along with confidence
"""
)

uploaded_file = st.file_uploader("Upload label image", type=["jpg", "jpeg", "png"])

# Initialize OCR engine once
if "ocr_engine" not in st.session_state:
    st.session_state.ocr_engine = OCREngine(gpu=False)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    # Decode original for display (without our preprocessing)
    npimg = np.frombuffer(file_bytes, np.uint8)
    bgr_raw = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rgb_raw = cv2.cvtColor(bgr_raw, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(rgb_raw, use_column_width=True)

    with col2:
        st.subheader("Preprocessed (Auto-rotated) Image")
        # preprocess_image returns BGR, auto-rotated and resized
        pre_bgr = preprocess_image(file_bytes)
        pre_rgb = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2RGB)
        st.image(pre_rgb, use_column_width=True)

    if st.button("Run OCR & Extract"):
        with st.spinner("Running OCR..."):
            ocr_results = st.session_state.ocr_engine.run_ocr(file_bytes)

        # Display all OCR lines with confidence
        st.subheader("Full OCR Output")
        if not ocr_results:
            st.warning("No text detected by OCR.")
        else:
            lines = [f"{text} (conf={conf:.2f})" for _, text, conf in ocr_results]
            st.text("\n".join(lines))

        # Extract target line
        st.subheader("Target `_1_` Line")
        target_text, conf = extract_target_line(ocr_results)

        if target_text is not None:
            st.success(f"Extracted: `{target_text}`")
            st.write(f"Confidence: **{conf:.2f}**")
        else:
            st.error("No line containing `_1_` was found in the OCR output.")

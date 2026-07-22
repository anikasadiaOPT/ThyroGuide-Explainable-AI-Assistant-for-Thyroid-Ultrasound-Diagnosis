import streamlit as st
import cv2
import torch
import numpy as np
import json
import os

from ultralytics import YOLO
from dotenv import load_dotenv

from src.classification.cnn_classifier import ThyroidCNN
from src.pipeline.inference_pipeline import ThyroidInferencePipeline
from src.grad_cam.gradcamGenerator import GradCAMGenerator
from src.gemma_integration.gemma_analyzer import GemmaAnalyzer
from src.report_generation.pdf_report_generator import ThyroGuidePDFReport


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ThyroGuide AI",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

# ============================================================
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "cv_json" not in st.session_state:
    st.session_state.cv_json = None

if "gemma_report" not in st.session_state:
    st.session_state.gemma_report = None

if "patient_info" not in st.session_state:
    st.session_state.patient_info = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    yolo_model = YOLO(
        "models/yolo/best.pt"
    )

    cnn_model = ThyroidCNN(
        model_path="models/cnn/thyroid_cnn.pt",
        device=device
    )

    gradcam = GradCAMGenerator(
        model=cnn_model,
        transform=cnn_model.transform,
        device=device
    )

    pipeline = ThyroidInferencePipeline(
        yolo_model=yolo_model,
        cnn_model=cnn_model,
        gradcam=gradcam,
        device=device
    )

    gemma_analyzer = GemmaAnalyzer(
        model_id="google/gemma-4-26B-A4B-it"
    )

    return device, pipeline, gemma_analyzer


device, pipeline, gemma_analyzer = load_models()


# ============================================================
# HEADER
# ============================================================

st.title("🩺 ThyroGuide AI")

st.markdown(
    """
    ### Explainable AI Assistant for Thyroid Ultrasound Analysis

    ThyroGuide combines **YOLO-based nodule detection**,
    **CNN classification**, **Grad-CAM explainability**,
    and **Gemma 4-powered report generation**.
    """
)

st.divider()


# ============================================================
# PATIENT INFORMATION + IMAGE UPLOAD
# ============================================================

st.subheader("🩺 Examination Information")

# ============================================================
# MAIN INPUT SECTION
# ============================================================

left_col, right_col = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT: IMAGE UPLOAD
# ============================================================

with left_col:

    st.markdown(
        "### 📷 Ultrasound Image"
    )

    uploaded_file = st.file_uploader(
        "Click the box below to upload an ultrasound image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )

    # Empty upload placeholder
    if uploaded_file is None:

        st.markdown(
            """
            <div style="
                height: 320px;
                border: 2px dashed #888;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                color: #888;
                font-size: 18px;
                margin-top: 10px;
            ">
                <div>
                    <div style="font-size: 50px;">📷</div>
                    <div>Upload Thyroid Ultrasound Image</div>
                    <div style="font-size: 13px;">
                        JPG, JPEG or PNG
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # DECODE IMAGE ONLY AFTER UPLOAD
    # ========================================================

    image = None

    if uploaded_file is not None:

        file_bytes = np.frombuffer(
            uploaded_file.getvalue(),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error(
                "Unable to decode the uploaded image."
            )

        else:

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Uploaded Ultrasound Image",
                width="stretch"
            )


# ============================================================
# RIGHT: PATIENT INFORMATION
# ============================================================

with right_col:

    st.markdown(
        "### 👤 Patient Information"
    )

    patient_name = st.text_input(
        "Patient Name",
        placeholder="Enter patient name"
    )

    col1, col2 = st.columns(2)

    with col1:

        patient_age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=0
        )

    with col2:

        patient_gender = st.selectbox(
            "Gender",
            [
                "Not Provided",
                "Male",
                "Female",
                "Other"
            ]
        )

    clinical_information = st.text_area(
        "Clinical Information",
        placeholder=(
            "Enter relevant clinical information "
            "if available..."
        ),
        height=180
    )

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if uploaded_file is not None and image is not None:
        #
        analyze_button = st.button(
            "🔍 Analyze Image",
            type="primary",
            width="stretch"
        )
        if analyze_button:
            # Clear previous results
            st.session_state.analysis_result = None
            st.session_state.cv_json = None
            st.session_state.gemma_report = None
            st.session_state.patient_info = None

            # Save uploaded filename
            st.session_state.uploaded_filename = uploaded_file.name


            # 1. COMPUTER VISION ANALYSIS
            with st.spinner("Running computer vision analysis..."):
                result = pipeline.analyze(
                    image,
                    image_name=uploaded_file.name
                )
                cv_json = result["json"]
        # ----------------------------------------------------
        # 3. GEMMA ANALYSIS
        # ----------------------------------------------------
            with st.spinner( "Gemma 4 is generating the explanation..."):

                try:
                    gemma_report = (
                        gemma_analyzer.generate_report(
                            cv_json
                        )
                    )
                except Exception as e:
                    st.error(
                        f"Gemma API Error: {e}"
                    )
                    gemma_report = None
#
            st.session_state.analysis_result = result
            st.session_state.cv_json = cv_json
            st.session_state.gemma_report = gemma_report
            st.session_state.patient_info = {

                "patient_name": (patient_name if patient_name else "Not Provided"),

                "age": (patient_age if patient_age > 0 else "Not Provided"),
                "gender": patient_gender,

                "clinical_information": (clinical_information if clinical_information else "Not Provided")
            }
            st.success(
                "Analysis completed successfully."
            )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.analysis_result is not None:


    # --------------------------------------------------------
    # GET SAVED RESULTS
    # --------------------------------------------------------

    result = (
        st.session_state.analysis_result
    )

    cv_json = (
        st.session_state.cv_json
    )

    gemma_report = (
        st.session_state.gemma_report
    )

    patient_info = (
        st.session_state.patient_info
    )


    st.divider()

    st.header(
        "📊 Analysis Results"
    )


    # ========================================================
    # COMPUTER VISION RESULTS
    # ========================================================

    st.subheader(
        "🔍 Computer Vision Analysis"
    )


    if cv_json["nodule_detected"]:

        st.success(
            "Nodule detected by YOLO"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Detection Confidence",
                f"{cv_json['detection_confidence'] * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Classification",
                cv_json["classification"]
            )

        with col3:

            st.metric(
                "Classification Confidence",
                f"{cv_json['classification_confidence'] * 100:.2f}%"
            )

        
    else:

        st.warning(
            "No thyroid nodule was detected."
        )

# ============================================================
# VISUAL EXPLANATION + REPORT
# ============================================================

if st.session_state.analysis_result is not None:

    # ========================================================
    # ROI + GRAD-CAM SIDE BY SIDE
    # ========================================================

    roi = result.get("roi")
    gradcam = result.get("gradcam")

    if roi is not None or gradcam is not None:

        st.subheader(
            "🧠 Explainable AI Visualization"
        )

        left_col, right_col = st.columns(2)


        # ====================================================
        # LEFT: ROI
        # ====================================================

        with left_col:

            st.markdown(
                "### 🎯 Detected Nodule ROI"
            )

            if roi is not None:

                st.image(

                    cv2.cvtColor(
                        roi,
                        cv2.COLOR_BGR2RGB
                    ),

                    width="stretch"

                )

                st.caption(
                    "Region extracted from the YOLO-detected nodule."
                )

            else:

                st.info(
                    "No ROI available."
                )


        # ====================================================
        # RIGHT: GRAD-CAM
        # ====================================================

        with right_col:

            st.markdown(
                "### 🔥 Grad-CAM Explainability"
            )

            if gradcam is not None:

                st.image(

                    cv2.cvtColor(
                        gradcam,
                        cv2.COLOR_BGR2RGB
                    ),

                    width="stretch"

                )

                st.info(
                    """
                    Grad-CAM highlights regions that contributed
                    to the CNN classification decision.

                    It is an explainability visualization and does
                    not prove a medical diagnosis.
                    """
                )

            else:

                st.info(
                    "Grad-CAM visualization is not available."
                )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.warning(
        """
        ⚠️ **Important Disclaimer**

        ThyroGuide provides an AI-assisted analysis for
        research and decision-support purposes.

        The results are not a definitive medical diagnosis
        and should not replace evaluation by a qualified
        healthcare professional.
        """
    )


    # ========================================================
    # AI-ASSISTED INTERPRETATION
    # ========================================================

    st.divider()

    st.header(
        "🤖 AI-Assisted Interpretation"
    )


    if gemma_report:

        st.markdown(
            gemma_report
        )

    else:

        st.warning(
            "AI-assisted interpretation is unavailable."
        )
    st.divider()


    # ========================================================
    # GENERATE PDF
    # ========================================================

    if gemma_report:
        os.makedirs(
            "results/reports",
            exist_ok=True
        )
        uploaded_filename = (
            st.session_state.uploaded_filename
        )
        report_filename = (
            uploaded_filename
            .rsplit(".", 1)[0]
            + "_ThyroGuide_Medical_Report.pdf"
        )
        report_path = os.path.join(
            "results/reports",
            report_filename
        )
        if "pdf_path" not in st.session_state:
            with st.spinner("Generating professional medical report..."):
                pdf_generator = (
                    ThyroGuidePDFReport()
                )
                pdf_path = (
                    pdf_generator.generate_pdf(
                        output_path=report_path,
                        patient_info=st.session_state.patient_info,
                        cv_json=cv_json,
                        gemma_report=gemma_report
                    )
                )
            st.session_state.pdf_path = pdf_path

            st.session_state.pdf_filename = report_filename
            #Download

        if (st.session_state.get("pdf_path")and os.path.exists(st.session_state.pdf_path)):
            st.success("Professional medical report generated successfully.")
            with open(pdf_path,"rb") as pdf_file:
                st.download_button(
                    label=(
                        "⬇️ Download Professional PDF Report"
                    ),
                    data=pdf_file.read(),
                    file_name=report_filename,
                    mime="application/pdf",
                    width="stretch"
                )
else:

    st.info(
        "The professional PDF report will be available after analysis."
    )
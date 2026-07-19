import streamlit as st
import cv2, torch
import numpy as np
from src.classification.cnn_classifier import ThyroidCNN
from src.pipeline.inference_pipeline import ThyroidInferencePipeline
from src.grad_cam.gradcamGenerator import GradCAMGenerator
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

from ultralytics import YOLO

yolo_model = YOLO(
    "models/yolo/best.pt"
)

cnn_model = ThyroidCNN(
    model_path="D:\\ECA\\RM_Thyroid\\thyroguide_app\\ThyroGuide-Explainable-AI-Assistant-for-Thyroid-Ultrasound-Diagnosis\\appfolder\\models\\cnn\\thyroid_cnn.pt",
    device=device
)


st.title("🩺 ThyroGuide AI")

uploaded_file = st.file_uploader(
    "Upload thyroid ultrasound image",
    type=["jpg", "jpeg", "png"]
)

gradcam = GradCAMGenerator(
    model=cnn_model,
    device=device
)
pipeline = ThyroidInferencePipeline(
    yolo_model=yolo_model,
    cnn_model=cnn_model,
    gradcam=gradcam,
    device=device
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Uploaded Ultrasound Image"
    )

    if st.button("🔍 Analyze"):

        with st.spinner("Running AI analysis..."):

            result = pipeline.analyze(
                image,
                image_name=uploaded_file.name
            )

        st.json(
            result["json"]
        )
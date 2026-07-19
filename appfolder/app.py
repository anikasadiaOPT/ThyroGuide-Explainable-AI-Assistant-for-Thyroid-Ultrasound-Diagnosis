import streamlit as st
import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO

from src.classification.cnn_classifier import ThyroidCNN
from src.pipeline.inference_pipeline import ThyroidInferencePipeline
from src.grad_cam.gradcamGenerator import GradCAMGenerator
from src.gemma_integration.gemma_analyzer import GemmaAnalyzer
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
# ==================================================
# 1. DEVICE
# ==================================================

device = ("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================
# 2. LOAD YOLO
# ==================================================

yolo_model = YOLO(

    "models/yolo/best.pt"

)


# ==================================================
# 3. LOAD CNN
# ==================================================

cnn_model = ThyroidCNN(

    model_path=(

        "models/cnn/thyroid_cnn.pt"

    ),
    device=device

)


# ==================================================
# 4. LOAD GRAD-CAM
# ==================================================

gradcam = GradCAMGenerator(

    model=cnn_model,

    transform=cnn_model.transform,

    device=device

)


# ==================================================
# 5. CREATE PIPELINE
# ==================================================

pipeline = ThyroidInferencePipeline(

    yolo_model=yolo_model,

    cnn_model=cnn_model,

    gradcam=gradcam,

    device=device

)


# ==================================================
# # 6. LOAD GEMMA
# # ==================================================

# gemma_analyzer = GemmaAnalyzer(

#     model_id="google/gemma-4-26B-A4B-it",

#     token=HF_TOKEN

# )


# ==================================================
# 7. STREAMLIT UI
# ==================================================

st.title(

    "🩺 ThyroGuide AI"

)

st.subheader(

    "Explainable AI Assistant for Thyroid Ultrasound Analysis"

)


uploaded_file = st.file_uploader(

    "Upload thyroid ultrasound image",

    type=[

        "jpg",

        "jpeg",

        "png"

    ]

)


if uploaded_file is not None:


    # ==============================================
    # READ IMAGE
    # ==============================================

    file_bytes = np.asarray(

        bytearray(

            uploaded_file.read()

        ),

        dtype=np.uint8

    )


    image = cv2.imdecode(

        file_bytes,

        cv2.IMREAD_COLOR

    )


    # ==============================================
    # DISPLAY ORIGINAL IMAGE
    # ==============================================

    st.image(

        cv2.cvtColor(

            image,

            cv2.COLOR_BGR2RGB

        ),

        caption="Uploaded Ultrasound Image",

        use_container_width=True

    )


    if st.button(

        "🔍 Analyze"

    ):


        with st.spinner(

            "Running AI analysis..."

        ):


            # ======================================
            # COMPUTER VISION PIPELINE
            # ======================================

            result = pipeline.analyze(

                image,

                image_name=uploaded_file.name

            )


            # ======================================
            # GET STRUCTURED JSON
            # ======================================

            cv_json = result[

                "json"

            ]


            # ======================================
            # GEMMA ANALYSIS
            # ======================================

            # gemma_report = (

            #     gemma_analyzer.generate_report(

            #         cv_json

            #     )

            # )


        # ==========================================
        # DISPLAY DETECTION RESULTS
        # ==========================================

        st.subheader(

            "🔍 Computer Vision Analysis"

        )


        st.json(

            cv_json

        )


        # ==========================================
        # DISPLAY ROI
        # ==========================================

        if result["roi"] is not None:

            st.subheader(

                "🎯 Detected Nodule ROI"

            )


            st.image(

                cv2.cvtColor(

                    result["roi"],

                    cv2.COLOR_BGR2RGB

                ),

                use_container_width=True

            )


        # ==========================================
        # DISPLAY GRAD-CAM
        # ==========================================

        if result["gradcam"] is not None:

            st.subheader(

                "🔥 Grad-CAM Explainability"

            )


            st.image(

                cv2.cvtColor(

                    result["gradcam"],

                    cv2.COLOR_BGR2RGB

                ),

                use_container_width=True

            )


        # ==========================================
        # DISPLAY GEMMA ANALYSIS
        # ==========================================

        st.subheader(

            "🤖 Gemma AI-Assisted Analysis"

        )


        st.write(

            gemma_report

        )


        # ==========================================
        # FINAL COMBINED OUTPUT
        # ==========================================

        final_output = {

            "image_name":

                uploaded_file.name,

            "computer_vision_analysis":

                cv_json,

            "gemma_analysis":

                gemma_report

        }


        st.subheader(

            "📄 Final Structured Output"

        )


        st.json(

            final_output

        )
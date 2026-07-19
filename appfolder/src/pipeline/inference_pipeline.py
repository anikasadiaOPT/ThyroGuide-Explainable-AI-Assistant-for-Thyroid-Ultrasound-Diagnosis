# src/pipeline/inference_pipeline.py

import cv2
import torch
import numpy as np

from src.preprocessing.preprocessing import crop_roi_from_bbox


class ThyroidInferencePipeline:

    def __init__(
        self,
        yolo_model,
        cnn_model,
        gradcam,
        device
    ):

        self.yolo_model = yolo_model
        self.cnn_model = cnn_model
        self.gradcam = gradcam
        self.device = device

        self.cnn_model.eval()


    def analyze(self, image, image_name = "uploaded_image"):

        # ==================================================
        # 1. YOLO DETECTION
        # ==================================================

        results = self.yolo_model(
            image,
            verbose=False
        )

        detection = self._extract_detection(
            results
        )


        # ==================================================
        # 2. NO NODULE CASE
        # ==================================================

        if detection is None:

            return {
                "image_name": image_name,
                "nodule_detected": False,
                "roi": None,
                "gradcam": None,
                "json": {
                    "nodule_detected": False,
                    "classification": None,
                    "classification_confidence": None
                }
            }


        bbox = detection["bounding_box"]


        # ==================================================
        # 3. ROI CROP
        # ==================================================

        roi = crop_roi_from_bbox(
            image,
            bbox
        )
        
        # ==================================================
        # 4. CNN CLASSIFICATION
        # ==================================================

        classification_result = self.cnn_model.predict(
            roi
        )

        classification = (
            classification_result["classification"]
        )

        classification_confidence = (
            classification_result[
                "classification_confidence"
            ]
        ) 


        # ==================================================
        # 6. GRAD-CAM
        # ==================================================

        gradcam_result = self.gradcam.generate(
            roi=roi,
            save_path=f"results/gradcam/{image_name}_gradcam.png"
        )

        grayscale_cam = cv2.imread(
            gradcam_result["gradcam_path"]
        )


        # ==================================================
        # 7. CREATE STRUCTURED JSON
        # ==================================================

        result = {
            "image_name": image_name,
            "nodule_detected": True,

            "bounding_box": bbox,

            "detection_confidence":
                detection[
                    "detection_confidence"
                ],

            "classification":
                classification,

            "classification_confidence":
                classification_confidence

        }


        # ==================================================
        # 8. RETURN EVERYTHING
        # ==================================================

        return {

            "roi": roi,

            "gradcam":
                grayscale_cam,

            "json":
                result

        }


    # ======================================================
    # YOLO DETECTION EXTRACTION
    # ======================================================

    def _extract_detection(
        self,
        results
    ):

        result = results[0]

        boxes = result.boxes


        if boxes is None:

            return None


        if len(boxes) == 0:

            return None


        # Select highest-confidence detection

        confidences = (
            boxes.conf
            .detach()
            .cpu()
            .numpy()
        )


        best_index = np.argmax(
            confidences
        )


        confidence = float(
            confidences[
                best_index
            ]
        )


        xyxy = (
            boxes.xyxy[
                best_index
            ]
            .detach()
            .cpu()
            .numpy()
        )


        xmin, ymin, xmax, ymax = (
            map(
                int,
                xyxy
            )
        )


        return {

            "bounding_box": {

                "xmin": xmin,

                "ymin": ymin,

                "xmax": xmax,

                "ymax": ymax

            },

            "detection_confidence":
                confidence

        }
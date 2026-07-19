# src/pipeline/inference_pipeline.py

import cv2
import numpy as np

from src.preprocessing.preprocessing import (
    crop_roi_from_bbox
)


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


    def analyze(

        self,

        image,

        image_name="uploaded_image"

    ):

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
        # 2. NO NODULE DETECTED
        # ==================================================

        if detection is None:

            return {

                "image_name":

                    image_name,

                "nodule_detected":

                    False,

                "roi":

                    None,

                "gradcam":

                    None,

                "json": {

                    "image_name":

                        image_name,

                    "nodule_detected":

                        False,

                    "bounding_box":

                        None,

                    "detection_confidence":

                        None,

                    "classification":

                        None,

                    "classification_confidence":

                        None

                }

            }


        # ==================================================
        # 3. GET BOUNDING BOX
        # ==================================================

        bbox = detection[

            "bounding_box"

        ]


        # ==================================================
        # 4. CROP ROI
        # ==================================================

        roi = crop_roi_from_bbox(

            image,

            bbox

        )


        # ==================================================
        # 5. CNN CLASSIFICATION
        # ==================================================

        classification_result = (

            self.cnn_model.predict(

                roi

            )

        )


        classification = (

            classification_result[

                "classification"

            ]

        )


        classification_confidence = (

            classification_result[

                "classification_confidence"

            ]

        )


        # ==================================================
        # 6. GRAD-CAM
        # ==================================================

        gradcam_path = (

            f"results/gradcam/"

            f"{image_name}_gradcam.png"

        )


        gradcam_result = (

            self.gradcam.generate(

                roi=roi,

                save_path=gradcam_path

            )

        )


        # Read generated Grad-CAM image

        gradcam_image = cv2.imread(

            gradcam_result[

                "gradcam_path"

            ]

        )


        # ==================================================
        # 7. CREATE STRUCTURED JSON
        # ==================================================

        result = {

            "image_name":

                image_name,

            "nodule_detected":

                True,

            "bounding_box":

                bbox,

            "detection_confidence":

                detection[

                    "detection_confidence"

                ],

            "classification":

                classification,

            "classification_confidence":

                classification_confidence,

            "gradcam_path":

                gradcam_result[

                    "gradcam_path"

                ]

        }


        # ==================================================
        # 8. RETURN COMPLETE RESULT
        # ==================================================

        return {

            "image_name":

                image_name,

            "roi":

                roi,

            "gradcam":

                gradcam_image,

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

        result = results[

            0

        ]


        boxes = result.boxes


        # No detection

        if boxes is None:

            return None


        if len(boxes) == 0:

            return None


        # ==================================================
        # SELECT HIGHEST-CONFIDENCE DETECTION
        # ==================================================

        confidences = (

            boxes.conf

            .detach()

            .cpu()

            .numpy()

        )


        best_index = int(

            np.argmax(

                confidences

            )

        )


        detection_confidence = float(

            confidences[

                best_index

            ]

        )


        # ==================================================
        # EXTRACT BOUNDING BOX
        # ==================================================

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

                "xmin":

                    xmin,

                "ymin":

                    ymin,

                "xmax":

                    xmax,

                "ymax":

                    ymax

            },

            "detection_confidence":

                detection_confidence

        }
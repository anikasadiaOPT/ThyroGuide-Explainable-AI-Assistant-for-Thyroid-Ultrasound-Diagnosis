from ultralytics import YOLO


class YOLODetector:

    def __init__(self, model_path, confidence_threshold=0.25):

        self.model = YOLO(model_path)

        self.confidence_threshold = (
            confidence_threshold
        )


    def detect(self, image):

        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False
        )

        result = results[0]

        if len(result.boxes) == 0:

            return {
                "nodule_detected": False,
                "bounding_box": None,
                "detection_confidence": None
            }

        # Select highest-confidence detection
        best_box_index = (
            result.boxes.conf.argmax()
        )

        box = result.boxes[
            best_box_index
        ]

        coordinates = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        confidence = (
            box.conf[0]
            .cpu()
            .item()
        )

        xmin, ymin, xmax, ymax = coordinates

        return {

            "nodule_detected": True,

            "bounding_box": {
                "xmin": int(xmin),
                "ymin": int(ymin),
                "xmax": int(xmax),
                "ymax": int(ymax)
            },

            "detection_confidence": float(
                confidence
            )
        }
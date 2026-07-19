import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()


class GemmaAnalyzer:

    def __init__(
        self,
        model_id="google/gemma-4-26B-A4B-it"
    ):

        self.model_id = model_id

        self.client = InferenceClient(
            api_key=os.getenv("HF_TOKEN")
        )


    def create_prompt(self, data):

        image_name = data.get(
            "image_name",
            "Not available"
        )

        nodule_detected = data.get(
            "nodule_detected",
            None
        )

        nodule_text = (
            "Yes"
            if nodule_detected is True
            else "No"
            if nodule_detected is False
            else "Not available"
        )

        detection_confidence = data.get(
            "detection_confidence"
        )

        classification = data.get(
            "classification",
            "Not available"
        )

        classification_confidence = data.get(
            "classification_confidence"
        )

        bounding_box = data.get(
            "bounding_box",
            "Not available"
        )


        detection_confidence_text = (

            f"{detection_confidence * 100:.2f}%"

            if detection_confidence is not None

            else "Not available"
        )


        classification_confidence_text = (

            f"{classification_confidence * 100:.2f}%"

            if classification_confidence is not None

            else "Not available"
        )


        prompt = f"""

You are an AI-assisted medical report explanation assistant.

The computer vision system has already produced the following results.

Do not perform a new classification.
Do not change the classification.
Only explain the provided results.

FACTS:

Image: {image_name}

Nodule detected: {nodule_text}

Detection confidence: {detection_confidence_text}

CNN classification: {classification}

Classification confidence: {classification_confidence_text}

Bounding box: {bounding_box}


Write exactly these six sections:

Detection Summary:
Explain whether a nodule was detected.

Classification Summary:
State exactly the provided CNN classification: {classification}

Confidence Interpretation:
Explain separately the detection confidence and classification confidence.

Explainability Summary:
Explain that Grad-CAM highlights image regions that contributed to the CNN decision.
The bounding box comes from YOLO detection.
Grad-CAM does not prove a diagnosis.

Patient-Friendly Explanation:
Explain the result in simple language.
Do not state that the model prediction is a confirmed diagnosis.

Important Limitations:
State that this is an AI-assisted prediction, not a definitive diagnosis,
and professional medical review is recommended.

Do not invent symptoms, medical history, ultrasound features,
TI-RADS scores, biopsy results, or clinical information.

Return only the six sections.

"""


        return prompt


    def generate_report(self, data):

        prompt = self.create_prompt(data)


        response = self.client.chat_completion(

            model=self.model_id,

            messages=[

                {

                    "role": "user",

                    "content": prompt

                }

            ],

            max_tokens=400,

            temperature=0.1

        )


        return response.choices[0].message.content
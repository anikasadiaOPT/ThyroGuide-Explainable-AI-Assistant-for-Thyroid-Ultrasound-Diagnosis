import cv2
import torch
import numpy as np

from pathlib import Path

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCAMGenerator:

    def __init__(
        self,
        model,
        transform,
        device
    ):

        # model is your ThyroidCNN wrapper
        self.model = model.model

        self.device = device

        self.transform = transform

        self.model.eval()


        # Last Conv2d layer
        self.target_layers = [

            self.model.features[12]

        ]


        self.cam = GradCAM(

            model=self.model,

            target_layers=self.target_layers

        )


    def generate(

        self,

        roi,

        save_path

    ):


        # ----------------------------------
        # 1. BGR → RGB
        # ----------------------------------

        img = cv2.cvtColor(

            roi,

            cv2.COLOR_BGR2RGB

        )


        # ----------------------------------
        # 2. Resize
        # ----------------------------------

        img = cv2.resize(

            img,

            (224, 224)

        )


        # ----------------------------------
        # 3. Normalize for visualization
        # ----------------------------------

        rgb_img = (

            img.astype(

                np.float32

            )

            / 255.0

        )


        # ----------------------------------
        # 4. Same transform as CNN
        # ----------------------------------

        input_tensor = self.transform(

            img

        ).unsqueeze(

            0

        ).to(

            self.device

        )


        # ----------------------------------
        # 5. CNN prediction
        # ----------------------------------

        with torch.no_grad():

            outputs = self.model(

                input_tensor

            )


            probabilities = torch.softmax(

                outputs,

                dim=1

            )


            pred = probabilities.argmax(

                dim=1

            ).item()


            confidence = probabilities[

                0,

                pred

            ].item()


        # ----------------------------------
        # 6. Grad-CAM
        # ----------------------------------

        grayscale_cam = self.cam(

            input_tensor=input_tensor

        )[0]


        # ----------------------------------
        # 7. Create visualization
        # ----------------------------------

        visualization = show_cam_on_image(

            rgb_img,

            grayscale_cam,

            use_rgb=True

        )


        # ----------------------------------
        # 8. Save visualization
        # ----------------------------------

        save_path = Path(

            save_path

        )


        save_path.parent.mkdir(

            parents=True,

            exist_ok=True

        )


        cv2.imwrite(

            str(save_path),

            cv2.cvtColor(

                visualization,

                cv2.COLOR_RGB2BGR

            )

        )


        # ----------------------------------
        # 9. Return result
        # ----------------------------------

        return {

            "prediction": pred,

            "confidence": confidence,

            "gradcam_path":

                str(

                    save_path

                )

        }
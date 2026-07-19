import cv2
import torch
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCAMGenerator:

    def __init__(self, model, device):

        self.model = model
        self.device = device

        self.model.eval()

        # Last Conv2d layer of your ThyroidCNN
        self.target_layers = [self.model.features[12]]

        self.cam = GradCAM(
            model=self.model,
            target_layers=self.target_layers
        )

    def generate(self, roi, save_path):

        # roi is already a BGR NumPy array from YOLO
    
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
    
        rgb_img = img.astype(np.float32) / 255.0
    
        input_tensor = self.transform(
            image=img
        )["image"].unsqueeze(0).to(self.device)
    
        outputs = self.model(input_tensor)
    
        probabilities = torch.softmax(outputs, dim=1)
    
        pred = probabilities.argmax(dim=1).item()
    
        confidence = probabilities[0, pred].item()
    
        grayscale_cam = self.cam(input_tensor=input_tensor)[0]

        visualization = show_cam_on_image(
            rgb_img,
            grayscale_cam,
            use_rgb=True
        )
    
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
        cv2.imwrite(
            str(save_path),
            cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        )
    
        return {
            "prediction": pred,
            "confidence": confidence,
            "gradcam_path": str(save_path)
        }

        
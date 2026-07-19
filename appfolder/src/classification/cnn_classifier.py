import torch
import torch.nn as nn
from torchvision import transforms


class ThyroidCNNModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ThyroidCNN:

    def __init__(self, model_path, device):

        self.device = device

        self.model = ThyroidCNNModel()

        state_dict = torch.load(
            model_path,
            map_location=device
        )

        self.model.load_state_dict(state_dict)

        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, roi):

        roi_rgb = roi[:, :, ::-1].copy()

        input_tensor = self.transform(
            roi_rgb
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            output = self.model(input_tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            predicted_class = output.argmax(
                dim=1
            ).item()

            confidence = probabilities[
                0,
                predicted_class
            ].item()

        class_names = {
            0: "Benign",
            1: "Malignant"
        }

        return {
            "class_id": predicted_class,
            "classification": class_names[predicted_class],
            "classification_confidence": confidence,
            "input_tensor": input_tensor
        }
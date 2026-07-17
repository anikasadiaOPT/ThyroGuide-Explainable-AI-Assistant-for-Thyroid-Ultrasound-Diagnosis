from pathlib import Path
import cv2
from google.colab.patches import cv2_imshow

import xml.etree.ElementTree as ET
class ROIDataset:
  def __init__(self, annotation_dir,image_dir):
    self.annotation_dir = Path(annotation_dir)
    self.image_dir = Path(image_dir)
    self.image_files = sorted(self.image_dir.glob("*.jpg"))

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    img_path = self.image_files[idx]
    img = cv2.imread(str(img_path))
    img_h, img_w = img.shape[:2]
    xml_path = self.annotation_dir / (img_path.stem + ".xml")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    box = root.find("object").find("bndbox")

    xmin = int(box.find("xmin").text)
    xmax = int(box.find("xmax").text)
    ymin = int(box.find("ymin").text)
    ymax = int(box.find("ymax").text)

    margin = 0.15

    w = xmax - xmin
    h = ymax - ymin

    xmin = max(0, int(xmin - margin * w))
    ymin = max(0, int(ymin - margin * h))
    xmax = min(img_w, int(xmax + margin * w))
    ymax = min(img_h, int(ymax + margin * h))

    roi = img[ymin:ymax, xmin:xmax]
    return roi, img_path.stem

  def save_all_roi(self, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(len(self)):
        roi, filename = self[i]      # Calls __getitem__()

        cv2.imwrite(
            str(save_dir / f"{filename}.png"),
            roi
        )

    print(f"Saved {len(self)} ROI images.")




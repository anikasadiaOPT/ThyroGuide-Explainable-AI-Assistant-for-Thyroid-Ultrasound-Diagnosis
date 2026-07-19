from pathlib import Path
import xml.etree.ElementTree as ET

class XML_to_YOLO:
  def __init__(self,annotation_dir, save_dir):
    self.annotation_dir = Path(annotation_dir)
    self.save_dir = Path(save_dir)
    self.save_dir.mkdir(parents=True, exist_ok=True)

    self.xml_files = sorted(self.annotation_dir.glob("*.xml"))


  #converting one xml
  def convert_one(self, xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)
    obj = root.find("object")
    label = 0

    box = obj.find("bndbox")

    xmin = float(box.find("xmin").text)
    ymin = float(box.find("ymin").text)
    xmax = float(box.find("xmax").text)
    ymax = float(box.find("ymax").text)

    x_center = ((xmin+xmax)/2)/width
    y_center = ((ymin+ ymax)/2)/height

    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    txt_path = self.save_dir / f"{xml_path.stem}.txt"

    with open(txt_path, "w") as f:
      f.write(
        f"{label} "
          f"{x_center:.6f} "
          f"{y_center:.6f} "
          f"{box_width:.6f} "
          f"{box_height:.6f}"
      )

  def convert_all(self):
    for xml_file in self.xml_files:
      self.convert_one(xml_file)
    print(f"Converted{len(self.xml_files)} XML files.")












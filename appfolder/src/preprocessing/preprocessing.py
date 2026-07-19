import cv2


def crop_roi_from_bbox(image, bbox, margin=0.15):
    """
    Crop ROI from an image using a bounding box.

    bbox format:
    {
        "xmin": int,
        "ymin": int,
        "xmax": int,
        "ymax": int
    }
    """

    img_h, img_w = image.shape[:2]

    xmin = bbox["xmin"]
    ymin = bbox["ymin"]
    xmax = bbox["xmax"]
    ymax = bbox["ymax"]

    # Bounding-box dimensions
    w = xmax - xmin
    h = ymax - ymin

    # Add margin
    xmin = max(0, int(xmin - margin * w))
    ymin = max(0, int(ymin - margin * h))

    xmax = min(img_w, int(xmax + margin * w))
    ymax = min(img_h, int(ymax + margin * h))

    # Crop ROI
    roi = image[ymin:ymax, xmin:xmax]

    return roi
import cv2
import os

VALID_EXTS = {"jpg", "jpeg", "png", "bmp", "tiff"}


def save_face_images(image_path, faces, output_dir, image_format="jpg"):
    if image_format.lower() not in VALID_EXTS:
        raise ValueError(f"Invalid image format: {image_format}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    base = os.path.splitext(os.path.basename(image_path))[0]

    for idx, face in enumerate(faces):
        x1, y1, x2, y2 = face["bbox"]
        crop = img[y1:y2, x1:x2]

        filename = f"{base}_face_{idx}.{image_format}"
        path = os.path.join(output_dir, filename)

        cv2.imwrite(path, crop)

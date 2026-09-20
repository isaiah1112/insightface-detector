import cv2
import os

VALID_EXTS = {"jpg", "jpeg", "png", "bmp", "tiff"}


def save_face_images(image_path, faces, output_dir, image_format="jpg", padding=0):
    if image_format.lower() not in VALID_EXTS:
        raise ValueError(f"Invalid image format: {image_format}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = img.shape[:2]
    base = os.path.splitext(os.path.basename(image_path))[0]

    for idx, face in enumerate(faces):
        x1, y1, x2, y2 = face["bbox"]

        # Apply padding safely
        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(width, x2 + padding)
        y2p = min(height, y2 + padding)

        crop = img[y1p:y2p, x1p:x2p]

        score = face.get("score", 0.0)
        sim = face.get("similarity", None)

        if sim is not None:
            filename = f"{base}_face_{idx}_score-{score:.3f}_sim-{sim:.3f}.{image_format}"
        else:
            filename = f"{base}_face_{idx}_score-{score:.3f}.{image_format}"

        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, crop)

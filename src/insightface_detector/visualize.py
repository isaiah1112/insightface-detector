import cv2
import os

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def ensure_extension(path):
    root, ext = os.path.splitext(path)
    if ext.lower() in VALID_EXTS:
        return path
    return root + ".jpg"   # default fallback

def draw_faces(image_path, faces, output_path=None):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for lx, ly in face["landmarks"]:
            cv2.circle(img, (int(lx), int(ly)), 2, (0, 0, 255), -1)

    if output_path:
        output_path = ensure_extension(output_path)
        cv2.imwrite(output_path, img)

    return img

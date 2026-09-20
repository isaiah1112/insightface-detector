import cv2
import numpy as np
from insightface.app import FaceAnalysis

try:
    from insightface.utils.face_align import norm_crop
except ImportError:  # pragma: no cover - optional module not available in some installs
    norm_crop = None


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class InsightFaceDetector:
    def __init__(self, det_size=(640, 640), providers=None, ctx_id=0):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        self.rec_model = self.app.models.get("recognition")

    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        faces = self.app.get(img)
        results = []

        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            landmarks = face.landmark_2d.tolist() if face.landmark_2d is not None else []
            embedding = face.embedding.tolist() if face.embedding is not None else None

            results.append({
                "bbox": bbox,
                "landmarks": landmarks,
                "score": float(face.det_score),
                "age": int(face.age) if face.age is not None else None,
                "gender": int(face.gender) if face.gender is not None else None,
                "embedding": embedding,
            })

        return results

    def get_embedding(self, image_path):
        """Extract embedding from a target face image.
        Requires that InsightFace can detect a face in this image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read target-face image: {image_path}")

        faces = self.app.get(img)
        if not faces:
            raise ValueError(
                "Target face could not be detected. "
                "Use a less-cropped image or increase padding when saving the target face."
            )

        if faces[0].embedding is None:
            raise ValueError("Target face has no embedding.")

        return faces[0].embedding.tolist()



    def filter_by_similarity(self, faces, target_embedding, threshold):
        """Return only faces whose embedding matches the target."""
        matched = []
        for f in faces:
            if f["embedding"] is None:
                continue
            sim = cosine_similarity(f["embedding"], target_embedding)
            if sim >= threshold:
                f["similarity"] = sim
                matched.append(f)
        return matched

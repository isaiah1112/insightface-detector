import cv2
from insightface.app import FaceAnalysis


class InsightFaceDetector:
    def __init__(self, det_size=(640, 640), providers=None, ctx_id=0):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        faces = self.app.get(img)
        results = []

        for face in faces:
            bbox = face.bbox.astype(int).tolist()

            landmarks = (
                face.landmark_2d.tolist()
                if face.landmark_2d is not None
                else []
            )

            embedding = (
                face.embedding.tolist()
                if face.embedding is not None
                else None
            )

            results.append({
                "bbox": bbox,
                "landmarks": landmarks,
                "score": float(face.det_score),
                "age": int(face.age) if face.age is not None else None,
                "gender": int(face.gender) if face.gender is not None else None,
                "embedding": embedding,
            })

        return results

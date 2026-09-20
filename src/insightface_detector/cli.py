import argparse
import json

from .detector import InsightFaceDetector
from .visualize import draw_faces


def main():
    parser = argparse.ArgumentParser(
        description="InsightFace-based face detection"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "-o", "--output-image", help="Path to save visualization", default=None
    )
    parser.add_argument(
        "-j", "--output-json", help="Path to save JSON results", default=None
    )
    args = parser.parse_args()

    detector = InsightFaceDetector()
    faces = detector.detect(args.image)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(faces, f, indent=2)

    if args.output_image:
        draw_faces(args.image, faces, args.output_image)

    if not args.output_json and not args.output_image:
        print(json.dumps(faces, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
import os
from .detector import InsightFaceDetector
from .visualize import save_face_images


def main():
    parser = argparse.ArgumentParser(
        description="InsightFace-based face detection"
    )
    parser.add_argument("image", help="Path to input image")

    parser.add_argument(
        "-o", "--output",
        help="Directory to save detected face images",
        default=None
    )

    parser.add_argument(
        "-f", "--image-format",
        help="Image format for saved faces (jpg, png, bmp, tiff)",
        default="jpg"
    )

    parser.add_argument(
        "-p", "--padding",
        type=int,
        help="Padding around each face crop in pixels",
        default=250
    )

    parser.add_argument(
        "--min-score",
        type=float,
        help="Minimum detection confidence score (0.0–1.0)",
        default=0.0
    )

    parser.add_argument(
        "-j", "--output-json",
        help="Path to save JSON results",
        default=None
    )

    args = parser.parse_args()

    detector = InsightFaceDetector()
    faces = detector.detect(args.image)

    # Apply score filtering
    faces = [f for f in faces if f["score"] >= args.min_score]

    # Save JSON if requested
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(faces, f, indent=2)

    # Save face crops if requested
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        save_face_images(
            image_path=args.image,
            faces=faces,
            output_dir=args.output,
            image_format=args.image_format,
            padding=args.padding,
        )

    # If no output flags, print JSON to stdout
    if not args.output and not args.output_json:
        print(json.dumps(faces, indent=2))

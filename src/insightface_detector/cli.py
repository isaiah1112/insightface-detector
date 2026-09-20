import argparse
import glob
import json
import os

from .detector import InsightFaceDetector
from .visualize import save_face_images


def collect_images(path):
    """Return a list of image paths from a file or directory."""
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
        images = []
        for ext in exts:
            images.extend(glob.glob(os.path.join(path, ext)))
        return sorted(images)

    raise ValueError(f"Input path is neither a file nor directory: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="InsightFace-based face detection"
    )
    parser.add_argument("image", help="Path to input image or directory")

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
        "--target-face",
        help="Path to an image containing the face to match",
        default=None
    )

    parser.add_argument(
        "--match-threshold",
        type=float,
        help="Cosine similarity threshold for target face matching",
        default=0.45
    )

    parser.add_argument(
        "-j", "--output-json",
        help="Path to save JSON results",
        default=None
    )
    
    parser.add_argument(
        "--sort-by",
        choices=["score", "similarity"],
        help="Sort detected faces by score or similarity",
        default=None
    )

    parser.add_argument(
        "--max-faces",
        type=int,
        help="Maximum number of faces to keep per image after sorting",
        default=1
    )

    args = parser.parse_args()

    # Collect input images (file or directory)
    input_images = collect_images(args.image)
    if not input_images:
        raise ValueError("No images found in input path.")

    detector = InsightFaceDetector()

    # Load target face embedding if provided
    target_embedding = None
    if args.target_face:
        target_embedding = detector.get_embedding(args.target_face)

    all_results = {}

    for img_path in input_images:
        faces = detector.detect(img_path)

        # Filter by score
        faces = [f for f in faces if f["score"] >= args.min_score]

        # Filter by target face similarity
        if target_embedding is not None:
            faces = detector.filter_by_similarity(
                faces,
                target_embedding,
                threshold=args.match_threshold
            )

        # Sort faces if requested
        if args.sort_by == "score":
            faces.sort(key=lambda f: f["score"], reverse=True)

        elif args.sort_by == "similarity":
            faces.sort(key=lambda f: f.get("similarity", 0.0), reverse=True)
            
        # Apply max-faces limit
        if args.max_faces is not None:
            faces = faces[:args.max_faces]

        all_results[img_path] = faces

        # Save face crops
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            save_face_images(
                image_path=img_path,
                faces=faces,
                output_dir=args.output,
                image_format=args.image_format,
                padding=args.padding,
            )

    # Save JSON summary
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)

    # If no output flags, print JSON to stdout
    if not args.output and not args.output_json:
        print(json.dumps(all_results, indent=2))

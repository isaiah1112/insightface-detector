import glob
import json
import os

import click

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


def _process_image(detector, img_path, *, target_embedding, min_score, match_threshold, sort_by, max_faces):
    faces = detector.detect(img_path)
    faces = [face for face in faces if face["score"] >= min_score]

    if target_embedding is not None:
        faces = detector.filter_by_similarity(
            faces,
            target_embedding,
            threshold=match_threshold,
        )

    if sort_by == "score":
        faces.sort(key=lambda face: face["score"], reverse=True)
    elif sort_by == "similarity":
        faces.sort(key=lambda face: face.get("similarity", 0.0), reverse=True)

    if max_faces is not None:
        faces = faces[:max_faces]

    return faces


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="insightface-detector", prog_name="insightface-detect")
@click.argument("image", type=click.Path(exists=True, readable=True, path_type=str))
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False, path_type=str),
    default=None,
    help="Directory to save detected face images.",
)
@click.option(
    "-f",
    "--image-format",
    type=click.Choice(["jpg", "png", "bmp", "tiff"], case_sensitive=False),
    default="jpg",
    show_default=True,
    help="Image format for saved faces.",
)
@click.option(
    "-p",
    "--padding",
    type=click.IntRange(min=0),
    default=250,
    show_default=True,
    help="Padding around each face crop in pixels.",
)
@click.option(
    "--min-score",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    show_default=True,
    help="Minimum detection confidence score (0.0–1.0).",
)
@click.option(
    "--target-face",
    type=click.Path(exists=True, readable=True, path_type=str),
    default=None,
    help="Path to a reference face image to match.",
)
@click.option(
    "--match-threshold",
    type=click.FloatRange(0.0, 1.0),
    default=0.45,
    show_default=True,
    help="Cosine similarity threshold for target face matching.",
)
@click.option(
    "-j",
    "--output-json",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Path to save JSON results.",
)
@click.option(
    "--sort-by",
    type=click.Choice(["score", "similarity"], case_sensitive=False),
    default=None,
    help="Sort detected faces by score or similarity.",
)
@click.option(
    "--max-faces",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Maximum number of faces to keep per image after sorting.",
)
def main(
    image,
    output,
    image_format,
    padding,
    min_score,
    target_face,
    match_threshold,
    output_json,
    sort_by,
    max_faces,
):
    """InsightFace-based face detection."""
    input_images = collect_images(image)
    if not input_images:
        raise click.ClickException("No images found in input path.")

    detector = InsightFaceDetector()
    target_embedding = detector.get_embedding(target_face) if target_face else None
    all_results = {}

    with click.progressbar(input_images, label="Processing images") as bar:
        for img_path in bar:
            faces = _process_image(
                detector,
                img_path,
                target_embedding=target_embedding,
                min_score=min_score,
                match_threshold=match_threshold,
                sort_by=sort_by,
                max_faces=max_faces,
            )
            all_results[img_path] = faces

            if output:
                os.makedirs(output, exist_ok=True)
                save_face_images(
                    image_path=img_path,
                    faces=faces,
                    output_dir=output,
                    image_format=image_format,
                    padding=padding,
                )

    if output_json:
        with open(output_json, "w") as f:
            json.dump(all_results, f, indent=2)

    if not output and not output_json:
        click.echo(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()

# InsightFace Detector

A lightweight CLI for face detection, face embeddings, and similarity matching built on InsightFace.

This project detects faces in one image or a directory of images, optionally filters by confidence, compares detections against a reference face, saves cropped face images, and emits JSON results.

## Features

- Detect faces with InsightFace's `buffalo_l` model
- Extract face landmarks and embeddings
- Filter detections by minimum confidence score
- Match detections to a reference face using cosine similarity
- Sort detections by score or similarity
- Save cropped face images with configurable padding
- Write results as JSON for each input image
- Works with a single file or a directory of images

## Installation

```bash
pip install -e .
```

## CLI usage

```bash
insightface-detect IMAGE [OPTIONS]
```

`IMAGE` may be either:

- a single image file, or
- a directory containing image files

## Common options

- `--version` — show the installed version and exit
- `-o, --output DIRECTORY` — save detected face crops to a directory
- `-f, --image-format [jpg|png|bmp|tiff]` — output format for saved face crops
- `-p, --padding INTEGER` — padding around each detected crop, in pixels
- `--min-score FLOAT` — minimum detection confidence in the range `0.0` to `1.0`
- `--target-face PATH` — path to a reference face image to match against detections
- `--match-threshold FLOAT` — cosine similarity threshold for target matching
- `-j, --output-json FILE` — write JSON results to a file
- `--sort-by [score|similarity]` — sort results by detection score or similarity
- `--max-faces INTEGER` — keep only the top N faces after sorting
- `-h, --help` — show help text

## Examples

### Detect faces and print JSON to stdout

```bash
insightface-detect images/sample.jpg
```

### Detect faces in a directory and save crops

```bash
insightface-detect images/ -o output/faces
```

### Save crops in PNG format with custom padding

```bash
insightface-detect images/sample.jpg \
  --output output/faces \
  --image-format png \
  --padding 300
```

### Filter by confidence and match against a reference face

```bash
insightface-detect images/sample.jpg \
  --min-score 0.5 \
  --target-face reference.jpg \
  --match-threshold 0.6 \
  --sort-by score \
  --max-faces 3
```

### Save a JSON report

```bash
insightface-detect images/sample.jpg \
  --output output/faces \
  --output-json results.json
```

### Show the installed version

```bash
insightface-detect --version
```

## Returned data

The JSON output contains one entry per input image. Each detected face includes:

```json
[
  {
    "bbox": [x1, y1, x2, y2],
    "landmarks": [[x, y], ...],
    "score": 0.99,
    "age": null,
    "gender": null,
    "embedding": [0.123, -0.456, ...],
    "similarity": 0.83
  }
]
```

Notes:

- `bbox` is the detected face bounding box.
- `landmarks` is a 2D landmark list when available.
- `score` is the InsightFace detection confidence.
- `similarity` appears only when `--target-face` is used and the face matches the comparison threshold.
- `embedding` is the face embedding vector used for similarity comparisons.

## Notes

- This project is optimized for the current InsightFace-based pipeline and currently uses the `buffalo_l` detection model.
- For a directory input, the CLI scans common image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.tiff`.
- The output naming convention for saved crop files includes the detection score and, when relevant, the similarity score.

## Development

```bash
pytest
```

The project currently includes a small CLI test suite to validate the command interface and version output.
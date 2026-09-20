# InsightFace Detector

InsightFace-based face detection package designed for Apple Silicon (M1–M4).

## Install

```bash
pip install -e .
```

## CLI Usage

The CLI accepts a single image path or a directory containing images:

```bash
insightface-detect IMAGE [OPTIONS]
```

### Common options

- `--version` — show the installed package version and exit
- `-o, --output DIRECTORY` — save detected face crops to a directory
- `-f, --image-format [jpg|png|bmp|tiff]` — output format for saved face crops
- `-p, --padding INTEGER` — padding around each detected face crop in pixels
- `--min-score FLOAT` — minimum detection confidence, from `0.0` to `1.0`
- `--target-face PATH` — optional reference face image to match against detections
- `--match-threshold FLOAT` — cosine similarity threshold for target matching
- `-j, --output-json FILE` — write JSON results for each detected face to a file
- `--sort-by [score|similarity]` — sort results by detection score or similarity
- `--max-faces INTEGER` — limit the number of faces kept after sorting
- `-h, --help` — show help text

### Examples

Detect faces in a single image and print JSON results to stdout:

```bash
insightface-detect images/sample.jpg
```

Detect faces in a directory and save cropped images to an output folder:

```bash
insightface-detect images/ -o output/faces
```

Save crops in PNG format with custom padding and a JSON summary:

```bash
insightface-detect images/sample.jpg \
  --output output/faces \
  --image-format png \
  --padding 300 \
  --output-json results.json
```

Filter detections by confidence and similarity to a reference face:

```bash
insightface-detect images/sample.jpg \
  --min-score 0.5 \
  --target-face reference.jpg \
  --match-threshold 0.6 \
  --sort-by score \
  --max-faces 3
```

Check the installed version:

```bash
insightface-detect --version
```
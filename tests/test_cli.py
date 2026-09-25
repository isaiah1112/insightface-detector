import click
from click.testing import CliRunner

from insightface_detector.cli import collect_images, main


def test_cli_uses_click_command():
    assert isinstance(main, click.core.Command)

    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cli_reports_version():
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "insightface-detect" in result.output
    assert "0.1.0" in result.output


def test_collect_images_supports_directory_and_multiple_paths(tmp_path):
    img1 = tmp_path / "one.jpg"
    img2 = tmp_path / "two.png"
    img1.write_bytes(b"fake")
    img2.write_bytes(b"fake")

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    img3 = image_dir / "three.jpg"
    img3.write_bytes(b"fake")

    result = collect_images([str(img1), str(image_dir), str(img2)])
    assert result == sorted([str(img1), str(img2), str(img3)])


def test_cli_accepts_multiple_images(monkeypatch, tmp_path):
    img1 = tmp_path / "one.jpg"
    img2 = tmp_path / "two.jpg"
    img1.write_bytes(b"fake")
    img2.write_bytes(b"fake")

    class FakeDetector:
        def get_embedding(self, path):
            return None

        def detect(self, image_path):
            return []

    monkeypatch.setattr("insightface_detector.cli.InsightFaceDetector", FakeDetector)

    result = CliRunner().invoke(main, [str(img1), str(img2)])

    assert result.exit_code == 0
    assert str(img1) in result.output
    assert str(img2) in result.output

import click
from click.testing import CliRunner

from insightface_detector.cli import main


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

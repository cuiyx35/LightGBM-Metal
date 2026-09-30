"""Regression checks for source-distribution inventory verification."""

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("check_sdist", Path(__file__).parents[1] / ".ci/check-sdist.py")
assert SPEC
assert SPEC.loader
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def make_archive(path, files):
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files:
            member = tarfile.TarInfo(f"lightgbm-4.7.0/{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


@pytest.mark.parametrize("count", [2, 810])
def test_changed_source_count_preserves_exact_inventory(tmp_path, count):
    source = tmp_path / "source"
    source.mkdir()
    files = [("pyproject.toml", b"[project]\n")]
    files += [(f"source-{index}.h", b"source\n") for index in range(count)]
    for name, content in files:
        (source / name).write_bytes(content)
    archive = tmp_path / "package.tar.gz"
    make_archive(archive, [*files, ("PKG-INFO", b"metadata")])
    assert CHECK.check_sdist(archive, source) == count + 2


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ([("extra.h", b"unexpected")], "Unexpected archive file"),
        ([("pyproject.toml", b"changed")], "Archive content differs"),
        ([], "Missing staged source files"),
        ([("../escape", b"unsafe")], "Unsafe archive path"),
        ([("pyproject.toml", b"original"), ("pyproject.toml", b"original")], "duplicate archive file"),
    ],
)
def test_invalid_archive_is_rejected(tmp_path, files, message):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_bytes(b"original")
    archive = tmp_path / "package.tar.gz"
    make_archive(archive, [*files, ("PKG-INFO", b"metadata")])
    with pytest.raises(ValueError, match=message):
        CHECK.check_sdist(archive, source)

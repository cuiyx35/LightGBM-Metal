"""Check a source archive against the deliberately staged package sources."""

import argparse
import hashlib
import tarfile
from pathlib import Path, PurePosixPath


def check_sdist(archive: Path, source: Path) -> int:
    """Reject missing, unexpected, changed, duplicate or unsafe archive files."""
    expected = {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).digest()
        for path in source.rglob("*")
        if path.is_file()
    }
    if not expected or "pyproject.toml" not in expected:
        raise ValueError("Missing isolated package source directory")
    seen = set()
    roots = set()
    with tarfile.open(archive, "r:gz") as distribution:
        for member in distribution.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            roots.add(path.parts[0])
            if member.isdir():
                continue
            name = PurePosixPath(*path.parts[1:]).as_posix()
            if not member.isfile() or name in seen:
                raise ValueError(f"Non-regular or duplicate archive file: {member.name}")
            seen.add(name)
            if name == "PKG-INFO":
                continue  # the build backend generates this package metadata
            if name not in expected:
                raise ValueError(f"Unexpected archive file: {name}")
            stream = distribution.extractfile(member)
            if stream is None or hashlib.sha256(stream.read()).digest() != expected[name]:
                raise ValueError(f"Archive content differs from staged source: {name}")
    if len(roots) != 1 or "PKG-INFO" not in seen:
        raise ValueError("Expected one package root and generated PKG-INFO")
    missing = expected.keys() - seen
    if missing:
        raise ValueError(f"Missing staged source files: {sorted(missing)}")
    return len(seen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    # The shell caller uses this verified count as pydistcheck's file limit.
    print(check_sdist(args.archive, args.source))  # noqa: T201

"""Apply the minimal Axolotl 0.17 role-boundary DictDefault compatibility fix."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


OLD = '''        if hasattr(spec, "model_dump"):
            d = spec.model_dump()
        else:
            d = dict(spec)
'''

NEW = '''        model_dump = getattr(spec, "model_dump", None)
        if callable(model_dump):
            d = model_dump()
        else:
            d = dict(spec)
'''


def sha256(path: Path) -> str:
    """Return the SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Patch one exact source block and retain the original beside it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    source = args.path.read_text(encoding="utf-8")
    if NEW in source and OLD not in source:
        print(f"already_patched sha256={sha256(args.path)}")
        return
    count = source.count(OLD)
    if count != 1:
        raise RuntimeError(f"expected exactly one target block, found {count}")

    backup = args.path.with_suffix(args.path.suffix + ".pre-role-boundary-fix")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    args.path.write_text(source.replace(OLD, NEW), encoding="utf-8")
    print(
        "patched "
        f"original_sha256={sha256(backup)} patched_sha256={sha256(args.path)} "
        f"backup={backup}"
    )


if __name__ == "__main__":
    main()

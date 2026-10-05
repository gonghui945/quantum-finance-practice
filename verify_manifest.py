"""Verify the original staged files without making network requests."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
failures = []
entries = (root / "MANIFEST.sha256").read_text().splitlines()
for line in entries:
    expected, name = line.split("  ", 1)
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(name)
if failures:
    raise SystemExit("Missing or changed files: " + ", ".join(failures))
print(f"Verified {len(entries)} staged files.")

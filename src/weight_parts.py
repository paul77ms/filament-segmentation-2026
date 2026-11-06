"""Store model weights in a plain Git repository as < 50 MB parts and reassemble them with checksums.

GitHub blocks files > 100 MiB (warns > 50 MiB); our fp32 fold checkpoints are 206,654,336 bytes each. Splitting
each checkpoint into equal byte ranges keeps every committed file below the warning size without Git LFS, and the
loader reassembles the exact original bytes (sha256 of every part and of the whole file is checked before use).

  split:   python src/weight_parts.py split  <ckpt.pt> <out_dir> [--max_bytes 50000000]
           -> <out_dir>/<name>.part00 .. partNN + <out_dir>/<name>.manifest.json
  verify:  python src/weight_parts.py verify <out_dir>/<name>.manifest.json [--against <ckpt.pt>]
  join:    python src/weight_parts.py join   <out_dir>/<name>.manifest.json [<out.pt>]   (writes the whole file)

  import sys; sys.path.insert(0, "src")
  from weight_parts import load_state_dict
  sd = load_state_dict("weights/b0_fold0.pt.manifest.json")

Manifest: {"file": name, "bytes": N, "sha256": <whole file>, "parts": [{"path": rel, "bytes": n, "sha256": h}, ...]}.
Only hashlib/json/io/pathlib (+ torch for loading) are used.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import sys
from pathlib import Path

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def split(ckpt: str | Path, out_dir: str | Path, max_bytes: int = 50_000_000) -> Path:
    ckpt, out_dir = Path(ckpt), Path(out_dir)
    data = ckpt.read_bytes()
    n_parts = max(1, math.ceil(len(data) / max_bytes))
    step = math.ceil(len(data) / n_parts)                      # equal-size parts (last one may be shorter)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for i in range(n_parts):
        chunk = data[i * step:(i + 1) * step]
        p = out_dir / f"{ckpt.name}.part{i:02d}"
        p.write_bytes(chunk)
        parts.append({"path": p.name, "bytes": len(chunk), "sha256": _sha256(chunk)})
    man = {"file": ckpt.name, "bytes": len(data), "sha256": _sha256(data), "parts": parts}
    mp = out_dir / f"{ckpt.name}.manifest.json"
    mp.write_text(json.dumps(man, indent=1))
    return mp


def reassemble(manifest: str | Path) -> bytes:
    """Concatenate the parts listed in the manifest; raise if any part or the whole file mismatches."""
    manifest = Path(manifest)
    man = json.loads(manifest.read_text())
    buf = bytearray()
    for p in man["parts"]:
        b = (manifest.parent / p["path"]).read_bytes()
        if b.startswith(LFS_POINTER_PREFIX):
            raise RuntimeError(f"{p['path']} is a Git LFS pointer, not data (run `git lfs pull`)")
        if len(b) != p["bytes"] or _sha256(b) != p["sha256"]:
            raise RuntimeError(f"part {p['path']} is corrupt or incomplete (size/sha256 mismatch)")
        buf += b
    if len(buf) != man["bytes"] or _sha256(bytes(buf)) != man["sha256"]:
        raise RuntimeError(f"reassembled {man['file']} does not match its sha256")
    return bytes(buf)


def join(manifest: str | Path, out: str | Path | None = None) -> Path:
    """Reassemble (with all checks) and write the whole file; returns its path (default: next to the manifest)."""
    manifest = Path(manifest)
    man = json.loads(manifest.read_text())
    out = Path(out) if out else manifest.parent / man["file"]
    data = reassemble(manifest)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(out)
    return out


def load_state_dict(manifest: str | Path, map_location="cpu"):
    import torch
    return torch.load(io.BytesIO(reassemble(manifest)), map_location=map_location, weights_only=True)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split"); s.add_argument("ckpt"); s.add_argument("out_dir")
    s.add_argument("--max_bytes", type=int, default=50_000_000)
    v = sub.add_parser("verify"); v.add_argument("manifest"); v.add_argument("--against", default=None)
    j = sub.add_parser("join"); j.add_argument("manifest"); j.add_argument("out", nargs="?", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "join":
        print("wrote", join(a.manifest, a.out))
    elif a.cmd == "split":
        mp = split(a.ckpt, a.out_dir, a.max_bytes)
        print(mp.read_text())
    else:
        data = reassemble(a.manifest)
        man = json.loads(Path(a.manifest).read_text())
        print(f"OK {man['file']}: {len(man['parts'])} parts, {len(data)} bytes, sha256 {man['sha256']}")
        if a.against:
            same = Path(a.against).read_bytes() == data
            print("byte-identical to", a.against, same)
            if not same:
                sys.exit(1)


if __name__ == "__main__":
    main()

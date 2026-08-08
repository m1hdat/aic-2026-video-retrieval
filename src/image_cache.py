from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from .settings import settings

def _roots() -> list[Path]:
    return [Path(x.strip().strip('"')) for x in settings.keyframe_roots.split(';') if x.strip()]

@lru_cache(maxsize=100_000)
def get_image(video_id: str, keyframe_file: str, image_relpath: str = "") -> str | None:
    """Resolve a local keyframe without assuming one fixed folder nesting."""
    names = list(dict.fromkeys([keyframe_file, f"{int(Path(keyframe_file).stem):06d}.jpg"]
                              if Path(keyframe_file).stem.isdigit() else [keyframe_file]))
    for root in _roots():
        candidates = []
        if image_relpath:
            candidates.append(root / image_relpath)
        for name in names:
            candidates += [root / video_id / name,
                           root / f"Keyframes_{video_id.split('_')[0]}" / "keyframes" / video_id / name,
                           root / "keyframes" / video_id / name]
        for path in candidates:
            if path.is_file():
                return str(path.resolve())
        for name in names:
            hit = next(root.rglob(f"{video_id}/{name}"), None) if root.is_dir() else None
            if hit and hit.is_file():
                return str(hit.resolve())
    return None

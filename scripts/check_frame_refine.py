from __future__ import annotations

from pathlib import Path

from src.settings import ROOT, settings


def configured_roots(value: str) -> list[Path]:
    return [
        Path(item.strip().strip('"'))
        for item in value.split(";")
        if item.strip()
    ]


def main() -> None:
    roots = configured_roots(settings.video_roots)
    existing = [path for path in roots if path.exists()]
    cache_root = ROOT / "data" / "cache" / "refined"
    cached = sum(1 for _ in cache_root.rglob("*.jpg")) if cache_root.exists() else 0

    print("Frame refinement code: available")
    print("ENABLE_FRAME_REFINE:", settings.enable_frame_refine)
    print("VIDEO_ROOTS configured:", len(roots))
    print("VIDEO_ROOTS existing:", len(existing))
    for path in roots:
        print(f"- {'OK' if path.exists() else 'MISSING'}: {path}")
    print("Refined frames already cached:", cached)

    if not settings.enable_frame_refine:
        print("STATUS: OFF. Retrieval returns the coarse keyframe frame_idx.")
        return
    if not existing:
        raise SystemExit(
            "STATUS: NOT READY. Frame refinement is enabled but no VIDEO_ROOTS path exists."
        )
    print(
        "STATUS: READY. Refined JPG files are generated on demand after a KIS/QA/TRAKE query."
    )


if __name__ == "__main__":
    main()

from __future__ import annotations
import os, shutil, zipfile
from pathlib import Path
from .settings import ROOT, settings

CACHE=ROOT/'data'/'video_cache'

def _sources():
    return [Path(x.strip().strip('"')) for x in settings.video_roots.split(';') if x.strip()]

def resolve_video(video_id: str) -> Path | None:
    """Find an mp4 directly or extract only the requested mp4 from a ZIP."""
    CACHE.mkdir(parents=True,exist_ok=True); cached=CACHE/f'{video_id}.mp4'
    if cached.exists(): return cached
    for source in _sources():
        if source.is_dir():
            direct=next(source.rglob(f'{video_id}.mp4'),None)
            if direct: return direct
            zips=source.rglob('*.zip')
        elif source.suffix.lower()=='.zip': zips=[source]
        else: continue
        for archive in zips:
            try:
                with zipfile.ZipFile(archive) as z:
                    member=next((n for n in z.namelist() if Path(n).name.lower()==f'{video_id}.mp4'.lower()),None)
                    if member:
                        with z.open(member) as src, cached.open('wb') as dst: shutil.copyfileobj(src,dst,8*1024*1024)
                        # Keep at most one extracted video to cap disk use.
                        for old in CACHE.glob('*.mp4'):
                            if old != cached: old.unlink(missing_ok=True)
                        return cached
            except (zipfile.BadZipFile,OSError): continue
    return None

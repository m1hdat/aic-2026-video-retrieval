from __future__ import annotations
import csv, tempfile
from pathlib import Path

def write_submission(kind: str, rows: list[dict]) -> str:
    out=Path(tempfile.gettempdir())/f'aic2026_{kind.lower()}_submission.csv'
    with out.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        if kind=='KIS':
            w.writerow(['video_id','frame_id'])
            for r in rows[:100]: w.writerow([r['video_id'],r['frame_id']])
        elif kind=='QA':
            w.writerow(['video_id','frame_id','answer'])
            for r in rows[:100]: w.writerow([r['video_id'],r['frame_id'],r['answer']])
        else:
            n=max((len(r['hits']) for r in rows[:100]),default=0); w.writerow(['video_id']+[f'frame_id_{i+1}' for i in range(n)])
            for r in rows[:100]: w.writerow([r['video_id']]+[h['frame_idx'] for h in r['hits']])
    return str(out)

from __future__ import annotations
import argparse, csv, hashlib, io, json, math, re, zipfile
from pathlib import Path
from typing import Iterable
from src.db import connect

VIDEO_RE = re.compile(r"L\d+_V\d+", re.I)
SUPPORTED_SUFFIXES = {'.jsonl', '.json', '.csv', '.parquet'}

def val(row, *names, default=None):
    lookup = {str(k).strip().lower(): v for k, v in row.items()}
    for name in names:
        v = lookup.get(name.lower())
        if v is not None and str(v).strip() not in ('', 'nan', 'None', 'null'):
            return v
    return default

def as_int(v):
    try: return int(float(v))
    except (TypeError, ValueError): return None

def as_float(v):
    try:
        x=float(v); return x if math.isfinite(x) else None
    except (TypeError, ValueError): return None

def infer_video(name: str):
    m=VIDEO_RE.search(name); return m.group(0).upper() if m else None

def records_from_bytes(name: str, raw: bytes) -> Iterable[dict]:
    suffix = Path(name).suffix.lower()

    def emit(payload):
        if isinstance(payload,list):
            for item in payload:
                if isinstance(item,dict): yield item
        elif isinstance(payload,dict):
            rows=payload.get('records') or payload.get('results') or payload.get('data')
            if isinstance(rows,list):
                for item in rows:
                    if isinstance(item,dict): yield item
            else: yield payload

    if suffix in {'.jsonl', '.json'}:
        text = raw.decode('utf-8-sig', errors='replace')
        decoder = json.JSONDecoder()
        position = 0
        while position < len(text):
            # Accept real newlines and broken JSONL files containing literal
            # separators such as "\\n", "\\r" or "\\r\\n".
            while position < len(text):
                if text[position].isspace() or text[position] in {',', '\ufeff'}:
                    position += 1
                    continue
                if text.startswith('\\r\\n', position):
                    position += 4
                    continue
                if text.startswith('\\n', position) or text.startswith('\\r', position):
                    position += 2
                    continue
                break
            if position >= len(text):
                break
            try:
                payload, position = decoder.raw_decode(text, position)
            except json.JSONDecodeError as exc:
                preview = text[position:position + 250].replace('\r','\\r').replace('\n','\\n')
                raise ValueError(
                    f'Invalid JSON in {name} at character {position}. Preview: {preview}'
                ) from exc
            yield from emit(payload)
    elif suffix == '.csv':
        yield from csv.DictReader(io.StringIO(raw.decode('utf-8-sig', errors='replace')))
    elif suffix == '.parquet':
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError('Parquet requires pandas + pyarrow; or ingest the companion CSV.') from e
        yield from pd.read_parquet(io.BytesIO(raw)).to_dict('records')

def iter_source(path: Path):
    def read_regular_file(file_path: Path):
        try:
            for row in records_from_bytes(str(file_path), file_path.read_bytes()):
                yield str(file_path), row
        except Exception as exc:
            raise RuntimeError(f'Failed to parse regular file: {file_path}') from exc

    def read_zip(zip_path: Path):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for member_name in sorted(archive.namelist()):
                    if member_name.endswith('/') or Path(member_name).suffix.lower() not in SUPPORTED_SUFFIXES:
                        continue
                    source_name = f'{zip_path}!{member_name}'
                    try:
                        for row in records_from_bytes(member_name, archive.read(member_name)):
                            yield source_name, row
                    except Exception as exc:
                        raise RuntimeError(
                            f'Failed to parse file inside ZIP: {source_name}'
                        ) from exc
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f'Invalid or corrupted ZIP: {zip_path}') from exc

    if path.is_dir():
        for p in sorted(path.rglob('*')):
            if not p.is_file():
                continue
            if p.suffix.lower()=='.zip':
                yield from read_zip(p)
            elif p.suffix.lower() in SUPPORTED_SUFFIXES:
                yield from read_regular_file(p)
    elif path.suffix.lower()=='.zip':
        yield from read_zip(path)
    elif path.suffix.lower() in SUPPORTED_SUFFIXES:
        yield from read_regular_file(path)
    else: raise ValueError(f'Unsupported input: {path}')

def source_key(kind, payload):
    raw=json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    return hashlib.sha1(f'{kind}|{raw}'.encode()).hexdigest()

def keyframe_lookup(pg, video_id, keyframe_n, frame_idx):
    if not video_id: return None
    if keyframe_n is not None:
        row=pg.execute("SELECT keyframe_n,frame_idx FROM keyframes WHERE video_id=%s AND keyframe_n=%s",
                       (video_id,keyframe_n)).fetchone()
        if row: return int(row['keyframe_n']),int(row['frame_idx'])
    if frame_idx is not None:
        row=pg.execute("""SELECT keyframe_n,frame_idx FROM keyframes WHERE video_id=%s
          ORDER BY abs(frame_idx-%s),keyframe_n LIMIT 1""",(video_id,frame_idx)).fetchone()
        if row: return int(row['keyframe_n']),int(row['frame_idx'])
    return None

def bbox_values(d):
    box=val(d,'bbox_xyxy','bbox','box','detection_box')
    if isinstance(box,str):
        try: box=json.loads(box)
        except Exception: box=None
    if isinstance(box,dict): box=[box.get(k) for k in ('x1','y1','x2','y2')]
    if isinstance(box,(list,tuple)) and len(box)>=4: return [as_float(x) for x in box[:4]]
    return [as_float(val(d,k)) for k in ('x1','y1','x2','y2')]

def expand_objects(name,row):
    video=str(val(row,'video_id',default=infer_video(name)) or '').upper() or None
    kn=as_int(val(row,'keyframe_n','n','keyframe_id','keyframe_idx'))
    fi=as_int(val(row,'frame_idx','frame_id','frame_index'))
    nested=val(row,'detections','objects')
    if isinstance(nested,str):
        try: nested=json.loads(nested)
        except Exception: nested=None
    if isinstance(nested,list):
        for d in nested: yield video,kn,fi,d
        return
    names=val(row,'detection_class_names')
    scores=val(row,'detection_scores')
    boxes=val(row,'detection_boxes')
    ids=val(row,'detection_class_ids')
    for xname in ('names','scores','boxes','ids'):
        x=locals()[xname]
        if isinstance(x,str) and x.strip().startswith('['):
            try:
                if xname=='names': names=json.loads(x)
                elif xname=='scores': scores=json.loads(x)
                elif xname=='boxes': boxes=json.loads(x)
                else: ids=json.loads(x)
            except Exception: pass
    if isinstance(names,list):
        for i,c in enumerate(names):
            yield video,kn,fi,{'class_name':c,'class_id':ids[i] if isinstance(ids,list) and i<len(ids) else None,
              'confidence':scores[i] if isinstance(scores,list) and i<len(scores) else 0,
              'bbox_xyxy':boxes[i] if isinstance(boxes,list) and i<len(boxes) else None}
        return
    if val(row,'class_name','label','object_name') is not None: yield video,kn,fi,row

def ingest_objects(pg, paths):
    inserted=skipped=0
    for path in paths:
        for name,row in iter_source(Path(path)):
            for video,kn,fi,d in expand_objects(name,row):
                mapped=keyframe_lookup(pg,video,kn,fi)
                cname=str(val(d,'class_name','label','object_name',default='')).strip().lower()
                conf=as_float(val(d,'confidence','score','detection_score',default=0))
                if not mapped or not cname or conf is None: skipped+=1; continue
                kn2,fi2=mapped; box=bbox_values(d)
                payload={'video_id':video,'keyframe_n':kn2,'class_name':cname,
                         'class_id':as_int(val(d,'class_id','label_id')),'confidence':conf,'bbox':box}
                pg.execute("""INSERT INTO object_detections
                  (video_id,keyframe_n,frame_idx,class_id,class_name,confidence,x1,y1,x2,y2,source_key)
                  VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                  ON CONFLICT(source_key) DO UPDATE SET confidence=excluded.confidence""",
                  (video,kn2,fi2,payload['class_id'],cname,conf,*box,source_key('object',payload)))
                inserted+=1
                if inserted%5000==0: pg.commit(); print(f'Objects: {inserted:,}',end='\r')
    pg.commit(); return inserted,skipped

def expand_ocr(name,row):
    video=str(val(row,'video_id',default=infer_video(name)) or '').upper() or None
    kn=as_int(val(row,'keyframe_n','n','keyframe_id','keyframe_idx'))
    fi=as_int(val(row,'frame_idx','frame_id','frame_index','start_frame'))
    nested=val(row,'ocr_results','texts','detections','results')
    if isinstance(nested,str):
        try: nested=json.loads(nested)
        except Exception: nested=None
    if isinstance(nested,list):
        for d in nested:
            if isinstance(d,dict): yield video,kn,fi,d
        return
    yield video,kn,fi,row

def ingest_ocr(pg,paths):
    inserted=skipped=0
    for path in paths:
        for name,row in iter_source(Path(path)):
            for video,kn,fi,d in expand_ocr(name,row):
                text=str(val(d,'text_content','text','ocr_text','transcription','content','label',default='')).strip()
                mapped=keyframe_lookup(pg,video,kn,fi)
                if not mapped or not text: skipped+=1; continue
                kn2,fi2=mapped; conf=as_float(val(
                    d,'confidence','score','probability','ocr_mean_conf','ocr_max_conf'
                ))
                box=bbox_values(d); bbox=None if all(x is None for x in box) else json.dumps(box)
                payload={'video_id':video,'keyframe_n':kn2,'text':text,'confidence':conf,'bbox':box}
                pg.execute("""INSERT INTO text_segments
                  (video_id,source,start_frame,end_frame,text_content,keyframe_n,frame_idx,confidence,bbox,source_key)
                  VALUES(%s,'ocr',%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
                  ON CONFLICT(source,source_key) WHERE source_key IS NOT NULL DO UPDATE SET
                  text_content=excluded.text_content,confidence=excluded.confidence,bbox=excluded.bbox""",
                  (video,fi2,fi2,text,kn2,fi2,conf,bbox,source_key('ocr',payload)))
                inserted+=1
                if inserted%5000==0: pg.commit(); print(f'OCR: {inserted:,}',end='\r')
    pg.commit(); return inserted,skipped

def main():
    ap=argparse.ArgumentParser(description='Incrementally ingest OCR and object outputs; does not touch Milvus.')
    ap.add_argument('--objects',nargs='*',default=[])
    ap.add_argument('--ocr',nargs='*',default=[])
    args=ap.parse_args()
    if not args.objects and not args.ocr: ap.error('Provide --objects and/or --ocr')
    with connect() as pg:
        if args.objects:
            n,bad=ingest_objects(pg,args.objects); print(f'\nObject rows processed: {n:,}; skipped: {bad:,}')
        if args.ocr:
            n,bad=ingest_ocr(pg,args.ocr); print(f'\nOCR rows processed: {n:,}; skipped: {bad:,}')
    print('Done. Milvus/SigLIP2 were not changed. Re-running is safe (upsert by source key).')

if __name__=='__main__': main()

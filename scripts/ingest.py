from __future__ import annotations
import argparse, csv, io, json, zipfile
from pathlib import Path
import numpy as np
import yaml
from pymilvus import MilvusClient
from tqdm import tqdm
from src.db import connect
from src.settings import ROOT, settings
from src.identity import pick, stable_pk

def sources(path: Path, suffix: str):
    if path.is_dir():
        for p in path.rglob(f"*{suffix}"): yield p.name, p.read_bytes()
    elif path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.lower().endswith(suffix): yield Path(n).name, z.read(n)
    else: raise ValueError(f"Không hỗ trợ: {path}")

def parse_maps(path: Path):
    maps={}
    for name, raw in sources(path, '.csv'):
        text=raw.decode('utf-8-sig'); rows=list(csv.DictReader(text.splitlines()))
        maps[Path(name).stem]=rows
    return maps

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features', type=Path, action='append', required=True, help='ZIP/thư mục output SigLIP2; lặp 4 lần')
    ap.add_argument('--maps', type=Path, required=True, help='ZIP/thư mục map-keyframes CSV')
    ap.add_argument('--batch-size', type=int, default=2000)
    args=ap.parse_args()
    cfg=yaml.safe_load((ROOT/'config/datasets.yaml').read_text(encoding='utf-8'))
    maps=parse_maps(args.maps); mc=MilvusClient(uri=settings.milvus_uri)
    total=0
    with connect() as pg:
      for feature_source in args.features:
        json_meta={Path(n).stem:json.loads(b) for n,b in sources(feature_source,'.json')}
        for filename, raw in tqdm(sources(feature_source,'.npy'), desc=feature_source.name):
            video_id=Path(filename).stem
            if video_id not in maps: raise RuntimeError(f"Thiếu map CSV: {video_id}")
            with io.BytesIO(raw) as f:
                vec=np.load(f, allow_pickle=False)
                rows=maps[video_id]; meta=json_meta.get(video_id,{})
                files=meta.get('image_files') or [f'{i:06d}.jpg' for i in range(len(rows))]
                if len(vec)!=len(rows) or len(files)!=len(rows):
                    raise RuntimeError(f"{video_id}: vector={len(vec)}, map={len(rows)}, files={len(files)}")
                if vec.ndim != 2 or vec.shape[1] != settings.embedding_dim:
                    raise RuntimeError(
                        f"{video_id}: embedding shape phải là "
                        f"[N,{settings.embedding_dim}], nhận {vec.shape}"
                    )
                if meta.get('model') not in (None, settings.embedding_model):
                    raise RuntimeError(
                        f"{video_id}: model={meta.get('model')!r}, "
                        f"cần {settings.embedding_model!r}"
                    )
                map_ns=[int(pick(r,'n','keyframe_idx','keyframe_id',default=i)) for i,r in enumerate(rows)]
                if len(set(map_ns)) != len(map_ns): raise RuntimeError(f"{video_id}: n trong map bị trùng")
                for i,(n,file) in enumerate(zip(map_ns,files)):
                    if Path(file).stem.isdigit() and int(Path(file).stem) != n:
                        raise RuntimeError(f"{video_id}: vector[{i}] / {file} không khớp map n={n}")
                group=video_id.split('_')[0]; data_part=str(cfg.get('parts',{}).get(group,''))
                pg.execute("INSERT INTO videos(video_id,group_id,data_part) VALUES(%s,%s,%s) ON CONFLICT(video_id) DO UPDATE SET data_part=excluded.data_part",(video_id,group,data_part))
                for start in range(0,len(rows),args.batch_size):
                    end=min(start+args.batch_size,len(rows)); payload=[]; sqlrows=[]
                    for i in range(start,end):
                        r=rows[i]; n=map_ns[i]; file=files[i]
                        frame=int(float(pick(r,'frame_idx','frame_id',default=n)))
                        pts=float(pick(r,'pts_time','timestamp','timestamp_sec',default=n))
                        fps=float(pick(r,'fps',default=0) or 0)
                        rel=f"Keyframes_{group}/keyframes/{video_id}/{file}"
                        pk=stable_pk(video_id,n)
                        payload.append({'id':pk,'video_id':video_id,'keyframe_n':n,'embedding':np.asarray(vec[i],dtype='float32').tolist()})
                        sqlrows.append((video_id,n,file,frame,pts,fps,rel))
                    with pg.cursor() as cur:
                        cur.executemany("""INSERT INTO keyframes(video_id,keyframe_n,keyframe_file,frame_idx,pts_time,fps,image_relpath)
                          VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(video_id,keyframe_n) DO UPDATE SET
                          keyframe_file=excluded.keyframe_file,frame_idx=excluded.frame_idx,pts_time=excluded.pts_time,
                          fps=excluded.fps,image_relpath=excluded.image_relpath""",sqlrows)
                    mc.upsert(settings.collection,payload); pg.commit(); total+=len(payload)
            pg.execute("INSERT INTO ingest_runs(source_name,row_count,status) VALUES(%s,%s,'done') ON CONFLICT(source_name) DO UPDATE SET row_count=excluded.row_count,status='done',updated_at=now()",(video_id,len(rows))); pg.commit()
    mc.flush(settings.collection); print(f"Hoàn tất: {total:,} vector")
if __name__=='__main__': main()

from __future__ import annotations
import argparse, csv, io, json, zipfile
from pathlib import Path
import numpy as np

def members(path: Path, suffix: str):
    if path.is_dir():
        for p in path.rglob(f"*{suffix}"):
            yield p.stem, p.read_bytes()
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.lower().endswith(suffix):
                    yield Path(name).stem, z.read(name)
    else:
        raise ValueError(f"Nguồn không hợp lệ: {path}")

def main():
    ap=argparse.ArgumentParser(description="Kiểm tra NPY/JSON/map trước khi ingest")
    ap.add_argument("--features", type=Path, action="append", required=True)
    ap.add_argument("--maps", type=Path, required=True)
    args=ap.parse_args()
    maps={k:list(csv.DictReader(io.StringIO(v.decode("utf-8-sig")))) for k,v in members(args.maps,".csv")}
    errors=[]; checked=0; vectors=0; seen=set()
    for source in args.features:
        metas={k:json.loads(v) for k,v in members(source,".json")}
        for video_id, raw in members(source,".npy"):
            if video_id in seen: errors.append(f"{video_id}: lặp ở nhiều feature source"); continue
            seen.add(video_id); checked += 1
            try:
                arr=np.load(io.BytesIO(raw),allow_pickle=False)
                rows=maps[video_id]; files=metas[video_id]["image_files"]
                if arr.ndim != 2 or arr.shape[1] != 512: errors.append(f"{video_id}: shape {arr.shape}, cần [N,512]")
                if not (len(arr)==len(rows)==len(files)): errors.append(f"{video_id}: vector={len(arr)}, map={len(rows)}, JSON={len(files)}")
                if metas[video_id].get("model") not in (None,"openai/clip-vit-base-patch32"): errors.append(f"{video_id}: sai CLIP model")
                ns=[int(float(r.get("n",r.get("keyframe_idx",i)))) for i,r in enumerate(rows)]
                if len(ns)!=len(set(ns)): errors.append(f"{video_id}: n bị trùng")
                for i,(n,name) in enumerate(zip(ns,files)):
                    if Path(name).stem.isdigit() and int(Path(name).stem)!=n:
                        errors.append(f"{video_id}: vector[{i}]={name} nhưng map n={n}"); break
                vectors += len(arr)
            except KeyError as exc: errors.append(f"{video_id}: thiếu {exc} trong map/JSON")
            except Exception as exc: errors.append(f"{video_id}: {exc}")
    missing_features=sorted(set(maps)-seen)
    extra_features=sorted(seen-set(maps))
    print(f"Videos kiểm tra: {checked:,}; vectors: {vectors:,}; map CSV: {len(maps):,}")
    groups=sorted({x.split('_')[0] for x in seen})
    print("Nhóm tìm thấy:", ", ".join(groups))
    if missing_features: errors.append(f"{len(missing_features)} map chưa có embedding; ví dụ {missing_features[:10]}")
    if extra_features: errors.append(f"{len(extra_features)} embedding chưa có map; ví dụ {extra_features[:10]}")
    if errors:
        print("\nKHÔNG HỢP LỆ:")
        for e in errors[:100]: print("-",e)
        raise SystemExit(1)
    print("DATA HỢP LỆ: mọi video đều khớp NPY = JSON = map CSV.")

if __name__ == "__main__": main()

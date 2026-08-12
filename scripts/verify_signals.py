from src.db import connect

def main():
    with connect() as pg:
        ocr = pg.execute("SELECT count(*) n FROM text_segments WHERE source='ocr'").fetchone()['n']
        objects = pg.execute("SELECT count(*) n FROM object_detections").fetchone()['n']
        orphan_ocr = pg.execute("""SELECT count(*) n FROM text_segments t
          WHERE source='ocr' AND NOT EXISTS (SELECT 1 FROM keyframes k
          WHERE k.video_id=t.video_id AND k.keyframe_n=t.keyframe_n)""").fetchone()['n']
        orphan_obj = pg.execute("""SELECT count(*) n FROM object_detections o
          WHERE NOT EXISTS (SELECT 1 FROM keyframes k
          WHERE k.video_id=o.video_id AND k.keyframe_n=o.keyframe_n)""").fetchone()['n']
        by_group = pg.execute("""SELECT split_part(video_id,'_',1) group_id,
          count(*) FILTER (WHERE source='ocr') ocr_rows
          FROM text_segments GROUP BY 1 ORDER BY 1""").fetchall()
        obj_group = pg.execute("""SELECT split_part(video_id,'_',1) group_id,count(*) object_rows
          FROM object_detections GROUP BY 1 ORDER BY 1""").fetchall()
    print(f"OCR rows: {ocr:,}")
    print(f"Object detections: {objects:,}")
    print("OCR by group:", ", ".join(f"{r['group_id']}={r['ocr_rows']:,}" for r in by_group) or "none")
    print("Objects by group:", ", ".join(f"{r['group_id']}={r['object_rows']:,}" for r in obj_group) or "none")
    if orphan_ocr or orphan_obj:
        raise SystemExit(f"VERIFY FAIL: orphan_ocr={orphan_ocr}, orphan_objects={orphan_obj}")
    print("VERIFY SIGNALS OK: every OCR/Object row maps to an existing keyframe.")

if __name__ == '__main__': main()

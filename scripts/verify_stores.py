from pymilvus import MilvusClient
from src.db import connect
from src.settings import settings

def main():
    mc=MilvusClient(uri=settings.milvus_uri)
    stats=mc.get_collection_stats(settings.collection)
    milvus_count=int(stats.get("row_count",0))
    with connect() as pg:
        pg_count=pg.execute("SELECT count(*) AS n FROM keyframes").fetchone()["n"]
        groups=pg.execute("SELECT group_id,count(*) videos FROM videos GROUP BY group_id ORDER BY group_id").fetchall()
        dup=pg.execute("SELECT count(*) n FROM (SELECT video_id,keyframe_n,count(*) FROM keyframes GROUP BY 1,2 HAVING count(*)>1)x").fetchone()["n"]
        empty=pg.execute("SELECT count(*) n FROM videos v WHERE NOT EXISTS(SELECT 1 FROM keyframes k WHERE k.video_id=v.video_id)").fetchone()["n"]
    print("PostgreSQL keyframes:",pg_count); print("Milvus entities:",milvus_count)
    print("Groups:",", ".join(f"{r['group_id']}={r['videos']} videos" for r in groups))
    if pg_count!=milvus_count or dup or empty:
        raise SystemExit(f"VERIFY FAIL: count_match={pg_count==milvus_count}, duplicate_keys={dup}, empty_videos={empty}")
    # Milvus is queried by deterministic PK during ingest; sample all PostgreSQL keys in bounded batches.
    from src.identity import stable_pk
    missing=[]
    with connect() as pg:
        with pg.cursor(name="verify_keys") as cur:
            cur.execute("SELECT video_id,keyframe_n FROM keyframes ORDER BY video_id,keyframe_n")
            while True:
                rows=cur.fetchmany(5000)
                if not rows: break
                ids=[stable_pk(r['video_id'],r['keyframe_n']) for r in rows]
                got=mc.get(settings.collection,ids=ids,output_fields=["video_id","keyframe_n"])
                actual={(x['video_id'],int(x['keyframe_n'])) for x in got}
                missing.extend((r['video_id'],r['keyframe_n']) for r in rows if (r['video_id'],r['keyframe_n']) not in actual)
                if missing: break
    if missing: raise SystemExit(f"VERIFY FAIL: Milvus thiếu khóa, ví dụ {missing[:10]}")
    print("VERIFY OK: PostgreSQL và Milvus có cùng mọi khóa video_id + n.")

if __name__ == "__main__": main()

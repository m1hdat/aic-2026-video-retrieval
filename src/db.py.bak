from __future__ import annotations
import re
import psycopg
from psycopg.rows import dict_row
from .settings import settings

def connect():
    return psycopg.connect(settings.pg_dsn, row_factory=dict_row)

def fetch_metadata(keys: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
    if not keys:
        return {}
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT video_id, keyframe_n, keyframe_file, frame_idx, pts_time,
                   fps, image_relpath
            FROM keyframes
            WHERE (video_id, keyframe_n) IN (
                SELECT * FROM unnest(%s::text[], %s::int[])
            )
        """, ([x[0] for x in keys], [x[1] for x in keys]))
        return {(r['video_id'], r['keyframe_n']): r for r in cur.fetchall()}

def search_ocr(query: str, limit: int = 300) -> list[dict]:
    q=query.strip()
    if not q: return []
    # Natural-language queries are usually longer than the literal text visible
    # in a frame. Search meaningful tokens with OR, while still strongly ranking
    # an exact phrase match.
    tokens=list(dict.fromkeys(
        x.lower() for x in re.findall(r"[^\W_]+",q,flags=re.UNICODE) if len(x)>=3
    ))[:20]
    ts_query=" | ".join(tokens) if tokens else q
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT t.video_id,t.keyframe_n,t.frame_idx,t.text_content,t.confidence,
          ts_rank_cd(t.search_vector,to_tsquery('simple',%s)) +
            CASE WHEN t.text_content ILIKE %s THEN 2.0 ELSE 0.0 END rank
          FROM text_segments t WHERE t.source='ocr' AND t.keyframe_n IS NOT NULL
          AND (t.search_vector @@ to_tsquery('simple',%s) OR t.text_content ILIKE %s)
          ORDER BY rank DESC,coalesce(t.confidence,0) DESC LIMIT %s""",
          (ts_query,f'%{q}%',ts_query,f'%{q}%',limit))
        return cur.fetchall()

def search_objects(terms: list[str], limit: int = 300) -> list[dict]:
    cleaned=list(dict.fromkeys(x.strip().lower() for x in terms if x.strip()))
    if not cleaned: return []
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT video_id,keyframe_n,frame_idx,class_name,max(confidence) confidence
          FROM object_detections WHERE lower(class_name)=ANY(%s::text[])
          GROUP BY video_id,keyframe_n,frame_idx,class_name
          ORDER BY max(confidence) DESC LIMIT %s""",(cleaned,limit))
        return cur.fetchall()

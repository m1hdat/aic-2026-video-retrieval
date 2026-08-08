from __future__ import annotations
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


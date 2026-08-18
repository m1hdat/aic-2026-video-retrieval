from __future__ import annotations
import re
import psycopg
from psycopg.rows import dict_row
from .settings import settings

OCR_ALIASES = (
    ('đồng bằng sông cửu long', 'đbscl'),
    ('thành phố hồ chí minh', 'tphcm'),
)

def _ocr_concepts(query: str) -> list[str]:
    """Build FTS concepts; every full-name/acronym pair counts once."""
    remaining=query.lower()
    concepts=[]
    for full_name,acronym in OCR_ALIASES:
        full_pattern=rf'\b{re.escape(full_name)}\b'
        acronym_pattern=rf'\b{re.escape(acronym)}\b'
        if re.search(full_pattern,remaining,flags=re.IGNORECASE) or re.search(
            acronym_pattern,remaining,flags=re.IGNORECASE
        ):
            full_tokens=re.findall(r"[^\W_]+",full_name,flags=re.UNICODE)
            concepts.append(f"({' & '.join(full_tokens)}) | {acronym}")
            remaining=re.sub(full_pattern,' ',remaining,flags=re.IGNORECASE)
            remaining=re.sub(acronym_pattern,' ',remaining,flags=re.IGNORECASE)
    concepts.extend(dict.fromkeys(
        token for token in re.findall(r"[^\W_]+",remaining,flags=re.UNICODE)
        if len(token)>=3
    ))
    return concepts[:20]

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
    q = query.strip()
    if not q:
        return []

    # OCR text is noisy, so requiring every word would lose useful matches.
    # However, ranking a plain OR-query with ts_rank_cd lets one common word
    # repeated many times beat a frame that contains nearly the whole query.
    # Rank by distinct query-term coverage first; term frequency is only a
    # tie-breaker after coverage.
    concepts = _ocr_concepts(q)
    if not concepts:
        return []
    ts_query = " | ".join(f"({concept})" for concept in concepts)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""
          WITH candidates AS (
            SELECT t.video_id, t.keyframe_n, t.frame_idx,
                   t.text_content, t.confidence,
                   (
                     SELECT count(*)
                     FROM unnest(%s::text[]) AS u(term_query)
                     WHERE t.search_vector @@ to_tsquery('simple', term_query)
                   ) AS matched_terms,
                   ts_rank_cd(t.search_vector, to_tsquery('simple', %s)) AS fts_rank,
                   CASE WHEN t.text_content ILIKE %s THEN 1 ELSE 0 END AS exact_phrase,
                   %s::int AS query_terms
            FROM text_segments t
            WHERE t.source = 'ocr' AND t.keyframe_n IS NOT NULL
              AND (t.search_vector @@ to_tsquery('simple', %s)
                   OR t.text_content ILIKE %s)
          )
          SELECT video_id, keyframe_n, frame_idx, text_content, confidence,
                 (matched_terms + exact_phrase * 10 + LEAST(fts_rank, 0.99))::real AS rank,
                 matched_terms, query_terms
          FROM candidates
          ORDER BY exact_phrase DESC,
                   matched_terms DESC,
                   fts_rank DESC,
                   coalesce(confidence, 0) DESC
          LIMIT %s
        """, (concepts, ts_query, f'%{q}%', len(concepts),
              ts_query, f'%{q}%', limit))
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

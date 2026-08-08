CREATE TABLE IF NOT EXISTS videos (
  video_id text PRIMARY KEY,
  group_id text NOT NULL,
  data_part text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS keyframes (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  video_id text NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
  keyframe_n integer NOT NULL,
  keyframe_file text NOT NULL,
  frame_idx integer NOT NULL,
  pts_time double precision,
  fps double precision,
  image_relpath text NOT NULL,
  video_relpath text,
  UNIQUE(video_id, keyframe_n)
);
CREATE INDEX IF NOT EXISTS keyframes_video_frame_idx ON keyframes(video_id, frame_idx);

CREATE TABLE IF NOT EXISTS text_segments (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  video_id text NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
  source text NOT NULL CHECK (source IN ('ocr','asr')),
  start_frame integer,
  end_frame integer,
  text_content text NOT NULL,
  search_vector tsvector GENERATED ALWAYS AS
    (to_tsvector('simple', coalesce(text_content, ''))) STORED
);
CREATE INDEX IF NOT EXISTS text_segments_search_idx ON text_segments USING gin(search_vector);
CREATE INDEX IF NOT EXISTS text_segments_video_idx ON text_segments(video_id, source);

CREATE TABLE IF NOT EXISTS ingest_runs (
  source_name text PRIMARY KEY,
  row_count bigint NOT NULL DEFAULT 0,
  status text NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);

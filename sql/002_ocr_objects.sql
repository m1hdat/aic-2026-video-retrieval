ALTER TABLE text_segments ADD COLUMN IF NOT EXISTS keyframe_n integer;
ALTER TABLE text_segments ADD COLUMN IF NOT EXISTS frame_idx integer;
ALTER TABLE text_segments ADD COLUMN IF NOT EXISTS confidence double precision;
ALTER TABLE text_segments ADD COLUMN IF NOT EXISTS bbox jsonb;
ALTER TABLE text_segments ADD COLUMN IF NOT EXISTS source_key text;

CREATE UNIQUE INDEX IF NOT EXISTS text_segments_source_key_uq
  ON text_segments(source, source_key) WHERE source_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS text_segments_keyframe_idx
  ON text_segments(video_id, keyframe_n);

CREATE TABLE IF NOT EXISTS object_detections (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  video_id text NOT NULL,
  keyframe_n integer NOT NULL,
  frame_idx integer NOT NULL,
  class_id integer,
  class_name text NOT NULL,
  confidence double precision NOT NULL,
  x1 double precision,
  y1 double precision,
  x2 double precision,
  y2 double precision,
  source_key text NOT NULL UNIQUE,
  FOREIGN KEY(video_id, keyframe_n)
    REFERENCES keyframes(video_id, keyframe_n) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS object_class_idx
  ON object_detections(lower(class_name));
CREATE INDEX IF NOT EXISTS object_keyframe_idx
  ON object_detections(video_id, keyframe_n);


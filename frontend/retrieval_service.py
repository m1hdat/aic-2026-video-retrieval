from __future__ import annotations
from src.search_engine import SearchEngine
from src.image_cache import get_image
from src.visual_qa import VisualQA
from src.settings import settings

class RetrievalService:
    """Giữ nguyên contract frontend cũ: KIS, QA và TRAKE."""
    def __init__(self): self.engine=SearchEngine(); self.vqa=VisualQA()
    def _adapt(self, rows):
        out=[]
        for r in rows:
            image=get_image(r['video_id'],r['keyframe_file'],r.get('image_relpath',''))
            out.append({**r,'frame_id':r['frame_idx'],'keyframe_id':r['keyframe_file'],'image_path':image})
        return out
    def search_kis(self, query, top_k=20):
        rows=self._adapt(self.engine.search_by_text(query,top_k))
        refine_query=self.engine.translator.to_english(query)
        for r in rows[:settings.refine_top_n]:
            rr=self.engine.refiner.refine(r['video_id'],r['frame_idx'],r.get('fps') or 0,refine_query)
            r.update(rr); r['frame_id']=r['frame_idx']
            if r.get('refined_image_path'): r['image_path']=r['refined_image_path']
        return rows
    def search_qa(self, event_description, question, top_k=20):
        query=' '.join(x.strip() for x in [event_description,question] if x.strip())
        rows=self.search_kis(query,min(100,max(top_k*3,30))); output=[]
        for r in rows:
            if not r.get('image_path'): continue
            try: answer=self.vqa.answer(r['image_path'],self.engine.translator.to_english(question))
            except Exception:
                continue
            if not answer.strip():
                continue
            output.append({**r,'answer':answer})
            if len(output)>=top_k: break
        return output
    def search_trake(self, events, top_videos=10):
        rows = self.engine.search_sequence(events, top_videos)

        for video_row in rows:
            video_id = video_row.get("video_id", "")
            adapted_hits = []

            for hit in video_row.get("hits", []):
                hit_video_id = hit.get("video_id") or video_id
                keyframe_file = (
                    hit.get("keyframe_file")
                    or hit.get("keyframe_id")
                    or ""
                )
                image_relpath = hit.get("image_relpath") or ""

                image_path = None

                if keyframe_file or image_relpath:
                    try:
                        image_path = get_image(
                            hit_video_id,
                            keyframe_file,
                            image_relpath,
                        )
                    except Exception:
                        image_path = None

                adapted_hits.append(
                    {
                        **hit,
                        "video_id": hit_video_id,
                        "frame_id": hit.get(
                            "frame_idx",
                            hit.get("frame_id"),
                        ),
                        "keyframe_id": keyframe_file,
                        "image_path": image_path,
                    }
                )

            video_row["hits"] = adapted_hits

        return rows
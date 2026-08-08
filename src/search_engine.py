from __future__ import annotations
from collections import defaultdict
from pymilvus import MilvusClient
from .clip_encoder import TextEncoder
from .db import fetch_metadata
from .settings import settings
from .frame_refiner import FrameRefiner
from .translator import Translator

class SearchEngine:
    def __init__(self):
        self.client = MilvusClient(uri=settings.milvus_uri)
        self.encoder = TextEncoder()
        self.refiner = FrameRefiner(self.encoder)
        self.translator = Translator()

    @staticmethod
    def _variants(query: str) -> list[str]:
        q = query.strip()
        return list(dict.fromkeys([q, f"a photo of {q}", f"a video frame showing {q}"]))

    def search_by_text(self, query: str, top_k: int = 100) -> list[dict]:
        variants = self._variants(self.translator.to_english(query))
        vectors = self.encoder.encode(variants)
        fused: dict[int, dict] = {}
        for result_set in self.client.search(
            collection_name=settings.collection,
            data=vectors.tolist(), limit=max(150, top_k),
            output_fields=["video_id", "keyframe_n"],
            search_params={"metric_type":"COSINE", "params":{"nprobe":64}},
        ):
            for rank, hit in enumerate(result_set, 1):
                pk = int(hit["id"])
                score = 1.0 / (60 + rank)  # reciprocal-rank fusion
                item = fused.setdefault(pk, {"rrf":0.0, "hit":hit})
                item["rrf"] += score
                if hit["distance"] > item["hit"]["distance"]:
                    item["hit"] = hit
        ordered = sorted(fused.values(), key=lambda x:x["rrf"], reverse=True)[:top_k]
        keys = [(x["hit"]["entity"]["video_id"], int(x["hit"]["entity"]["keyframe_n"])) for x in ordered]
        metadata = fetch_metadata(keys)
        output = []
        for rank, (entry, key) in enumerate(zip(ordered, keys), 1):
            row = metadata.get(key)
            if not row:
                continue
            output.append({"rank":rank, "score":float(entry["hit"]["distance"]), **row})
        return output

    def search_sequence(self, events: list[str], top_videos: int = 10) -> list[dict]:
        english_events=[self.translator.to_english(e) for e in events if e.strip()]
        per_event = [self.search_by_text(e, 300) for e in english_events]
        by_video = defaultdict(lambda: defaultdict(list))
        for ei, rows in enumerate(per_event):
            for r in rows:
                by_video[r["video_id"]][ei].append(r)
        candidates=[]
        for vid, hits in by_video.items():
            if len(hits) != len(per_event):
                continue
            # Dynamic programming: globally optimal increasing sequence, not greedy.
            layers=[sorted(hits[i][:120],key=lambda r:r['frame_idx']) for i in range(len(per_event))]
            states=[(r['score'],[r]) for r in layers[0]]
            for layer in layers[1:]:
                nxt=[]
                for r in layer:
                    prev=[s for s in states if s[1][-1]['frame_idx']<r['frame_idx']]
                    if prev:
                        score,path=max(prev,key=lambda x:x[0]); nxt.append((score+r['score'],path+[r]))
                states=nxt
                if not states: break
            if states: total,selected=max(states,key=lambda x:x[0])
            else: selected=[]; total=0.0
            if len(selected)==len(per_event):
                candidates.append({"video_id":vid,"score":total/len(selected),"hits":selected})
        candidates.sort(key=lambda x:x["score"], reverse=True)
        output=[]
        for i,x in enumerate(candidates[:top_videos]):
            refined=[]
            for event,hit in zip(english_events,x['hits']):
                rr=(self.refiner.refine(x['video_id'],hit['frame_idx'],hit.get('fps') or 0,event)
                    if i < settings.trake_refine_top_n else {'frame_idx':hit['frame_idx'],'refined':False})
                refined.append({**hit,**rr})
            output.append({'rank':i+1,**x,'hits':refined})
        return output

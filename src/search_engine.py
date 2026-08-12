from __future__ import annotations
from collections import defaultdict
from pymilvus import MilvusClient
from .clip_encoder import TextEncoder
from .db import fetch_metadata, search_ocr, search_objects
from .settings import settings
from .frame_refiner import FrameRefiner
from .translator import Translator

class SearchEngine:
    COCO_LABELS = (
        'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
        'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
        'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
        'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
        'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
        'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
        'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
        'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
        'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
        'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
    )
    def __init__(self):
        self.client = MilvusClient(uri=settings.milvus_uri)
        self.encoder = TextEncoder()
        self.refiner = FrameRefiner(self.encoder)
        self.translator = Translator()

    @staticmethod
    def _variants(original: str, english: str) -> list[str]:
        # SigLIP2 is multilingual: retain Vietnamese and add English API output.
        candidates = [
            original.strip(),
            english.strip(),
            f"This is a photo of {english.strip()}.",
            f"A video frame showing {english.strip()}.",
        ]
        return [item for item in dict.fromkeys(candidates) if item]

    def search_by_text(self, query: str, top_k: int = 100) -> list[dict]:
        english=self.translator.to_english(query)
        variants = self._variants(query, english)
        vectors = self.encoder.encode(variants)
        fused: dict[tuple[str,int], dict] = {}
        for result_set in self.client.search(
            collection_name=settings.collection,
            data=vectors.tolist(), limit=max(150, top_k),
            output_fields=["video_id", "keyframe_n"],
            search_params={"metric_type":"COSINE", "params":{"nprobe":64}},
        ):
            for rank, hit in enumerate(result_set, 1):
                key=(hit["entity"]["video_id"],int(hit["entity"]["keyframe_n"]))
                score = 1.0 / (60 + rank)  # reciprocal-rank fusion
                item = fused.setdefault(key, {"rrf":0.0,"clip_score":0.0,"sources":set(),"matches":[]})
                item["rrf"] += score
                item['clip_score']=max(item['clip_score'],float(hit['distance']))
                item['sources'].add('siglip2')

        if settings.enable_ocr_search:
            for rank,row in enumerate(search_ocr(query,settings.signal_candidate_k),1):
                key=(row['video_id'],int(row['keyframe_n']))
                item=fused.setdefault(key,{"rrf":0.0,"clip_score":0.0,"sources":set(),"matches":[]})
                item['rrf']+=1.0/(60+rank); item['sources'].add('ocr')
                if len(item['matches'])<3: item['matches'].append(row['text_content'])

        if settings.enable_object_search:
            # COCO labels are English; use query tokens plus a few safe Vietnamese aliases.
            aliases={'người':'person','xe':'car','ô tô':'car','oto':'car','xe máy':'motorcycle',
                     'xemay':'motorcycle','mô tô':'motorcycle','xe buýt':'bus','chó':'dog','mèo':'cat',
                     'chai':'bottle','ghế':'chair','điện thoại':'cell phone','laptop':'laptop'}
            low=query.lower(); english_low=english.lower()
            terms=[label for label in self.COCO_LABELS if label in english_low]
            terms += [x.strip('.,!?;:') for x in english_low.split()]
            terms += [v for k,v in aliases.items() if k in low]
            for rank,row in enumerate(search_objects(terms,settings.signal_candidate_k),1):
                key=(row['video_id'],int(row['keyframe_n']))
                item=fused.setdefault(key,{"rrf":0.0,"clip_score":0.0,"sources":set(),"matches":[]})
                item['rrf']+=1.0/(60+rank); item['sources'].add('object')
                label=f"object: {row['class_name']} ({float(row['confidence']):.2f})"
                if label not in item['matches'] and len(item['matches'])<3: item['matches'].append(label)

        ordered = sorted(fused.items(), key=lambda x:x[1]["rrf"], reverse=True)[:top_k]
        keys = [x[0] for x in ordered]
        metadata = fetch_metadata(keys)
        output = []
        for rank, (entry, key) in enumerate(zip(ordered, keys), 1):
            row = metadata.get(key)
            if not row:
                continue
            info=entry[1]
            output.append({"rank":rank,"score":float(info['rrf']),"clip_score":info['clip_score'],
                           "matched_sources":"+".join(sorted(info['sources'])),
                           "matched_text":" | ".join(info['matches']),**row})
        return output

    def search_sequence(self, events: list[str], top_videos: int = 10) -> list[dict]:
        original_events=[e.strip() for e in events if e.strip()]
        english_events=[self.translator.to_english(e) for e in original_events]
        per_event = [self.search_by_text(e, 300) for e in original_events]
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

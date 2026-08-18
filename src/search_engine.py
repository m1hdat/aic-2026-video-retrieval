from __future__ import annotations
from collections import defaultdict
import re
from pymilvus import MilvusClient
from .clip_encoder import TextEncoder
from .db import fetch_metadata, search_ocr, search_objects
from .settings import settings
from .frame_refiner import FrameRefiner
from .translator import Translator

class SearchEngine:
    TEMPORAL_DEDUP_SECONDS = 2.0
    OCR_INTENT_MARKERS = (
        'dòng chữ', 'chữ ghi', 'ghi dòng', 'ghi chữ', 'tiêu đề',
        'phụ đề', 'văn bản', 'nội dung chữ', 'text trên', 'text trong',
    )
    OCR_NEWS_PREFIXES = (
        'bản tin truyền hình về', 'bản tin nói về', 'bản tin có nội dung',
        'bản tin về',
    )
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

    def search_by_text(self, query: str, top_k: int = 100,
                       signal_sources=None) -> list[dict]:
        return self.search_many_by_text([query],top_k,signal_sources)[0]

    @staticmethod
    def _empty_fused_item():
        return {
            "rrf":0.0, "clip_rrf":0.0, "ocr_rrf":0.0, "object_rrf":0.0,
            "clip_score":0.0, "final_score":0.0,
            "sources":set(), "matches":[],
        }

    @classmethod
    def _prepare_ocr_query(cls, query: str) -> tuple[str, bool]:
        """Return the literal text portion and whether OCR is explicit intent."""
        low = query.lower()
        for marker in cls.OCR_NEWS_PREFIXES:
            start = low.find(marker)
            if start >= 0:
                literal = query[start + len(marker):].strip(' \t\r\n:;-–—"\'')
                if literal:
                    return cls._normalize_ocr_literal(literal), True
        for marker in cls.OCR_INTENT_MARKERS:
            start = low.find(marker)
            if start >= 0:
                literal = query[start + len(marker):].strip(' \t\r\n:;-–—"\'')
                if literal:
                    return cls._normalize_ocr_literal(literal), True
                return cls._normalize_ocr_literal(query), True
        return cls._normalize_ocr_literal(query), False

    @staticmethod
    def _normalize_ocr_literal(text: str) -> str:
        # Alias expansion is handled by db.search_ocr so both the full name and
        # acronym remain searchable with equal concept weight.
        return text.strip()

    @staticmethod
    def _selected_signals(signal_sources):
        # None preserves the old API behavior: use every enabled auxiliary signal.
        if signal_sources is None:
            return {'ocr','object'}
        return {str(source).strip().lower() for source in signal_sources}

    def _add_signal_results(self,fused,query,english,signal_sources=None):
        profile={"ocr_boosted":False,"has_object":False}
        selected=self._selected_signals(signal_sources)
        if settings.enable_ocr_search and 'ocr' in selected:
            ocr_query, explicit_ocr = self._prepare_ocr_query(query)
            ocr_rows = search_ocr(ocr_query,settings.signal_candidate_k)
            max_matched = max((int(row.get('matched_terms') or 0) for row in ocr_rows), default=0)
            query_terms = set(
                token.lower() for token in re.findall(r"[^\W_]+", ocr_query, flags=re.UNICODE)
                if len(token) >= 3
            )
            concept_count=max(
                (int(row.get('query_terms') or 0) for row in ocr_rows),
                default=len(query_terms),
            )
            top_coverage = max_matched / max(concept_count, 1)
            # A pasted/lowercase title may not contain an explicit cue such as
            # "dòng chữ". A high OCR term-coverage is itself strong evidence
            # that this is a literal-text query, independent of letter case.
            strong_ocr_match = max_matched >= 4 and top_coverage >= 0.65
            boosted_ocr = explicit_ocr or strong_ocr_match
            profile['ocr_boosted']=boosted_ocr
            ocr_weight = 5.0 if boosted_ocr else 1.0
            for rank,row in enumerate(ocr_rows,1):
                key=(row['video_id'],int(row['keyframe_n']))
                item=fused.setdefault(key,self._empty_fused_item())
                # Text/image retrieval uses up to four SigLIP query variants.
                # Give explicit OCR queries comparable aggregate weight so a
                # strong literal-text match is not buried by visual-only hits.
                # Weaker OCR rows are discounted by their term coverage so a
                # long unrelated page cannot ride along on a few common words.
                coverage = ((int(row.get('matched_terms') or 0) / max_matched)
                            if boosted_ocr and max_matched else 1.0)
                strength = 0.25 + 0.75 * coverage if boosted_ocr else 1.0
                contribution=(ocr_weight*strength)/(60+rank)
                item['ocr_rrf']+=contribution
                item['rrf']+=contribution; item['sources'].add('ocr')
                if len(item['matches'])<3: item['matches'].append(row['text_content'])

        if settings.enable_object_search and 'object' in selected:
            aliases={'người':'person','xe':'car','ô tô':'car','oto':'car','xe máy':'motorcycle',
                     'xemay':'motorcycle','mô tô':'motorcycle','xe buýt':'bus','chó':'dog','mèo':'cat',
                     'chai':'bottle','ghế':'chair','điện thoại':'cell phone','laptop':'laptop'}
            low=query.lower(); english_low=english.lower()
            terms=[label for label in self.COCO_LABELS if label in english_low]
            terms += [x.strip('.,!?;:') for x in english_low.split()]
            terms += [v for k,v in aliases.items() if k in low]
            object_rows=search_objects(terms,settings.signal_candidate_k)
            profile['has_object']=bool(object_rows)
            for rank,row in enumerate(object_rows,1):
                key=(row['video_id'],int(row['keyframe_n']))
                item=fused.setdefault(key,self._empty_fused_item())
                contribution=1.0/(60+rank)
                item['object_rrf']+=contribution
                item['rrf']+=contribution; item['sources'].add('object')
                label=f"object: {row['class_name']} ({float(row['confidence']):.2f})"
                if label not in item['matches'] and len(item['matches'])<3: item['matches'].append(label)
        return profile

    @staticmethod
    def _rerank(fused,profile):
        """Normalize heterogeneous scores, then apply query-aware weights."""
        if not fused:
            return
        clip_values=[item['clip_score'] for item in fused.values() if 'siglip2' in item['sources']]
        clip_min=min(clip_values,default=0.0); clip_max=max(clip_values,default=0.0)
        component_max={name:max((item[name] for item in fused.values()),default=0.0)
                       for name in ('clip_rrf','ocr_rrf','object_rrf')}
        if profile.get('ocr_boosted'):
            weights={'clip':0.25,'clip_rrf':0.10,'ocr_rrf':0.60,'object_rrf':0.05}
        elif profile.get('has_object'):
            # Object detections are supporting evidence only: generic classes
            # such as person/boat/tv are too frequent to lead the ranking.
            weights={'clip':0.62,'clip_rrf':0.23,'ocr_rrf':0.05,'object_rrf':0.10}
        else:
            weights={'clip':0.65,'clip_rrf':0.25,'ocr_rrf':0.05,'object_rrf':0.05}
        for item in fused.values():
            if 'siglip2' not in item['sources']:
                clip_norm=0.0
            elif clip_max-clip_min > 1e-9:
                clip_norm=(item['clip_score']-clip_min)/(clip_max-clip_min)
            else:
                clip_norm=1.0
            item['clip_norm']=clip_norm
            item['final_score']=weights['clip']*clip_norm
            for name in ('clip_rrf','ocr_rrf','object_rrf'):
                maximum=component_max[name]
                normalized=item[name]/maximum if maximum > 0 else 0.0
                item[f'{name}_norm']=normalized
                item['final_score']+=weights[name]*normalized

    @staticmethod
    def _materialize(fused,top_k):
        candidate_limit=min(len(fused),max(top_k*8,500))
        ordered=sorted(
            fused.items(),key=lambda x:(x[1]['final_score'],x[1]['rrf']),reverse=True
        )[:candidate_limit]
        keys=[x[0] for x in ordered]
        metadata=fetch_metadata(keys)
        output=[]
        selected_frames=defaultdict(list)
        for key,info in ordered:
            row=metadata.get(key)
            if not row:
                continue
            fps=float(row.get('fps') or 25.0)
            frame_idx=int(row['frame_idx'])
            radius=max(1,int(SearchEngine.TEMPORAL_DEDUP_SECONDS*fps))
            if any(abs(frame_idx-old_frame)<=radius for old_frame in selected_frames[row['video_id']]):
                continue
            selected_frames[row['video_id']].append(frame_idx)
            rank=len(output)+1
            output.append({"rank":rank,"score":float(info['final_score']),
                           "rrf_score":float(info['rrf']),"clip_score":info['clip_score'],
                           "matched_sources":"+".join(sorted(info['sources'])),
                           "matched_text":" | ".join(info['matches']),**row})
            if len(output)>=top_k:
                break
        return output

    def search_many_by_text(self,queries: list[str],top_k: int = 100,
                            signal_sources=None) -> list[list[dict]]:
        """Batch text encoding/Milvus search for TRAKE instead of one model call per event."""
        english_queries=[self.translator.to_english(query) for query in queries]
        flat_variants=[]; owners=[]
        for query_index,(query,english) in enumerate(zip(queries,english_queries)):
            variants=self._variants(query,english)
            flat_variants.extend(variants)
            owners.extend([query_index]*len(variants))
        vectors=self.encoder.encode(flat_variants)
        fused_by_query=[{} for _ in queries]
        result_sets=self.client.search(
            collection_name=settings.collection,
            data=vectors.tolist(), limit=max(150, top_k),
            output_fields=["video_id", "keyframe_n"],
            search_params={"metric_type":"COSINE", "params":{"nprobe":64}},
        )
        for owner,result_set in zip(owners,result_sets):
            fused=fused_by_query[owner]
            for rank, hit in enumerate(result_set, 1):
                key=(hit["entity"]["video_id"],int(hit["entity"]["keyframe_n"]))
                score = 1.0 / (60 + rank)  # reciprocal-rank fusion
                item=fused.setdefault(key,self._empty_fused_item())
                item["clip_rrf"] += score
                item["rrf"] += score
                item['clip_score']=max(item['clip_score'],float(hit['distance']))
                item['sources'].add('siglip2')
        for fused,query,english in zip(fused_by_query,queries,english_queries):
            profile=self._add_signal_results(fused,query,english,signal_sources)
            self._rerank(fused,profile)
        return [self._materialize(fused,top_k) for fused in fused_by_query]

    def search_sequence(self, events: list[str], top_videos: int = 10,
                        signal_sources=None) -> list[dict]:
        original_events=[e.strip() for e in events if e.strip()]
        english_events=[self.translator.to_english(e) for e in original_events]
        per_event=self.search_many_by_text(original_events,300,signal_sources)
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
                    if getattr(settings,'auto_frame_refine',False) and i < settings.trake_refine_top_n
                    else {'frame_idx':hit['frame_idx'],'refined':False})
                refined.append({**hit,**rr})
            output.append({'rank':i+1,**x,'hits':refined})
        return output

from __future__ import annotations
import re
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
    def search_kis(self, query, top_k=20, signal_sources=None):
        rows=self._adapt(self.engine.search_by_text(query,top_k,signal_sources))
        for r in rows:
            r['coarse_frame_idx']=r['frame_idx']
        if getattr(settings,'auto_frame_refine',False):
            refine_query=self.engine.translator.to_english(query)
            for r in rows[:settings.refine_top_n]:
                self._refine_row(r,refine_query)
        return rows
    @staticmethod
    def _answer_from_ocr(question,text):
        """Extract simple location answers from reliable OCR context."""
        if not text:
            return None
        q=question.casefold()
        location_question=any(cue in q for cue in (
            'ở đâu','đến đâu','về đâu','tại đâu','nơi nào','địa điểm nào',
            'where',
        ))
        if not location_question:
            return None
        stop_words=(
            'ghép','cho','để','khi','và','đang','sau','trước','nhằm','giây',
            'tin','cùng','được','thực hiện','diễn ra',
        )
        # OCR frequently drops Vietnamese tone marks (VỀ -> VÊ/VE,
        # ĐẾN -> ĐEN/DEN). Accept those variants but return the original text.
        preposition_patterns=(
            r'v[ềêe]',
            r'(?:đ|d)[ếêe]n',
            r't[ạa]i',
            r'[ởo]',
        )
        for preposition_pattern in preposition_patterns:
            pattern=rf'\b(?:{preposition_pattern})\s+([^,.;|\n]{{1,80}})'
            for match in re.finditer(pattern,text,flags=re.IGNORECASE):
                phrase=match.group(1).strip()
                stop_pattern=r'\b(?:'+"|".join(re.escape(word) for word in stop_words)+r')\b'
                phrase=re.split(stop_pattern,phrase,maxsplit=1,flags=re.IGNORECASE)[0].strip()
                words=re.findall(r"[^\W_]+(?:[-'][^\W_]+)?",phrase,flags=re.UNICODE)
                if words:
                    return ' '.join(words[:5])[:100]
        return None

    def search_qa(self, event_description, question, top_k=20, signal_sources=None):
        # The event description locates the frame. Appending the question often
        # dilutes retrieval with generic words such as color/how many/where.
        query=event_description.strip() or question.strip()
        rows=self.search_kis(query,min(100,max(top_k*3,30)),signal_sources); output=[]
        translated_question=self.engine.translator.to_english(question)
        for r in rows:
            if not r.get('image_path'): continue
            answer=(self._answer_from_ocr(question,r.get('matched_text',''))
                    if 'ocr' in r.get('matched_sources','') else None)
            if not answer:
                try: answer=self.vqa.answer(r['image_path'],translated_question)
                except Exception:
                    continue
            if not answer.strip():
                continue
            output.append({**r,'answer':answer.strip()[:100]})
            if len(output)>=top_k: break
        return output
    def search_trake(self,events,top_videos=10,signal_sources=None):
        rows=self.engine.search_sequence(events,top_videos,signal_sources)
        for row in rows:
            adapted=[]
            for hit in row.get('hits',[]):
                image=(hit.get('refined_image_path') or get_image(
                    hit['video_id'],hit['keyframe_file'],hit.get('image_relpath','')
                ))
                adapted.append({**hit,
                    'coarse_frame_idx':hit.get('coarse_frame_idx',hit['frame_idx']),
                    'frame_id':hit['frame_idx'],'image_path':image,
                })
            row['hits']=adapted
        return rows

    @staticmethod
    def _rank_index(rows, rank):
        if not rows:
            raise ValueError('Chưa có kết quả để refine.')
        try:
            index=int(rank)-1
        except (TypeError,ValueError):
            raise ValueError('Rank refine phải là số nguyên.')
        if index < 0 or index >= len(rows):
            raise ValueError(f'Rank phải nằm trong khoảng 1-{len(rows)}.')
        return index

    @staticmethod
    def _copy_results(rows):
        copied=[]
        for row in rows:
            item=dict(row)
            if 'hits' in item:
                item['hits']=[dict(hit) for hit in item.get('hits',[])]
            copied.append(item)
        return copied

    @staticmethod
    def _reset_ranks(rows):
        for index,row in enumerate(rows,1):
            row['rank']=index
        return rows

    def promote_result(self,rows,rank):
        index=self._rank_index(rows,rank)
        updated=self._copy_results(rows)
        selected=updated.pop(index)
        updated.insert(0,selected)
        self._reset_ranks(updated)
        return updated,f"Đã đưa {selected['video_id']} từ rank {index+1} lên rank 1."

    def delete_result(self,rows,rank):
        index=self._rank_index(rows,rank)
        updated=self._copy_results(rows)
        selected=updated.pop(index)
        self._reset_ranks(updated)
        return updated,f"Đã xóa {selected['video_id']} ở rank {index+1} khỏi file CSV."

    def _refine_row(self,row,query):
        coarse=int(row.get('coarse_frame_idx',row['frame_idx']))
        row.setdefault('coarse_frame_idx',coarse)
        result=self.engine.refiner.refine(
            row['video_id'],coarse,row.get('fps') or 0,query
        )
        row.update(result)
        row['frame_id']=row['frame_idx']
        if row.get('refined_image_path'):
            row['image_path']=row['refined_image_path']
        return result

    def refine_kis_result(self,rows,rank,query):
        index=self._rank_index(rows,rank)
        updated=[dict(row) for row in rows]
        refine_query=self.engine.translator.to_english(query)
        result=self._refine_row(updated[index],refine_query)
        if not result.get('refined'):
            return updated,result.get('warning','Không refine được frame đã chọn.')
        row=updated[index]
        return updated,(
            f"Đã refine KIS rank {index+1}: {row['video_id']} "
            f"{row['coarse_frame_idx']} → {row['frame_idx']}"
        )

    def refine_qa_result(self,rows,rank,event_description,question):
        index=self._rank_index(rows,rank)
        updated=[dict(row) for row in rows]
        query=event_description.strip() or question.strip()
        result=self._refine_row(updated[index],self.engine.translator.to_english(query))
        row=updated[index]
        if result.get('refined') and row.get('image_path'):
            try:
                answer=(self._answer_from_ocr(question,row.get('matched_text',''))
                        if 'ocr' in row.get('matched_sources','') else None)
                if not answer:
                    answer=self.vqa.answer(
                        row['image_path'],self.engine.translator.to_english(question)
                    ).strip()
                if answer:
                    row['answer']=answer[:100]
            except Exception as exc:
                return updated,f"Frame đã refine nhưng QA không chạy lại được: {exc}"
        if not result.get('refined'):
            return updated,result.get('warning','Không refine được frame đã chọn.')
        return updated,(
            f"Đã refine QA rank {index+1}: {row['video_id']} "
            f"{row['coarse_frame_idx']} → {row['frame_idx']}; answer: {row['answer']}"
        )

    def refine_trake_result(self,rows,rank,events):
        index=self._rank_index(rows,rank)
        clean_events=[event.strip() for event in events if event.strip()]
        updated=[]
        for row in rows:
            copy={**row,'hits':[dict(hit) for hit in row.get('hits',[])]}
            updated.append(copy)
        selected=updated[index]
        if len(clean_events) != len(selected['hits']):
            raise ValueError(
                f"TRAKE có {len(clean_events)} events nhưng kết quả có "
                f"{len(selected['hits'])} frame."
            )
        original=[int(hit.get('coarse_frame_idx',hit['frame_idx'])) for hit in selected['hits']]
        proposals=[]
        for event,hit,coarse in zip(clean_events,selected['hits'],original):
            hit.setdefault('coarse_frame_idx',coarse)
            result=self.engine.refiner.refine(
                selected['video_id'],coarse,hit.get('fps') or 0,
                self.engine.translator.to_english(event),
            )
            proposal={**hit,**result,'frame_id':result.get('frame_idx',coarse)}
            if result.get('refined_image_path'):
                proposal['image_path']=result['refined_image_path']
            proposals.append(proposal)
        if not any(hit.get('refined') for hit in proposals):
            warning=next((hit.get('warning') for hit in proposals if hit.get('warning')),None)
            return updated,warning or 'Không refine được chuỗi TRAKE đã chọn.'
        frame_ids=[int(hit['frame_idx']) for hit in proposals]
        if any(a >= b for a,b in zip(frame_ids,frame_ids[1:])):
            return updated,(
                'Không áp dụng refine vì các frame mới không còn tăng dần. '
                'Kết quả TRAKE cũ được giữ nguyên.'
            )
        selected['hits']=proposals
        return updated,(
            f"Đã refine TRAKE rank {index+1}: {selected['video_id']} · "
            + ', '.join(f'{a}→{b}' for a,b in zip(original,frame_ids))
        )

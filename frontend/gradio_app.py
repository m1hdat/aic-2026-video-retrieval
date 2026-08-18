from __future__ import annotations
import base64
import html
import mimetypes
from functools import lru_cache
import gradio as gr
from frontend.retrieval_service import RetrievalService
from pathlib import Path
from src.settings import ROOT, settings
from src.submission import write_submission

service=RetrievalService()
CSS="""
.container{max-width:1500px!important}.result-card{border-radius:14px} footer{display:none!important}
.trake-card{border:1px solid #3f3f46;border-radius:12px;padding:14px;margin:12px 0;background:#18181b}
.trake-title{font-weight:700;margin-bottom:12px;color:#fafafa}
.trake-sequence{display:flex;align-items:center;gap:10px;overflow-x:auto;padding-bottom:6px}
.trake-event{min-width:210px;max-width:260px}
.trake-event img{width:100%;height:145px;object-fit:cover;border-radius:9px;border:1px solid #52525b}
.trake-caption{font-size:13px;margin-top:6px;color:#e4e4e7;text-align:center}
.trake-arrow{font-size:26px;color:#f97316;flex:0 0 auto}
.trake-missing{height:145px;border:1px dashed #71717a;border-radius:9px;display:flex;align-items:center;justify-content:center;color:#a1a1aa}
"""

@lru_cache(maxsize=5000)
def _image_data_uri(path):
    if not path:
        return None
    try:
        raw=Path(path).read_bytes()
    except OSError:
        return None
    mime=mimetypes.guess_type(str(path))[0] or 'image/jpeg'
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

def _selected_rank(evt: gr.SelectData):
    index=evt.index[0] if isinstance(evt.index,(tuple,list)) else evt.index
    return int(index)+1

def _kis_view(rows):
    gallery=[(r['image_path'],f"#{r['rank']} · {r['video_id']} · frame {r['frame_id']} · {r['matched_sources']}") for r in rows if r['image_path']]
    table=[[r['rank'],r['video_id'],r['frame_id'],r['keyframe_id'],r['matched_sources'],r['matched_text'],round(r['score'],5)] for r in rows]
    return gallery,table

def _qa_view(rows):
    gallery=[(r['image_path'],f"#{r['rank']} · {r['video_id']} · frame {r['frame_id']} · answer: {r['answer']}") for r in rows if r['image_path']]
    table=[[r['rank'],r['video_id'],r['frame_id'],r['answer'],round(r['score'],5)] for r in rows]
    return gallery,table

def _trake_view(rows):
    cards=[]
    table=[]
    for row in rows:
        table.append([row['rank'],row['video_id'],round(row['score'],5),
                      ', '.join(str(h['frame_idx']) for h in row['hits'])])
        events=[]
        for event_index,hit in enumerate(row['hits'],1):
            uri=_image_data_uri(hit.get('image_path'))
            image=(f'<img src="{uri}" alt="Event {event_index}">' if uri else
                   '<div class="trake-missing">Không tìm thấy keyframe</div>')
            events.append(
                '<div class="trake-event">'
                f'{image}<div class="trake-caption">Event {event_index} · '
                f'frame {int(hit["frame_idx"])}</div></div>'
            )
        sequence='<div class="trake-arrow">→</div>'.join(events)
        cards.append(
            '<div class="trake-card">'
            f'<div class="trake-title">Rank {int(row["rank"])} · '
            f'{html.escape(str(row["video_id"]))} · score {float(row["score"]):.5f}</div>'
            f'<div class="trake-sequence">{sequence}</div></div>'
        )
    content=''.join(cards) if cards else '<p>Chưa có chuỗi kết quả.</p>'
    return content,table

def kis(query, top_k, signals):
    rows=service.search_kis(query,int(top_k),signals)
    gallery,table=_kis_view(rows)
    return gallery,table,rows

def qa(event,question,top_k,signals):
    rows=service.search_qa(event,question,int(top_k),signals)
    gallery,table=_qa_view(rows)
    return gallery,table,rows

def trake(events,top_videos,signals):
    ev=[x.strip() for x in events.splitlines() if x.strip()]
    rows=service.search_trake(ev,int(top_videos),signals)
    gallery,table=_trake_view(rows)
    return gallery,table,rows

def refine_kis(rows,rank,query):
    try:
        updated,status=service.refine_kis_result(rows or [],rank,query)
        gallery,table=_kis_view(updated)
        return gallery,table,updated,status
    except Exception as exc:
        return gr.skip(),gr.skip(),rows,f"Refine KIS lỗi: {exc}"

def refine_qa(rows,rank,event,question):
    try:
        updated,status=service.refine_qa_result(rows or [],rank,event,question)
        gallery,table=_qa_view(updated)
        return gallery,table,updated,status
    except Exception as exc:
        return gr.skip(),gr.skip(),rows,f"Refine QA lỗi: {exc}"

def refine_trake(rows,rank,events):
    try:
        ev=[x.strip() for x in events.splitlines() if x.strip()]
        updated,status=service.refine_trake_result(rows or [],rank,ev)
        gallery,table=_trake_view(updated)
        return gallery,table,updated,status
    except Exception as exc:
        return gr.skip(),gr.skip(),rows,f"Refine TRAKE lỗi: {exc}"

def _curate(rows,rank,action,view,label):
    try:
        if action=='promote':
            updated,status=service.promote_result(rows or [],rank)
        else:
            updated,status=service.delete_result(rows or [],rank)
        gallery,table=view(updated)
        return gallery,table,updated,status
    except Exception as exc:
        return gr.skip(),gr.skip(),rows,f"{label} lỗi: {exc}"

def promote_kis(rows,rank): return _curate(rows,rank,'promote',_kis_view,'KIS')
def delete_kis(rows,rank): return _curate(rows,rank,'delete',_kis_view,'KIS')
def promote_qa(rows,rank): return _curate(rows,rank,'promote',_qa_view,'QA')
def delete_qa(rows,rank): return _curate(rows,rank,'delete',_qa_view,'QA')
def promote_trake(rows,rank): return _curate(rows,rank,'promote',_trake_view,'TRAKE')
def delete_trake(rows,rank): return _curate(rows,rank,'delete',_trake_view,'TRAKE')

def export(kind,rows): return write_submission(kind,rows or [])

with gr.Blocks(title='AIC 2026 Video Retrieval',css=CSS) as demo:
    gr.Markdown('# AIC 2026 · Video Retrieval\nSigLIP2 + Milvus + PostgreSQL')
    with gr.Tabs():
      with gr.Tab('KIS'):
        q=gr.Textbox(label='Mô tả cảnh cần tìm',lines=2); k=gr.Slider(5,100,20,step=5,label='Top K')
        ksignals=gr.CheckboxGroup(['OCR','Object'],value=['OCR','Object'],label='Nguồn hỗ trợ (SigLIP luôn bật)')
        b=gr.Button('Tìm kiếm',variant='primary')
        ks=gr.State([]); g=gr.Gallery(label='Keyframes',columns=5,height=520); t=gr.Dataframe(headers=['rank','video_id','frame_idx','keyframe','nguồn khớp','OCR/Object khớp','score'],interactive=False)
        with gr.Row():
          kr=gr.Number(value=1,precision=0,minimum=1,label='Rank đã chọn'); kb=gr.Button('Refine frame')
          kup=gr.Button('Đưa lên #1'); kdel=gr.Button('Xóa khỏi CSV')
        kstatus=gr.Markdown('Chọn một dòng kết quả, rồi refine, đưa lên đầu hoặc xóa.')
        kd=gr.DownloadButton('Tải CSV nộp KIS'); b.click(kis,[q,k,ksignals],[g,t,ks]); kd.click(lambda x:export('KIS',x),ks,kd)
        t.select(_selected_rank,outputs=kr)
        kb.click(refine_kis,[ks,kr,q],[g,t,ks,kstatus])
        kup.click(promote_kis,[ks,kr],[g,t,ks,kstatus])
        kdel.click(delete_kis,[ks,kr],[g,t,ks,kstatus])
      with gr.Tab('QA'):
        e=gr.Textbox(label='Mô tả sự kiện'); question=gr.Textbox(label='Câu hỏi'); qk=gr.Slider(5,100,20,step=5,label='Top K')
        qsignals=gr.CheckboxGroup(['OCR','Object'],value=['OCR','Object'],label='Nguồn hỗ trợ (SigLIP luôn bật)')
        qb=gr.Button('Tìm kiếm',variant='primary')
        qs=gr.State([]); qg=gr.Gallery(label='Keyframes',columns=5,height=520); qt=gr.Dataframe(headers=['rank','video_id','frame_idx','answer','score'],interactive=False)
        with gr.Row():
          qr=gr.Number(value=1,precision=0,minimum=1,label='Rank đã chọn'); qrb=gr.Button('Refine + chạy lại QA')
          qup=gr.Button('Đưa lên #1'); qdel=gr.Button('Xóa khỏi CSV')
        qstatus=gr.Markdown('Chọn một dòng kết quả, rồi refine, đưa lên đầu hoặc xóa.')
        qd=gr.DownloadButton('Tải CSV nộp QA'); qb.click(qa,[e,question,qk,qsignals],[qg,qt,qs]); qd.click(lambda x:export('QA',x),qs,qd)
        qt.select(_selected_rank,outputs=qr)
        qrb.click(refine_qa,[qs,qr,e,question],[qg,qt,qs,qstatus])
        qup.click(promote_qa,[qs,qr],[qg,qt,qs,qstatus])
        qdel.click(delete_qa,[qs,qr],[qg,qt,qs,qstatus])
      with gr.Tab('TRAKE'):
        events=gr.Textbox(label='Mỗi dòng là một sự kiện, đúng thứ tự thời gian',lines=6); tv=gr.Slider(1,30,10,step=1,label='Top videos')
        tsignals=gr.CheckboxGroup(['OCR','Object'],value=['OCR','Object'],label='Nguồn hỗ trợ (SigLIP luôn bật)')
        tb=gr.Button('Tìm chuỗi sự kiện',variant='primary')
        ts=gr.State([]); tg=gr.HTML('<p>Chưa có chuỗi kết quả.</p>',label='Chuỗi ảnh TRAKE')
        tt=gr.Dataframe(headers=['rank','video_id','score','frame_idx theo thứ tự'],interactive=False); td=gr.DownloadButton('Tải CSV nộp TRAKE')
        tb.click(trake,[events,tv,tsignals],[tg,tt,ts]); td.click(lambda x:export('TRAKE',x),ts,td)
        with gr.Row():
          tr=gr.Number(value=1,precision=0,minimum=1,label='Rank đã chọn'); trb=gr.Button('Refine toàn bộ chuỗi')
          tup=gr.Button('Đưa lên #1'); tdel=gr.Button('Xóa khỏi CSV')
        trstatus=gr.Markdown('Chọn một dòng kết quả, rồi refine, đưa lên đầu hoặc xóa.')
        tt.select(_selected_rank,outputs=tr)
        trb.click(refine_trake,[ts,tr,events],[tg,tt,ts,trstatus])
        tup.click(promote_trake,[ts,tr],[tg,tt,ts,trstatus])
        tdel.click(delete_trake,[ts,tr],[tg,tt,ts,trstatus])

if __name__=='__main__':
    local_roots=[Path(x.strip().strip('"')) for x in settings.keyframe_roots.split(';') if x.strip()]
    allowed=[str(ROOT/'data/cache')]+[str(p.resolve()) for p in local_roots if p.exists()]
    demo.launch(server_name=settings.host,server_port=settings.port,allowed_paths=allowed)

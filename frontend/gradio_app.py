from __future__ import annotations
import gradio as gr
from frontend.retrieval_service import RetrievalService
from pathlib import Path
from src.settings import ROOT, settings
from src.submission import write_submission

service=RetrievalService()
CSS=""".container{max-width:1500px!important}.result-card{border-radius:14px} footer{display:none!important}"""

def kis(query, top_k):
    rows=service.search_kis(query,int(top_k))
    gallery=[(r['image_path'],f"#{r['rank']} · {r['video_id']} · frame {r['frame_id']} · {r['score']:.4f}") for r in rows if r['image_path']]
    table=[[r['rank'],r['video_id'],r['frame_id'],r['keyframe_id'],round(r['score'],5)] for r in rows]
    return gallery,table,rows

def qa(event,question,top_k):
    rows=service.search_qa(event,question,int(top_k))
    gallery=[(r['image_path'],f"#{r['rank']} · {r['video_id']} · frame {r['frame_id']} · answer: {r['answer']}") for r in rows if r['image_path']]
    table=[[r['rank'],r['video_id'],r['frame_id'],r['answer'],round(r['score'],5)] for r in rows]
    return gallery,table,rows

def trake(events,top_videos):
    ev=[x.strip() for x in events.splitlines() if x.strip()]
    rows=service.search_trake(ev,int(top_videos))
    table=[[r['rank'],r['video_id'],round(r['score'],5),', '.join(str(h['frame_idx']) for h in r['hits'])] for r in rows]
    return table,rows

def export(kind,rows): return write_submission(kind,rows or [])

with gr.Blocks(title='AIC 2026 Video Retrieval',css=CSS) as demo:
    gr.Markdown('# AIC 2026 · Video Retrieval\nCLIP + Milvus + PostgreSQL')
    with gr.Tabs():
      with gr.Tab('KIS'):
        q=gr.Textbox(label='Mô tả cảnh cần tìm',lines=2); k=gr.Slider(5,100,20,step=5,label='Top K'); b=gr.Button('Tìm kiếm',variant='primary')
        ks=gr.State([]); g=gr.Gallery(label='Keyframes',columns=5,height=520); t=gr.Dataframe(headers=['rank','video_id','frame_idx','keyframe','score'],interactive=False)
        kd=gr.DownloadButton('Tải CSV nộp KIS'); b.click(kis,[q,k],[g,t,ks]); kd.click(lambda x:export('KIS',x),ks,kd)
      with gr.Tab('QA'):
        e=gr.Textbox(label='Mô tả sự kiện'); question=gr.Textbox(label='Câu hỏi'); qk=gr.Slider(5,100,20,step=5,label='Top K'); qb=gr.Button('Tìm kiếm',variant='primary')
        qs=gr.State([]); qg=gr.Gallery(label='Keyframes',columns=5,height=520); qt=gr.Dataframe(headers=['rank','video_id','frame_idx','answer','score'],interactive=False)
        qd=gr.DownloadButton('Tải CSV nộp QA'); qb.click(qa,[e,question,qk],[qg,qt,qs]); qd.click(lambda x:export('QA',x),qs,qd)
      with gr.Tab('TRAKE'):
        events=gr.Textbox(label='Mỗi dòng là một sự kiện, đúng thứ tự thời gian',lines=6); tv=gr.Slider(1,30,10,step=1,label='Top videos'); tb=gr.Button('Tìm chuỗi sự kiện',variant='primary')
        ts=gr.State([]); tt=gr.Dataframe(headers=['rank','video_id','score','frame_idx theo thứ tự'],interactive=False); td=gr.DownloadButton('Tải CSV nộp TRAKE')
        tb.click(trake,[events,tv],[tt,ts]); td.click(lambda x:export('TRAKE',x),ts,td)

if __name__=='__main__':
    local_roots=[Path(x.strip().strip('"')) for x in settings.keyframe_roots.split(';') if x.strip()]
    allowed=[str(ROOT/'data/cache')]+[str(p.resolve()) for p in local_roots if p.exists()]
    demo.launch(server_name=settings.host,server_port=settings.port,allowed_paths=allowed)

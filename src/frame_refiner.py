from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from .settings import ROOT
from .clip_encoder import TextEncoder
from .settings import settings
from .video_resolver import resolve_video

class FrameRefiner:
    def __init__(self, encoder: TextEncoder): self.encoder=encoder

    def refine(self, video_id: str, coarse_frame: int, fps_hint: float, query: str) -> dict:
        """Score every native frame near a 1-FPS hit and return the exact frame index."""
        if not settings.enable_frame_refine: return {'frame_idx':coarse_frame,'refined':False}
        path=resolve_video(video_id)
        if not path: return {'frame_idx':coarse_frame,'refined':False,'warning':'Không tìm thấy video gốc'}
        cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS) or fps_hint or 25.0
        radius=max(1,int(settings.refine_seconds*fps)); start=max(0,coarse_frame-radius); end=coarse_frame+radius
        cap.set(cv2.CAP_PROP_POS_FRAMES,start); frames=[]; ids=[]; idx=start
        while idx<=end:
            ok,bgr=cap.read()
            if not ok: break
            if (idx-start)%max(1,settings.refine_stride)==0:
                frames.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))); ids.append(idx)
            idx+=1
        cap.release()
        if not frames: return {'frame_idx':coarse_frame,'refined':False}
        q=self.encoder.encode([query])[0]; scores=[]
        for s in range(0,len(frames),64): scores.extend((self.encoder.encode_images(frames[s:s+64])@q).tolist())
        best=int(np.argmax(scores))
        out=ROOT/'data'/'cache'/'refined'/video_id/f'{ids[best]:09d}.jpg'; out.parent.mkdir(parents=True,exist_ok=True)
        frames[best].save(out,quality=92)
        return {'frame_idx':ids[best],'refined':True,'refine_score':float(scores[best]),'refined_image_path':str(out)}

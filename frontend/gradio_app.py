from __future__ import annotations

from pathlib import Path

import gradio as gr

from frontend.retrieval_service import RetrievalService
from src.settings import ROOT, settings
from src.submission import write_submission


service = RetrievalService()

CSS = """
.container {
    max-width: 1500px !important;
}

.result-card {
    border-radius: 14px;
}

footer {
    display: none !important;
}
"""


# =========================================================
# KIS
# =========================================================

def kis(query, top_k):
    rows = service.search_kis(query, int(top_k))

    gallery = [
        (
            row["image_path"],
            (
                f"#{row['rank']} · {row['video_id']} · "
                f"frame {row['frame_id']} · "
                f"{row['matched_sources']}"
            ),
        )
        for row in rows
        if row.get("image_path")
    ]

    table = [
        [
            row["rank"],
            row["video_id"],
            row["frame_id"],
            row["keyframe_id"],
            row["matched_sources"],
            row["matched_text"],
            round(row["score"], 5),
        ]
        for row in rows
    ]

    return gallery, table, rows


# =========================================================
# QA
# =========================================================

def qa(event, question, top_k):
    rows = service.search_qa(event, question, int(top_k))

    gallery = [
        (
            row["image_path"],
            (
                f"#{row['rank']} · {row['video_id']} · "
                f"frame {row['frame_id']} · "
                f"answer: {row['answer']}"
            ),
        )
        for row in rows
        if row.get("image_path")
    ]

    table = [
        [
            row["rank"],
            row["video_id"],
            row["frame_id"],
            row["answer"],
            round(row["score"], 5),
        ]
        for row in rows
    ]

    return gallery, table, rows


# =========================================================
# TRAKE
# =========================================================

def trake(events, top_videos):
    event_list = [
        event.strip()
        for event in events.splitlines()
        if event.strip()
    ]

    rows = service.search_trake(event_list, int(top_videos))

    gallery = []

    # Mỗi video có nhiều hit, tương ứng với các sự kiện theo thời gian.
    for video_row in rows:
        video_rank = video_row.get("rank", "")
        video_id = video_row.get("video_id", "")
        hits = video_row.get("hits", [])

        for event_number, hit in enumerate(hits, start=1):
            image_path = hit.get("image_path")

            if not image_path:
                continue

            frame_idx = hit.get(
                "frame_idx",
                hit.get("frame_id", "?"),
            )

            event_text = (
                event_list[event_number - 1]
                if event_number <= len(event_list)
                else f"Sự kiện {event_number}"
            )

            caption = (
                f"Video #{video_rank}: {video_id} · "
                f"Sự kiện {event_number}: {event_text} · "
                f"frame {frame_idx}"
            )

            gallery.append((image_path, caption))

    table = [
        [
            row["rank"],
            row["video_id"],
            round(row["score"], 5),
            ", ".join(
                str(
                    hit.get(
                        "frame_idx",
                        hit.get("frame_id", "?"),
                    )
                )
                for hit in row.get("hits", [])
            ),
        ]
        for row in rows
    ]

    return gallery, table, rows


def export(kind, rows):
    return write_submission(kind, rows or [])


# =========================================================
# GRADIO UI
# =========================================================

with gr.Blocks(
    title="AIC 2026 Video Retrieval",
    css=CSS,
) as demo:

    gr.Markdown(
        "# AIC 2026 · Video Retrieval\n"
        "SigLIP2 + Milvus + PostgreSQL"
    )

    with gr.Tabs():

        # ---------------- KIS ----------------
        with gr.Tab("KIS"):
            q = gr.Textbox(
                label="Mô tả cảnh cần tìm",
                lines=2,
            )

            k = gr.Slider(
                minimum=5,
                maximum=100,
                value=20,
                step=5,
                label="Top K",
            )

            b = gr.Button(
                "Tìm kiếm",
                variant="primary",
            )

            ks = gr.State([])

            g = gr.Gallery(
                label="Keyframes",
                columns=5,
                height=520,
                format="jpeg",
                allow_preview=False,
            )

            t = gr.Dataframe(
                headers=[
                    "rank",
                    "video_id",
                    "frame_idx",
                    "keyframe",
                    "nguồn khớp",
                    "OCR/Object khớp",
                    "score",
                ],
                interactive=False,
            )

            kd = gr.DownloadButton("Tải CSV nộp KIS")

            b.click(
                fn=kis,
                inputs=[q, k],
                outputs=[g, t, ks],
            )

            kd.click(
                fn=lambda rows: export("KIS", rows),
                inputs=ks,
                outputs=kd,
            )

        # ---------------- QA ----------------
        with gr.Tab("QA"):
            e = gr.Textbox(
                label="Mô tả sự kiện",
                lines=2,
            )

            question = gr.Textbox(
                label="Câu hỏi",
                lines=1,
            )

            qk = gr.Slider(
                minimum=5,
                maximum=100,
                value=20,
                step=5,
                label="Top K",
            )

            qb = gr.Button(
                "Tìm kiếm",
                variant="primary",
            )

            qs = gr.State([])

            qg = gr.Gallery(
                label="Keyframes",
                columns=5,
                height=520,
                format="jpeg",
                allow_preview=False,
            )

            qt = gr.Dataframe(
                headers=[
                    "rank",
                    "video_id",
                    "frame_idx",
                    "answer",
                    "score",
                ],
                interactive=False,
            )

            qd = gr.DownloadButton("Tải CSV nộp QA")

            qb.click(
                fn=qa,
                inputs=[e, question, qk],
                outputs=[qg, qt, qs],
            )

            qd.click(
                fn=lambda rows: export("QA", rows),
                inputs=qs,
                outputs=qd,
            )

        # ---------------- TRAKE ----------------
        with gr.Tab("TRAKE"):
            events = gr.Textbox(
                label="Mỗi dòng là một sự kiện, đúng thứ tự thời gian",
                lines=6,
                placeholder=(
                    "Một người bước vào cửa hàng\n"
                    "Người đó lấy một chai nước\n"
                    "Người đó thanh toán tại quầy"
                ),
            )

            tv = gr.Slider(
                minimum=1,
                maximum=30,
                value=10,
                step=1,
                label="Top videos",
            )

            tb = gr.Button(
                "Tìm chuỗi sự kiện",
                variant="primary",
            )

            ts = gr.State([])

            tg = gr.Gallery(
                label="Các keyframe theo thứ tự sự kiện",
                columns=4,
                height=520,
                format="jpeg",
                allow_preview=False,
            )

            tt = gr.Dataframe(
                headers=[
                    "rank",
                    "video_id",
                    "score",
                    "frame_idx theo thứ tự",
                ],
                interactive=False,
            )

            td = gr.DownloadButton("Tải CSV nộp TRAKE")

            tb.click(
                fn=trake,
                inputs=[events, tv],
                outputs=[tg, tt, ts],
            )

            td.click(
                fn=lambda rows: export("TRAKE", rows),
                inputs=ts,
                outputs=td,
            )


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":
    local_roots = [
        Path(path.strip().strip('"'))
        for path in settings.keyframe_roots.split(";")
        if path.strip()
    ]

    allowed = [str(ROOT / "data/cache")] + [
        str(path.resolve())
        for path in local_roots
        if path.exists()
    ]

    demo.launch(
        server_name=settings.host,
        server_port=settings.port,
        allowed_paths=allowed,
    )
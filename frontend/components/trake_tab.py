from __future__ import annotations

from typing import Any

import gradio as gr
import pandas as pd

from frontend.components.answer_queue import build_answer_queue
from frontend.components.shared import normalize_select_index, parse_events, safe_int


def build_trake_tab(
    retrieval_service: Any,
    submission_manager: Any,
) -> None:
    with gr.Tab("3. TRAKE"):
        gr.Markdown(
            "Mỗi dòng là một event. Hệ thống xếp hạng video và đề xuất "
            "một frame cho từng event."
        )

        with gr.Row():
            events_input = gr.Textbox(
                label="Chuỗi sự kiện",
                value=(
                    "Chạy đà\n"
                    "Giậm nhảy\n"
                    "Bay qua xà\n"
                    "Tiếp đất"
                ),
                lines=7,
                scale=4,
            )
            top_videos_slider = gr.Slider(
                label="Top videos",
                minimum=1,
                maximum=50,
                value=10,
                step=1,
                scale=1,
            )

        search_button = gr.Button("Tìm video TRAKE", variant="primary")
        status = gr.Markdown()

        candidates_state = gr.State([])

        candidates_table = gr.Dataframe(
            headers=["rank", "score", "video_id", "frame_ids", "events"],
            label="Video ứng viên — chọn một dòng",
            interactive=False,
            wrap=True,
        )

        def run_search(events_text: str, top_videos: int):
            events = parse_events(events_text)

            if len(events) < 2:
                raise gr.Error(
                    "TRAKE cần ít nhất 2 sự kiện, mỗi sự kiện nằm trên một dòng."
                )

            candidates = retrieval_service.search_trake(
                events=events,
                top_videos=safe_int(top_videos, 10),
            )

            rows = []
            for item in candidates:
                rows.append(
                    [
                        item["rank"],
                        round(float(item["score"]), 4),
                        item["video_id"],
                        ", ".join(str(x) for x in item["frame_ids"]),
                        " → ".join(item["events"]),
                    ]
                )

            table = pd.DataFrame(
                rows,
                columns=["rank", "score", "video_id", "frame_ids", "events"],
            )

            return (
                candidates,
                table,
                f"Tìm thấy **{len(candidates)}** video ứng viên.",
            )

        search_button.click(
            fn=run_search,
            inputs=[events_input, top_videos_slider],
            outputs=[candidates_state, candidates_table, status],
        )

        selected_video_id = gr.Textbox(
            label="video_id đã chọn",
            interactive=False,
        )
        selected_frame_ids = gr.Textbox(
            label="frame_id theo thứ tự event",
            placeholder="Ví dụ: 101, 151, 203, 251",
        )
        selected_score = gr.Number(label="score", interactive=False)
        selected_detail = gr.Markdown("Chưa chọn video ứng viên.")

        def select_candidate(
            candidates: list[dict[str, Any]],
            evt: gr.SelectData,
        ):
            if not candidates:
                return "", "", None, "Chưa có ứng viên TRAKE."

            index = normalize_select_index(evt.index)
            if index is None or not 0 <= index < len(candidates):
                return "", "", None, "Không đọc được dòng đã chọn."

            item = candidates[index]
            frame_text = ", ".join(str(x) for x in item["frame_ids"])

            detail = (
                f'**Đã chọn video:** `{item["video_id"]}`  \n'
                f'Frames đề xuất: `{frame_text}`  \n'
                f'Score: `{item["score"]:.4f}`'
            )

            return (
                item["video_id"],
                frame_text,
                float(item["score"]),
                detail,
            )

        candidates_table.select(
            fn=select_candidate,
            inputs=[candidates_state],
            outputs=[
                selected_video_id,
                selected_frame_ids,
                selected_score,
                selected_detail,
            ],
        )

        queue = build_answer_queue(
            mode="trake",
            submission_manager=submission_manager,
        )

        add_button = gr.Button(
            "Thêm đáp án TRAKE vào Queue",
            variant="primary",
        )

        def add_answer(
            queue_value,
            video_id,
            frame_ids_text,
            score,
        ):
            if not video_id:
                raise gr.Error("Hãy chọn một video TRAKE trước.")

            frame_ids = submission_manager.parse_frame_ids(frame_ids_text)

            if len(frame_ids) < 2:
                raise gr.Error("TRAKE cần ít nhất 2 frame_id.")

            if frame_ids != sorted(frame_ids):
                raise gr.Error(
                    "Các frame_id TRAKE phải tăng dần theo thời gian."
                )

            updated = submission_manager.add(
                mode="trake",
                queue=queue_value,
                answer={
                    "video_id": video_id,
                    "frame_ids": frame_ids,
                    "score": float(score or 0.0),
                },
            )

            return updated, submission_manager.to_dataframe("trake", updated)

        add_button.click(
            fn=add_answer,
            inputs=[
                queue["state"],
                selected_video_id,
                selected_frame_ids,
                selected_score,
            ],
            outputs=[queue["state"], queue["table"]],
        )

from __future__ import annotations

from typing import Any

import gradio as gr

from web.components.answer_queue import build_answer_queue
from web.components.result_browser import build_result_browser
from web.components.shared import results_to_outputs, safe_int


def build_kis_tab(
    retrieval_service: Any,
    submission_manager: Any,
    video_service: Any,
) -> None:
    with gr.Tab("1. Textual KIS"):
        gr.Markdown(
            "Nhập mô tả sự kiện, chọn keyframe và tạo đáp án "
            "`<video_id, frame_id>`."
        )

        with gr.Row():
            query_input = gr.Textbox(
                label="Mô tả truy vấn",
                value="Một người đang mở laptop trong phòng",
                lines=3,
                scale=4,
            )
            top_k_slider = gr.Slider(
                label="Top K",
                minimum=1,
                maximum=100,
                value=20,
                step=1,
                scale=1,
            )

        search_button = gr.Button("Tìm kiếm KIS", variant="primary")
        status = gr.Markdown()

        browser = build_result_browser(
            video_service=video_service,
            label_prefix="KIS",
        )

        def run_search(query: str, top_k: int):
            query = (query or "").strip()
            if not query:
                raise gr.Error("Bạn cần nhập mô tả truy vấn Textual KIS.")

            results = retrieval_service.search_kis(
                query=query,
                top_k=safe_int(top_k, 20),
            )
            gallery, table = results_to_outputs(results)

            return (
                results,
                gallery,
                table,
                f"Tìm thấy **{len(results)}** kết quả.",
            )

        search_button.click(
            fn=run_search,
            inputs=[query_input, top_k_slider],
            outputs=[
                browser["state"],
                browser["gallery"],
                browser["table"],
                status,
            ],
        )

        queue = build_answer_queue(
            mode="kis",
            submission_manager=submission_manager,
        )

        add_button = gr.Button(
            "Thêm kết quả đã chọn vào KIS Queue",
            variant="primary",
        )

        def add_answer(queue_value, video_id, frame_id, score):
            if not video_id:
                raise gr.Error("Hãy chọn một keyframe trước.")

            updated = submission_manager.add(
                mode="kis",
                queue=queue_value,
                answer={
                    "video_id": video_id,
                    "frame_id": int(frame_id),
                    "score": float(score or 0.0),
                },
            )

            return updated, submission_manager.to_dataframe("kis", updated)

        add_button.click(
            fn=add_answer,
            inputs=[
                queue["state"],
                browser["video_id"],
                browser["frame_id"],
                browser["score"],
            ],
            outputs=[queue["state"], queue["table"]],
        )

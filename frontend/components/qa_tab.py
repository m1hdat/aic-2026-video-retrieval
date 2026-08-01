from __future__ import annotations

from typing import Any

import gradio as gr

from frontend.components.answer_queue import build_answer_queue
from frontend.components.result_browser import build_result_browser
from frontend.components.shared import results_to_outputs, safe_int


def build_qa_tab(
    retrieval_service: Any,
    submission_manager: Any,
    video_service: Any,
) -> None:
    with gr.Tab("2. Q&A"):
        gr.Markdown(
            "Truy xuất khoảnh khắc liên quan, sau đó nhập answer để tạo "
            "`<video_id, frame_id, answer>`."
        )

        event_description = gr.Textbox(
            label="Mô tả sự kiện",
            value="Trong video quay cảnh bữa tiệc có một người phụ nữ mặc váy đỏ",
            lines=2,
        )

        with gr.Row():
            question_input = gr.Textbox(
                label="Câu hỏi",
                value="Người phụ nữ đang cầm ly màu gì?",
                lines=2,
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

        search_button = gr.Button("Tìm ứng viên Q&A", variant="primary")
        status = gr.Markdown()

        browser = build_result_browser(
            video_service=video_service,
            label_prefix="Q&A",
        )

        def run_search(event_text: str, question: str, top_k: int):
            event_text = (event_text or "").strip()
            question = (question or "").strip()

            if not event_text and not question:
                raise gr.Error("Hãy nhập mô tả sự kiện hoặc câu hỏi.")

            results = retrieval_service.search_qa(
                event_description=event_text,
                question=question,
                top_k=safe_int(top_k, 20),
            )
            gallery, table = results_to_outputs(results)

            return (
                results,
                gallery,
                table,
                f"Tìm thấy **{len(results)}** ứng viên Q&A.",
            )

        search_button.click(
            fn=run_search,
            inputs=[event_description, question_input, top_k_slider],
            outputs=[
                browser["state"],
                browser["gallery"],
                browser["table"],
                status,
            ],
        )

        answer_input = gr.Textbox(
            label="Answer",
            placeholder="Nhập câu trả lời tiếng Việt hoặc tiếng Anh",
        )

        queue = build_answer_queue(
            mode="qa",
            submission_manager=submission_manager,
        )

        add_button = gr.Button(
            "Thêm đáp án Q&A vào Queue",
            variant="primary",
        )

        def add_answer(
            queue_value,
            video_id,
            frame_id,
            answer,
            score,
        ):
            answer = (answer or "").strip()

            if not video_id:
                raise gr.Error("Hãy chọn một keyframe trước.")
            if not answer:
                raise gr.Error("Q&A cần có answer.")

            updated = submission_manager.add(
                mode="qa",
                queue=queue_value,
                answer={
                    "video_id": video_id,
                    "frame_id": int(frame_id),
                    "answer": answer,
                    "score": float(score or 0.0),
                },
            )

            return updated, submission_manager.to_dataframe("qa", updated)

        add_button.click(
            fn=add_answer,
            inputs=[
                queue["state"],
                browser["video_id"],
                browser["frame_id"],
                answer_input,
                browser["score"],
            ],
            outputs=[queue["state"], queue["table"]],
        )

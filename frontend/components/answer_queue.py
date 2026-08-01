from __future__ import annotations

from typing import Any

import gradio as gr


QUEUE_HEADERS = {
    "kis": ["rank", "video_id", "frame_id", "score"],
    "qa": ["rank", "video_id", "frame_id", "answer", "score"],
    "trake": ["rank", "video_id", "frame_ids", "score"],
}


def build_answer_queue(
    mode: str,
    submission_manager: Any,
) -> dict[str, Any]:
    if mode not in QUEUE_HEADERS:
        raise ValueError(f"Unsupported queue mode: {mode}")

    queue_state = gr.State([])

    queue_table = gr.Dataframe(
        headers=QUEUE_HEADERS[mode],
        label="Answer Queue — tối đa 100 đáp án",
        interactive=False,
        wrap=True,
    )

    with gr.Row():
        rank_input = gr.Number(
            label="Rank cần thao tác",
            value=1,
            precision=0,
        )
        move_up_button = gr.Button("↑ Đưa lên")
        move_down_button = gr.Button("↓ Đưa xuống")
        delete_button = gr.Button("Xóa")
        clear_button = gr.Button("Xóa toàn bộ")

    with gr.Row():
        export_button = gr.Button("Export CSV", variant="primary")
        save_button = gr.Button("Lưu session JSON")
        session_file = gr.File(
            label="Nạp session JSON",
            file_types=[".json"],
        )
        load_button = gr.Button("Load session")

    exported_file = gr.File(label="Submission CSV")
    saved_session_file = gr.File(label="Session JSON")

    def move(queue, rank, direction):
        updated = submission_manager.move(queue, rank, direction)
        return updated, submission_manager.to_dataframe(mode, updated)

    def delete(queue, rank):
        updated = submission_manager.delete(queue, rank)
        return updated, submission_manager.to_dataframe(mode, updated)

    def clear():
        return [], submission_manager.to_dataframe(mode, [])

    def export(queue):
        if not queue:
            raise gr.Error("Answer Queue đang trống.")
        return submission_manager.export_csv(mode, queue)

    def save(queue):
        if not queue:
            raise gr.Error("Answer Queue đang trống.")
        return submission_manager.save_session(mode, queue)

    def load(uploaded_file):
        updated = submission_manager.load_session(
            uploaded_file,
            expected_mode=mode,
        )
        return updated, submission_manager.to_dataframe(mode, updated)

    move_up_button.click(
        fn=lambda queue, rank: move(queue, rank, -1),
        inputs=[queue_state, rank_input],
        outputs=[queue_state, queue_table],
    )

    move_down_button.click(
        fn=lambda queue, rank: move(queue, rank, 1),
        inputs=[queue_state, rank_input],
        outputs=[queue_state, queue_table],
    )

    delete_button.click(
        fn=delete,
        inputs=[queue_state, rank_input],
        outputs=[queue_state, queue_table],
    )

    clear_button.click(
        fn=clear,
        outputs=[queue_state, queue_table],
    )

    export_button.click(
        fn=export,
        inputs=[queue_state],
        outputs=[exported_file],
    )

    save_button.click(
        fn=save,
        inputs=[queue_state],
        outputs=[saved_session_file],
    )

    load_button.click(
        fn=load,
        inputs=[session_file],
        outputs=[queue_state, queue_table],
    )

    return {
        "state": queue_state,
        "table": queue_table,
        "rank_input": rank_input,
        "exported_file": exported_file,
        "saved_session_file": saved_session_file,
    }

from __future__ import annotations

from typing import Any

import gradio as gr

from web.components.shared import RESULT_HEADERS, normalize_select_index


def select_gallery_item(
    results: list[dict[str, Any]],
    video_service: Any,
    evt: gr.SelectData,
):
    if not results:
        return "", None, None, None, "", None, "Chưa có kết quả để chọn."

    index = normalize_select_index(evt.index)
    if index is None or not 0 <= index < len(results):
        return "", None, None, None, "", None, "Không đọc được kết quả đã chọn."

    selected = results[index]
    video_id = str(selected.get("video_id", ""))
    frame_id = int(selected.get("frame_id", 0))
    keyframe_id = str(selected.get("keyframe_id", ""))
    score = float(selected.get("score", 0.0))
    image_path = selected.get("image_path")
    video_path = selected.get("video_path")

    preview_path = video_service.create_preview(video_path, frame_id)

    objects = selected.get("objects", [])
    if isinstance(objects, str):
        objects_text = objects
    else:
        objects_text = ", ".join(objects)

    detail = (
        f"**Đã chọn:** `{video_id}` — frame `{frame_id}`  \n"
        f"Keyframe: `{keyframe_id}`  \n"
        f"Score: `{score:.4f}`  \n"
        f"Timestamp: `{selected.get('timestamp_sec', 'N/A')}` s  \
"
        f"Objects: `{objects_text or 'N/A'}`"
    )

    return (
        video_id,
        preview_path,
        image_path,
        frame_id,
        keyframe_id,
        score,
        detail,
    )


def build_result_browser(video_service: Any, label_prefix: str = "") -> dict[str, Any]:
    result_state = gr.State([])

    gallery = gr.Gallery(
        label=f"{label_prefix} keyframe ứng viên".strip(),
        columns=5,
        rows=2,
        height=430,
        object_fit="contain",
        preview=True,
    )

    result_table = gr.Dataframe(
        headers=RESULT_HEADERS,
        datatype=["number", "number", "str", "number", "str", "number", "str", "str"],
        label="Chi tiết kết quả",
        interactive=False,
        wrap=True,
    )

    with gr.Row():
        with gr.Column(scale=1):
            selected_image = gr.Image(
                label="Keyframe đã chọn",
                type="filepath",
                interactive=False,
                height=280,
            )

        with gr.Column(scale=1):
            selected_video = gr.Video(
                label="Video preview quanh frame",
                interactive=False,
                height=280,
            )

    with gr.Row():
        selected_video_id = gr.Textbox(label="video_id", interactive=False)
        selected_frame_id = gr.Number(
            label="frame_id",
            precision=0,
            interactive=False,
        )
        selected_keyframe_id = gr.Textbox(
            label="keyframe_id",
            interactive=False,
        )
        selected_score = gr.Number(label="score", interactive=False)

    selected_detail = gr.Markdown("Chưa chọn kết quả.")

    def on_select(results, evt: gr.SelectData):
        return select_gallery_item(results, video_service, evt)

    gallery.select(
        fn=on_select,
        inputs=[result_state],
        outputs=[
            selected_video_id,
            selected_video,
            selected_image,
            selected_frame_id,
            selected_keyframe_id,
            selected_score,
            selected_detail,
        ],
    )

    return {
        "state": result_state,
        "gallery": gallery,
        "table": result_table,
        "video_id": selected_video_id,
        "frame_id": selected_frame_id,
        "keyframe_id": selected_keyframe_id,
        "score": selected_score,
        "selected_image": selected_image,
        "selected_video": selected_video,
        "detail": selected_detail,
    }

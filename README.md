# AIC 2026 Local Retrieval — KIS / QA / TRAKE

Project dùng dữ liệu theo đúng cách sau:

- Milvus Standalone: embedding CLIP `[N,512]`.
- PostgreSQL: map `video_id + n -> frame_idx`, timestamp và FPS.
- Ảnh keyframe JPG: đọc trực tiếp từ ổ máy qua `KEYFRAME_ROOTS`.
- Video MP4 hoặc ZIP video: đọc từ ổ máy qua `VIDEO_ROOTS`; chỉ dùng khi refine frame.
- Không lưu JPG, MP4 hay NPY trong PostgreSQL.

Ba chế độ:

- KIS xuất `video_id,frame_id`.
- QA chạy retrieval + BLIP VQA và xuất `video_id,frame_id,answer`.
- TRAKE tìm chuỗi sự kiện cùng video theo thứ tự tăng dần và xuất `video_id,frame_id_1,...`.

## 1. Cấu hình

Yêu cầu Windows, Python 3.11 và Docker Desktop đang chạy.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

Mở `.env` và sửa hai dòng. Có thể khai báo nhiều thư mục, ngăn bằng `;`:

```dotenv
KEYFRAME_ROOTS=D:\AIC2026_DATA\keyframes
VIDEO_ROOTS=D:\AIC2026_DATA\videos;D:\AIC2026_DATA\video_zips
```

Code tự dò được các cấu trúc phổ biến, ví dụ:

```text
KEYFRAME_ROOTS\Keyframes_L21\keyframes\L21_V001\000010.jpg
KEYFRAME_ROOTS\keyframes\L21_V001\000010.jpg
KEYFRAME_ROOTS\L21_V001\000010.jpg

VIDEO_ROOTS\Videos_L21\video\L21_V001.mp4
VIDEO_ROOTS\...\L21_V001.mp4
VIDEO_ROOTS\some-video-part.zip  (chỉ giải nén video được yêu cầu)
```

Khởi động database:

```powershell
.\INIT_DB_AND_MILVUS.ps1
```

## 2. Dữ liệu embedding và map

Mỗi video phải có ba file cùng `video_id`:

```text
L21_V001.npy   # [N,512], openai/clip-vit-base-patch32, normalized
L21_V001.json  # có image_files theo đúng thứ tự từng vector
L21_V001.csv   # có n, frame_idx, pts_time, fps
```

Kiểm tra toàn bộ trước khi ghi database:

```powershell
.\VALIDATE_DATA.ps1 `
  -Features @(
    "D:\AIC\aic2026-clip-features-part1.zip",
    "D:\AIC\aic2026-clip-features-part2.zip",
    "D:\AIC\aic2026-clip-features-part3.zip",
    "D:\AIC\aic2026-clip-features-part4.zip"
  ) `
  -Maps "D:\AIC\aic2026-map-keyframes-batch1.zip"
```

Chỉ ingest khi lệnh trên in `DATA HỢP LỆ`:

```powershell
.\INGEST.ps1 `
  -Features @(
    "D:\AIC\aic2026-clip-features-part1.zip",
    "D:\AIC\aic2026-clip-features-part2.zip",
    "D:\AIC\aic2026-clip-features-part3.zip",
    "D:\AIC\aic2026-clip-features-part4.zip"
  ) `
  -Maps "D:\AIC\aic2026-map-keyframes-batch1.zip"
```

Bạn cũng có thể ingest từng part rồi xóa ZIP part đó để tiết kiệm ổ cứng. Upsert dùng khóa ổn định nên chạy tiếp không tạo bản ghi trùng.

Kiểm tra sau ingest:

```powershell
.\VERIFY.ps1
```

Script không chỉ so tổng số dòng mà còn đối chiếu từng khóa `video_id + n` giữa PostgreSQL và Milvus.

## 3. Chạy web

```powershell
.\START_WEB.ps1
```

Mở `http://127.0.0.1:7860`.

## 4. Lưu ý accuracy

- Query CLIP phải dùng đúng model đã tạo embedding: `openai/clip-vit-base-patch32`.
- Map CSV quyết định `frame_id` nộp bài; tên JPG chỉ quyết định `n`.
- KIS/TRAKE refine quanh keyframe 1 FPS bằng video gốc. `REFINE_STRIDE=1` xét mọi frame nhưng chạy chậm hơn.
- QA là VQA hình ảnh thật. Câu hỏi phụ thuộc chữ hoặc lời thoại cần thêm OCR/ASR để đạt độ chính xác cao; project có sẵn bảng `text_segments` nhưng gói này không tự tạo OCR/ASR vì đầu vào bạn chốt hiện chỉ có NPY, JSON, map, JPG và video.

## 5. Dừng dịch vụ

```powershell
docker compose down
```

Không dùng `docker compose down -v` trừ khi muốn xóa toàn bộ dữ liệu đã ingest.

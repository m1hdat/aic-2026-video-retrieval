# AIC 2026 Local Retrieval — KIS / QA / TRAKE

Project dùng dữ liệu theo đúng cách sau:

- Milvus Standalone: embedding SigLIP2 `[N,768]`.
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

Mở `.env` và sửa các đường dẫn cùng API key. Có thể khai báo nhiều thư mục,
ngăn bằng `;`:

```dotenv
KEYFRAME_ROOTS=D:\AIC2026_DATA\keyframes
VIDEO_ROOTS=D:\AIC2026_DATA\videos;D:\AIC2026_DATA\video_zips
GOOGLE_TRANSLATE_API_KEY=API_KEY_CUA_BAN
```

Nếu đang nâng cấp từ bản CLIP 512 chiều, đọc `SIGLIP2_MIGRATION.md` trước khi
ingest. Bản mới dùng collection Milvus riêng để giữ an toàn collection cũ.

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
L21_V001.npy   # [N,768], google/siglip2-base-patch16-224, normalized
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

## 3. Bổ sung OCR và Object (không ingest lại SigLIP2)

Giữ nguyên các Docker volume hiện tại. Không chạy lại `INGEST.ps1` nếu `VERIFY.ps1`
đã báo dữ liệu Milvus/PostgreSQL cũ đầy đủ.

Cài thêm dependency đọc CSV/Parquet sau khi chép bản cập nhật:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Tải output Kaggle về máy và để riêng theo loại, ví dụ:

```text
D:\AIC2026_SIGNALS\objects\objects_part1.zip
D:\AIC2026_SIGNALS\objects\objects_part2.zip
D:\AIC2026_SIGNALS\objects\objects_part3.zip
D:\AIC2026_SIGNALS\objects\objects_part4.zip
D:\AIC2026_SIGNALS\ocr\ocr_L21.zip
D:\AIC2026_SIGNALS\ocr\ocr_L22.zip
...
```

Chạy migration đúng một lần. Lệnh này chỉ thêm cột/bảng/index, không xóa dữ liệu:

```powershell
.\MIGRATE_SIGNALS.ps1
```

Ingest tất cả output. Script nhận thư mục, `.zip`, `.jsonl`, `.csv` hoặc `.parquet`;
có upsert nên có thể chạy lại an toàn:

```powershell
.\INGEST_SIGNALS.ps1 `
  -Objects @(
    "D:\AIC2026_SIGNALS\objects\objects_part1.zip",
    "D:\AIC2026_SIGNALS\objects\objects_part2.zip",
    "D:\AIC2026_SIGNALS\objects\objects_part3.zip",
    "D:\AIC2026_SIGNALS\objects\objects_part4.zip"
  ) `
  -Ocr @(
    "D:\AIC2026_SIGNALS\ocr\ocr_L21.zip",
    "D:\AIC2026_SIGNALS\ocr\ocr_L22.zip",
    "D:\AIC2026_SIGNALS\ocr\ocr_L23.zip",
    "D:\AIC2026_SIGNALS\ocr\ocr_L24.zip",
    "D:\AIC2026_SIGNALS\ocr\ocr_L25.zip",
    "D:\AIC2026_SIGNALS\ocr\ocr_L26.zip",
    "D:\AIC2026_SIGNALS\ocr\ocr_L27_L30.zip"
  )
```

Tên file trên chỉ là ví dụ; thay bằng đúng file bạn tải từ Kaggle. Sau đó kiểm tra:

```powershell
.\VERIFY_SIGNALS.ps1
```

Kết quả đúng phải có `OCR rows > 0`, `Object detections > 0` và dòng
`VERIFY SIGNALS OK`. OCR được nối bằng `video_id + keyframe_n`; Object ưu tiên
khóa này và đối chiếu lại `frame_idx` từ PostgreSQL.

## 4. Chạy web

```powershell
.\START_WEB.ps1
```

Mở `http://127.0.0.1:7860`.

Web tự kết hợp SigLIP2 + OCR + Object bằng reciprocal-rank fusion. Cột `nguồn khớp`
cho biết kết quả đến từ `siglip2`, `ocr`, `object` hoặc tổ hợp các nguồn.

## 5. Lưu ý accuracy

- Query phải dùng đúng model đã tạo embedding: `google/siglip2-base-patch16-224`.
- Google Translate API thay cho model dịch local; query tiếng Việt gốc và bản dịch
  tiếng Anh đều được đưa vào SigLIP2.
- Map CSV quyết định `frame_id` nộp bài; tên JPG chỉ quyết định `n`.
- KIS/TRAKE refine quanh keyframe 1 FPS bằng video gốc. `REFINE_STRIDE=1` xét mọi frame nhưng chạy chậm hơn.
- OCR hỗ trợ tìm chữ xuất hiện trong keyframe. Object YOLO11x dùng nhãn COCO và đóng vai trò bổ sung/rerank; nó không thay thế SigLIP2.
- QA là VQA hình ảnh thật. OCR giúp retrieval chọn đúng frame có chữ, nhưng câu trả lời cuối vẫn do BLIP VQA sinh ra.

## 6. Xóa dữ liệu nguồn để tiết kiệm ổ cứng

Sau khi `VERIFY.ps1` và `VERIFY_SIGNALS.ps1` đều thành công, có thể xóa ZIP `.npy`,
map, output OCR/Object đã tải vì dữ liệu cần tìm đã nằm trong Docker volumes.

- Có thể xóa video raw nếu đặt `ENABLE_FRAME_REFINE=false` trong `.env`.
- Chưa nên xóa keyframe JPG nếu vẫn cần xem gallery hoặc chạy QA; PostgreSQL chỉ
  lưu đường dẫn ảnh, không lưu bytes ảnh.
- Không chạy `docker compose down -v`, vì `-v` sẽ xóa cả Milvus và PostgreSQL.

## 7. Dừng dịch vụ

```powershell
docker compose down
```

Không dùng `docker compose down -v` trừ khi muốn xóa toàn bộ dữ liệu đã ingest.

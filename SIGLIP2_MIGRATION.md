# Chuyển hệ thống sang SigLIP2 + Google Translate API

## Những gì đã đổi

- Image/text encoder: `google/siglip2-base-patch16-224`.
- Vector Milvus: 768 chiều, cosine, L2-normalized.
- Collection mới mặc định: `aic2026_siglip2_keyframes`.
- Dịch query: Google Cloud Translation Basic v2 REST API.
- Query SigLIP2 dùng cả tiếng Việt gốc lẫn bản dịch tiếng Anh.
- OCR, Object và PostgreSQL hiện tại được giữ nguyên.
- Frame refinement tiếp tục dùng chính SigLIP2 để chấm các frame gốc gần keyframe.

## Bước 1 — cập nhật `.env` cũ

`setup.ps1` không ghi đè `.env` đã tồn tại. Nếu project đã từng chạy CLIP, mở
`.env` và sửa/thêm đúng các dòng sau:

```dotenv
MILVUS_COLLECTION=aic2026_siglip2_keyframes
EMBEDDING_MODEL=google/siglip2-base-patch16-224
EMBEDDING_DIM=768

ENABLE_GOOGLE_TRANSLATE=true
GOOGLE_TRANSLATE_API_KEY=API_KEY_CUA_BAN
GOOGLE_TRANSLATE_TIMEOUT=10
```

API key phải thuộc Google Cloud project đã bật **Cloud Translation API**. Không
commit hoặc gửi file `.env`. Nếu chưa muốn dùng API, đặt
`ENABLE_GOOGLE_TRANSLATE=false`; SigLIP2 vẫn nhận query tiếng Việt trực tiếp.

## Bước 2 — cài code và tạo collection mới

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
.\INIT_DB_AND_MILVUS.ps1
```

Lệnh này tạo collection 768 chiều mới. Collection CLIP 512 chiều cũ không bị
xóa. Không dùng lại tên collection cũ vì Milvus không thể trộn vector 512 và 768
chiều.

## Bước 3 — kiểm tra và ingest bốn output SigLIP2

Notebook Part 3 public đã được đối chiếu: ZIP chứa thư mục
`clip-features-32/`, bên trong có `Lxx_Vxxx.npy` và `Lxx_Vxxx.json`; vector là
`float32`, 768 chiều, đã normalized. Tên thư mục còn chữ `clip` chỉ để tương
thích pipeline cũ, không có nghĩa đó là vector CLIP cũ.

```powershell
.\VALIDATE_DATA.ps1 `
  -Features @(
    "D:\AIC\aic2026-clip-features-part1.zip",
    "D:\AIC\aic2026-clip-features-part2.zip",
    "D:\AIC\aic2026-clip-features-part3.zip",
    "D:\AIC\aic2026-clip-features-part4.zip"
  ) `
  -Maps "D:\AIC\aic2026-map-keyframes-batch1.zip"

.\INGEST.ps1 `
  -Features @(
    "D:\AIC\aic2026-clip-features-part1.zip",
    "D:\AIC\aic2026-clip-features-part2.zip",
    "D:\AIC\aic2026-clip-features-part3.zip",
    "D:\AIC\aic2026-clip-features-part4.zip"
  ) `
  -Maps "D:\AIC\aic2026-map-keyframes-batch1.zip"

.\VERIFY.ps1
```

Việc ingest SigLIP2 chỉ ghi collection Milvus mới và upsert lại metadata
keyframe. Các bảng OCR/Object đã ingest trong PostgreSQL không bị xóa.

## Bước 4 — kiểm tra frame refinement

```powershell
.\CHECK_FRAME_REFINE.ps1
```

Kết quả `STATUS: READY` nghĩa là code đã bật và tìm thấy nguồn video. Refined
frame không phải dataset dựng sẵn: JPG chỉ được sinh sau khi chạy query và lưu ở
`data/cache/refined/<video_id>/`.

Sau đó chạy web:

```powershell
.\START_WEB.ps1
```

Lưu ý: refine dùng SigLIP2 chấm nhiều frame ảnh. Trên máy không có NVIDIA GPU,
đây có thể là phần chậm nhất. Có thể giảm `REFINE_TOP_N`, tăng `REFINE_STRIDE`,
hoặc tạm đặt `ENABLE_FRAME_REFINE=false` khi cần retrieval nhanh.

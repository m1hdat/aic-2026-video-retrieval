# HƯỚNG DẪN CHẠY AIC 2026 VIDEO RETRIEVAL

Tài liệu này dành cho thành viên vừa clone project và chưa cài đặt gì. Làm lần lượt từ trên xuống, không bỏ qua bước kiểm tra dữ liệu.

Project hỗ trợ ba chế độ:

- **KIS:** tìm video/keyframe từ mô tả.
- **QA:** tìm keyframe rồi trả lời câu hỏi về hình ảnh.
- **TRAKE:** tìm nhiều sự kiện liên tiếp trong cùng video.

Hệ thống dùng **SigLIP2 + Milvus + PostgreSQL**, có thể kết hợp thêm **OCR**, **Object Detection** và **frame refinement**.

---

## 1. Máy cần có gì?

Bắt buộc:

- Windows 10/11 64-bit.
- Git.
- Python **3.11**.
- Docker Desktop và Docker Compose.
- Tối thiểu khoảng 16 GB RAM được khuyến nghị.
- Đủ dung lượng cho Docker volumes, keyframe và video.

Kiểm tra trong PowerShell:

```powershell
git --version
py -3.11 --version
docker --version
docker compose version
```

Sau đó mở **Docker Desktop** và đợi đến khi Docker báo đang chạy.

> Máy không có NVIDIA GPU vẫn chạy được, nhưng SigLIP2, QA và frame refine sẽ chậm hơn nhiều trên CPU.

---

## 2. Clone project

```powershell
cd "THU_MUC_MUON_LUU_PROJECT"
git clone https://github.com/m1hdat/aic-2026-video-retrieval.git
cd aic-2026-video-retrieval
```

Kiểm tra đã vào đúng thư mục:

```powershell
Get-ChildItem
```

Phải thấy các file như:

```text
setup.ps1
START_WEB.ps1
docker-compose.yml
requirements.txt
.env.example
```

---

## 3. Dữ liệu bắt buộc

### 3.1. Để chạy retrieval và xem kết quả

Cần có:

1. **SigLIP2 features** của tất cả video.
2. **Map-keyframe CSV**.
3. **Ảnh keyframe JPG** trên ổ máy.

Mỗi video phải khớp cùng `video_id`, ví dụ `L21_V001`:

```text
L21_V001.npy   # shape [N, 768]
L21_V001.json  # image_files đúng thứ tự N vector
L21_V001.csv   # có n, frame_idx, pts_time, fps
```

Điều kiện quan trọng:

- `.npy` phải được extract bằng `google/siglip2-base-patch16-224`.
- Vector phải có **768 chiều** và đã normalize.
- Không được dùng lẫn feature CLIP 512 chiều với collection SigLIP2 768 chiều.
- Số vector trong `.npy`, số `image_files` trong JSON và số keyframe hợp lệ phải khớp.
- `video_id + n` trong feature/map phải thống nhất.
- CSV map phải có `frame_idx`; đây là frame ID dùng để nộp bài.
- Ảnh JPG phải giữ đúng tên/thứ tự tương ứng với `n`.

Ví dụ cấu trúc ảnh được project tự nhận:

```text
D:\AIC2026_DATA\keyframes\Keyframes_L21\keyframes\L21_V001\000010.jpg
D:\AIC2026_DATA\keyframes\keyframes\L21_V001\000010.jpg
D:\AIC2026_DATA\keyframes\L21_V001\000010.jpg
```

### 3.2. Khi bật frame refinement

Phải có thêm video gốc `.mp4` hoặc ZIP chứa video, ví dụ:

```text
D:\AIC2026_DATA\videos\Videos_L21\video\L21_V001.mp4
D:\AIC2026_DATA\videos\video_part1.zip
```

Nếu không có video gốc, đặt:

```dotenv
ENABLE_FRAME_REFINE=false
```

Retrieval chính vẫn chạy bình thường; chỉ không tinh chỉnh từ keyframe 1 FPS sang native frame.

### 3.3. OCR và Object Detection

Đây là dữ liệu bổ sung để tăng khả năng tìm chữ/vật thể. Có thể là thư mục, `.zip`, `.jsonl`, `.csv` hoặc `.parquet`.

Ví dụ:

```text
D:\AIC2026_SIGNALS\ocr\ocr_L21.zip
D:\AIC2026_SIGNALS\objects\objects_part1.zip
```

Nếu chưa có dữ liệu này, đặt:

```dotenv
ENABLE_OCR_SEARCH=false
ENABLE_OBJECT_SEARCH=false
```

---

## 4. Cài project lần đầu

Trong PowerShell tại thư mục project:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

Script sẽ:

- Tạo `.env` từ `.env.example` nếu chưa có.
- Tạo môi trường Python `.venv`.
- Cài dependency trong `requirements.txt`.
- Khởi động PostgreSQL, Milvus và các container liên quan.

Nếu PowerShell vẫn chặn script, chạy từng lệnh bằng tiền tố:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

---

## 5. Sửa file `.env`

Mở `.env` bằng VS Code/Notepad và sửa đường dẫn theo máy đang chạy. Không sửa mỗi `.env.example`, vì chương trình đọc `.env`.

Ví dụ:

```dotenv
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=aic2026
POSTGRES_USER=aic
POSTGRES_PASSWORD=aic2026

MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=aic2026_siglip2_keyframes
EMBEDDING_MODEL=google/siglip2-base-patch16-224
EMBEDDING_DIM=768

APP_HOST=127.0.0.1
APP_PORT=7860

KEYFRAME_ROOTS=D:\AIC2026_DATA\keyframes
VIDEO_ROOTS=D:\AIC2026_DATA\videos

ENABLE_GOOGLE_TRANSLATE=true
GOOGLE_TRANSLATE_API_KEY=API_KEY_CUA_NHOM
GOOGLE_TRANSLATE_TIMEOUT=10

ENABLE_OCR_SEARCH=true
ENABLE_OBJECT_SEARCH=true
SIGNAL_CANDIDATE_K=300

ENABLE_FRAME_REFINE=true
REFINE_TOP_N=1
TRAKE_REFINE_TOP_N=1
REFINE_SECONDS=0.6
REFINE_STRIDE=6
REFINE_BATCH_SIZE=8
REFINE_MIN_GAIN=0.003
```

Có nhiều thư mục keyframe/video thì ngăn cách bằng dấu `;`:

```dotenv
KEYFRAME_ROOTS=D:\DataPart1\keyframes;E:\DataPart2\keyframes
VIDEO_ROOTS=D:\Videos;E:\VideoZips
```

Lưu ý:

- Không commit hoặc gửi file `.env` lên GitHub vì có thể chứa API key.
- Nếu chưa có Google Translate API key, đặt `ENABLE_GOOGLE_TRANSLATE=false`.
- Nếu chưa có OCR/Object, tắt hai cờ tương ứng.
- Frame refine trên CPU rất chậm. Có thể đặt `ENABLE_FRAME_REFINE=false` để trả kết quả nhanh.
- Các biến `$env:...` đã đặt trong PowerShell có thể ghi đè `.env`. Nếu nghi ngờ, đóng PowerShell rồi mở cửa sổ mới.

---

## 6. Khởi tạo PostgreSQL và Milvus

Chạy sau khi Docker Desktop đã mở:

```powershell
.\INIT_DB_AND_MILVUS.ps1
```

Kiểm tra container:

```powershell
docker compose ps
```

Các service chính phải có trạng thái `Up` hoặc `running`. Nếu Milvus chưa sẵn sàng, đợi khoảng 30–60 giây rồi chạy lại `INIT_DB_AND_MILVUS.ps1`.

---

## 7. Validate và ingest SigLIP2 + map

Phần này chỉ cần làm khi máy/Docker volumes chưa có dữ liệu retrieval.

### 7.1. Validate trước

Thay các đường dẫn bằng đúng file trên máy:

```powershell
.\VALIDATE_DATA.ps1 `
  -Features @(
    "D:\AIC_DATA\siglip2_features_part1.zip",
    "D:\AIC_DATA\siglip2_features_part2.zip",
    "D:\AIC_DATA\siglip2_features_part3.zip",
    "D:\AIC_DATA\siglip2_features_part4.zip"
  ) `
  -Maps "D:\AIC_DATA\map_keyframes_batch1.zip"
```

Chỉ ingest khi script báo dữ liệu hợp lệ. Nếu báo dimension 512, đó là feature CLIP cũ và không dùng được cho collection 768 chiều hiện tại.

### 7.2. Ingest

```powershell
.\INGEST.ps1 `
  -Features @(
    "D:\AIC_DATA\siglip2_features_part1.zip",
    "D:\AIC_DATA\siglip2_features_part2.zip",
    "D:\AIC_DATA\siglip2_features_part3.zip",
    "D:\AIC_DATA\siglip2_features_part4.zip"
  ) `
  -Maps "D:\AIC_DATA\map_keyframes_batch1.zip"
```

Có thể ingest từng part để tiết kiệm ổ cứng. Script dùng upsert nên chạy lại không tạo bản ghi trùng.

Kiểm tra sau ingest:

```powershell
.\VERIFY.ps1
```

Chỉ chuyển sang bước tiếp theo khi PostgreSQL và Milvus khớp số lượng/khóa `video_id + n`.

---

## 8. Ingest OCR/Object (nếu có)

Không cần ingest lại SigLIP2.

Chạy migration một lần:

```powershell
.\MIGRATE_SIGNALS.ps1
```

Ingest file thực tế của nhóm:

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
    "D:\AIC2026_SIGNALS\ocr\ocr_L22.zip"
  )
```

Không cần dùng đúng tên ví dụ; chỉ cần truyền đúng đường dẫn các file thật.

Kiểm tra:

```powershell
.\VERIFY_SIGNALS.ps1
```

Kết quả mong đợi:

- `OCR rows > 0` nếu đã ingest OCR.
- `Object detections > 0` nếu đã ingest Object.
- Có dòng `VERIFY SIGNALS OK`.

---

## 9. Chạy web

Mỗi lần mở máy:

1. Mở Docker Desktop.
2. Mở PowerShell tại thư mục project.
3. Chạy:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
docker compose up -d
.\START_WEB.ps1
```

Mở trình duyệt tại:

```text
http://127.0.0.1:7860
```

Dừng web bằng `Ctrl+C`. Dừng container khi không dùng nữa:

```powershell
docker compose down
```

**Không chạy `docker compose down -v`**, vì `-v` sẽ xóa dữ liệu PostgreSQL và Milvus đã ingest.

---

## 10. Checklist trước khi thi

Chạy:

```powershell
docker compose ps
.\VERIFY.ps1
.\VERIFY_SIGNALS.ps1
.\CHECK_FRAME_REFINE.ps1
```

Sau đó kiểm tra thủ công:

- [ ] Web mở được tại `127.0.0.1:7860`.
- [ ] KIS trả được gallery và đúng `video_id, frame_id`.
- [ ] QA trả được `video_id, frame_id, answer`.
- [ ] TRAKE trả các frame cùng video theo đúng thứ tự thời gian.
- [ ] Ảnh keyframe hiển thị, không bị `file not found`.
- [ ] Query tiếng Việt dịch được; nếu API lỗi thì tắt Google Translate và dùng tiếng Anh.
- [ ] OCR/Object có nguồn khớp nếu đã ingest signals.
- [ ] Frame refine không làm query vượt thời gian cho phép.
- [ ] Thử ít nhất hai query khó cho mỗi loại trước ngày thi.

Với thời gian thi khoảng 2 giờ cho khoảng 30 câu, không nên để một query chạy hàng trăm giây. Nếu chậm, ưu tiên:

```dotenv
ENABLE_FRAME_REFINE=false
```

hoặc chỉ refine top 1 bằng cấu hình nhanh trong phần `.env` ở trên.

---

## 11. Lỗi thường gặp

### PowerShell báo script không được ký

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Sau đó chạy lại script. Scope `Process` chỉ áp dụng cho cửa sổ PowerShell hiện tại.

### `docker` không kết nối được

- Mở Docker Desktop.
- Đợi Docker chạy hoàn toàn.
- Chạy `docker compose up -d`.

### Milvus báo connection refused tại port 19530

```powershell
docker compose ps
docker compose up -d
```

Đợi 30–60 giây rồi chạy lại `INIT_DB_AND_MILVUS.ps1`.

### PostgreSQL báo thiếu bảng/relation

```powershell
.\INIT_DB_AND_MILVUS.ps1
.\MIGRATE_SIGNALS.ps1
```

### Không thấy ảnh keyframe

- Kiểm tra `KEYFRAME_ROOTS` trong `.env`.
- Kiểm tra thư mục có chứa đúng `video_id` như `L21_V001`.
- Kiểm tra JPG vẫn còn trên ổ máy.
- Restart web sau khi sửa `.env`.

### Refine không chạy hoặc không tìm thấy video

- Kiểm tra `ENABLE_FRAME_REFINE=true`.
- Kiểm tra `VIDEO_ROOTS`.
- Kiểm tra video/ZIP thực sự chứa đúng `video_id`.
- Nếu không cần refine, tắt nó để retrieval vẫn chạy.

### Query rất chậm

- Đặt `ENABLE_FRAME_REFINE=false` để xác định refine có phải nguyên nhân không.
- Trên CPU, dùng `REFINE_TOP_N=1`, `REFINE_SECONDS=0.6`, `REFINE_STRIDE=6`.
- Không bật QA nếu chỉ cần KIS.
- Không chạy ingest trong lúc đang thi/query.

### Đổi `.env` nhưng chương trình vẫn dùng giá trị cũ

Dừng web bằng `Ctrl+C`, đóng PowerShell, mở cửa sổ mới rồi chạy lại. Có thể kiểm tra cấu hình refine bằng:

```powershell
& ".\.venv\Scripts\python.exe" -c "from src.settings import settings; print(settings.enable_frame_refine, settings.refine_top_n, settings.refine_seconds, settings.refine_stride)"
```

---

## 12. Cập nhật code mới từ GitHub

Không chạy lại `git clone` nếu đã có project. Trong thư mục project:

```powershell
git status
git pull origin main
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
docker compose up -d
.\START_WEB.ps1
```

Nếu `git status` có code đang sửa, không pull/ghi đè bừa. Commit hoặc hỏi người phụ trách repo trước.

File `.env` và Docker volumes nằm ở máy local, bình thường không bị `git pull` xóa.

---

## 13. Tóm tắt cực ngắn

### Máy mới, chưa có dữ liệu database

```powershell
git clone https://github.com/m1hdat/aic-2026-video-retrieval.git
cd aic-2026-video-retrieval
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
# Sửa .env
.\INIT_DB_AND_MILVUS.ps1
# VALIDATE_DATA.ps1 -> INGEST.ps1 -> VERIFY.ps1
# Nếu có OCR/Object: MIGRATE_SIGNALS.ps1 -> INGEST_SIGNALS.ps1 -> VERIFY_SIGNALS.ps1
.\START_WEB.ps1
```

### Máy đã ingest dữ liệu rồi

```powershell
Set-ExecutionPolicy -Scope Process Bypass
docker compose up -d
.\START_WEB.ps1
```

> Nhớ: code nằm trên GitHub, nhưng features, map, keyframe, video, OCR/Object và Docker database không tự xuất hiện sau khi clone. Nhóm phải chép/tải đúng bộ dữ liệu sang máy chạy hệ thống.

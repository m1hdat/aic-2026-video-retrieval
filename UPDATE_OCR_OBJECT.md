# Cập nhật OCR + Object vào project AIC2026 hiện tại

## Nguyên tắc quan trọng

Giải nén gói update **vào đúng thư mục project đang chạy hiện tại** và cho phép
ghi đè các file trùng tên. Không chạy từ một thư mục project mới, vì Docker
Compose có thể tạo bộ volume mới và database cũ sẽ tạm thời không xuất hiện.

Không thay hoặc xóa file `.env` hiện tại.

Cập nhật dependency để đọc được cả CSV và Parquet:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## 1. Kiểm tra dữ liệu cũ

Mở PowerShell tại thư mục project:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
docker compose up -d
.\VERIFY.ps1
```

Chỉ tiếp tục khi PostgreSQL và Milvus cũ đều có dữ liệu.

## 2. Tải output Kaggle

Tải 4 output Object và toàn bộ output OCR về một thư mục, ví dụ:

```text
D:\AIC2026_SIGNALS\objects\
D:\AIC2026_SIGNALS\ocr\
```

Giữ nguyên ZIP cũng được; không bắt buộc giải nén.

## 3. Thêm schema PostgreSQL

```powershell
.\MIGRATE_SIGNALS.ps1
```

Migration chỉ thêm cột/index và bảng `object_detections`. Không đụng tới
collection Milvus hoặc vector embedding hình ảnh.

## 4. Ingest OCR và Object

Cách đơn giản nhất là truyền thẳng hai thư mục; importer tự tìm JSONL, CSV,
Parquet và ZIP bên trong:

```powershell
.\INGEST_SIGNALS.ps1 `
  -Objects "D:\AIC2026_SIGNALS\objects" `
  -Ocr "D:\AIC2026_SIGNALS\ocr"
```

Hoặc truyền danh sách từng ZIP nếu muốn kiểm soát từng phần:

```powershell
.\INGEST_SIGNALS.ps1 `
  -Objects @(
    "D:\AIC2026_SIGNALS\objects\objects_part1.zip",
    "D:\AIC2026_SIGNALS\objects\objects_part2.zip",
    "D:\AIC2026_SIGNALS\objects\objects_part3.zip",
    "D:\AIC2026_SIGNALS\objects\objects_part4.zip"
  ) `
  -Ocr @(
    "D:\AIC2026_SIGNALS\ocr\ocr_part1.zip",
    "D:\AIC2026_SIGNALS\ocr\ocr_part2.zip",
    "D:\AIC2026_SIGNALS\ocr\ocr_part3.zip",
    "D:\AIC2026_SIGNALS\ocr\ocr_part4.zip"
  )
```

Thay tên ví dụ bằng tên file thực tế. Chạy lại không tạo bản ghi trùng vì
importer dùng `source_key` ổn định và upsert.

## 5. Xác minh

```powershell
.\VERIFY_SIGNALS.ps1
```

Phải thấy:

```text
OCR rows: ...
Object detections: ...
VERIFY SIGNALS OK
```

Sau đó chạy web:

```powershell
.\START_WEB.ps1
```

Trong tab KIS, cột `nguồn khớp` sẽ hiện `siglip2`, `ocr`, `object` hoặc tổ hợp.

## 6. Có thể xóa gì?

Sau khi cả `VERIFY.ps1` và `VERIFY_SIGNALS.ps1` thành công:

- Có thể xóa ZIP embedding, map, OCR và Object đã tải.
- Có thể xóa video raw nếu đổi `ENABLE_FRAME_REFINE=false` trong `.env`.
- Chưa nên xóa keyframe JPG nếu cần gallery và QA; PostgreSQL chỉ lưu đường dẫn.
- Tuyệt đối không chạy `docker compose down -v` vì lệnh đó xóa volume database.

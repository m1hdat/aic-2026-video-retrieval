# AIC 2026 Video Retrieval System

Hệ thống Video Retrieval được phát triển cho cuộc thi **AI Challenge 2026 (AIC 2026)**.

Project hỗ trợ toàn bộ các bài toán của vòng sơ tuyển:

- Textual Known-Item Search (KIS)
- Question Answering (Q&A)
- Temporal Reasoning and Knowledge Extraction (TRAKE)

Hệ thống sử dụng:

- CLIP để mã hóa truy vấn văn bản
- Milvus để tìm kiếm vector
- Gradio để xây dựng giao diện Web
- Docker để triển khai Milvus Server

---

# Project Structure

```
AIC26_web/

├── assets/                 # Ảnh, icon và tài nguyên giao diện

├── configs/                # File cấu hình hệ thống
│   └── config.yaml

├── data/
│   ├── milvus/             # Milvus Lite database (nếu dùng Lite)
│   ├── processed/          # Manifest sau khi preprocess
│   └── raw/                # Dataset AIC (nếu lưu local)

├── notebooks/              # Notebook preprocess và build Milvus

├── outputs/                # Docker runtime (Milvus, MinIO, etcd...)

├── scripts/                # Script tạo Collection, Index, Test Retrieval

├── src/                    # Backend của hệ thống
│
├── web/                    # Giao diện Gradio
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Chức năng của từng thư mục

## configs/

Quản lý toàn bộ cấu hình của project.

Ví dụ:

- Model CLIP
- Đường dẫn dataset
- Milvus
- Search parameters

---

## data/

Chứa dữ liệu sử dụng trong hệ thống.

### processed/

Sau khi preprocess sẽ sinh ra:

```
manifest_keyframes.parquet
manifest_videos.parquet
```

Đây là metadata dùng trong Retrieval.

---

### milvus/

Nếu chạy Milvus Lite:

```
aic2026_milvus.db
```

Nếu dùng Docker thì thư mục này không cần.

---

### raw/

Nếu lưu dataset local:

```
Part1
Part2
Part3
Part4
```

---

## notebooks/

Notebook chạy trên Kaggle.

Bao gồm:

- preprocess từng Part
- merge dataset
- build Milvus

---

## scripts/

Các script backend.

Ví dụ:

```
01_prepare_dataset.py

02_create_collection.py

03_index_images.py

04_test_search.py

05_reset_collection.py
```

---

## src/

Toàn bộ backend của hệ thống.

Bao gồm:

- CLIP Encoder
- Milvus Client
- Search Engine
- Metadata
- Retrieval Service
- Video Service
- Submission Manager

---

## web/

Giao diện Gradio.

Bao gồm:

- KIS
- Q&A
- TRAKE
- Gallery
- Answer Queue
- Export CSV

---

# System Pipeline

## 1. Dataset Preprocessing

BTC cung cấp:

```
Videos
Keyframes
CLIP Features
Metadata
Objects
```

↓

Notebook preprocess tạo:

```
manifest_keyframes.parquet

manifest_videos.parquet
```

↓

Merge toàn bộ 4 Part thành một bộ metadata thống nhất.

---

## 2. Build Milvus

Đọc:

```
manifest_keyframes.parquet

CLIP Features (.npy)
```

↓

Tạo Collection trong Milvus.

Milvus lưu:

- embedding
- video_id
- frame_id
- timestamp
- keyframe path
- video path

---

## 3. Retrieval

Người dùng nhập:

```
Text Query
```

↓

CLIP Text Encoder

↓

Vector Query

↓

Milvus Search

↓

Top-K Keyframes

↓

Metadata

↓

Hiển thị lên Web

---

## 4. User Interaction

Người dùng có thể:

- xem keyframe
- xem video
- chọn kết quả
- thêm vào Answer Queue
- xuất Submission CSV

---

# Development Workflow

```
BTC Dataset

        │

        ▼

Notebook Preprocessing

        │

        ▼

Merge Dataset

        │

        ▼

Build Milvus

        │

        ▼

Search Engine

        │

        ▼

Web Interface

        │

        ▼

Submission CSV
```

---

# Clone & Run Project

## 1. Clone project

```bash
git clone https://github.com/m1hdat/aic-2026-4funXD.git

cd AIC26_web
```

---

## 2. Tạo môi trường

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

## 4. Chuẩn bị dữ liệu

### Sau khi preprocess

Đặt các file:

```
data/

processed/

    manifest_keyframes.parquet

    manifest_videos.parquet
```

Nếu dùng Milvus Lite:

```
data/

milvus/

    aic2026_milvus.db
```

---

## 5. Chạy Milvus

### Milvus Lite

Không cần Docker.

Trong `config.yaml`:

```yaml
milvus:
    mode: lite
```

---

### Milvus Standalone (Khuyến nghị)

```bash
docker compose up -d
```

Trong `config.yaml`

```yaml
milvus:
    mode: server
```

---

## 6. Kiểm tra Retrieval

```bash
python -m scripts.04_test_search "một người đang lái xe máy" --top-k 10
```

Nếu hiển thị:

```
Top-10 Results
```

thì Milvus hoạt động bình thường.

---

## 7. Chạy Web

```bash
python -m web.gradio_app
```

Mở trình duyệt:

```
http://127.0.0.1:7860
```

---

## 8. Thực hiện Retrieval

- Nhập truy vấn
- Xem kết quả
- Chọn đáp án
- Export CSV

---

# Công nghệ sử dụng

- Python
- PyTorch
- Transformers
- CLIP
- Milvus
- Docker
- Gradio
- OpenCV

---

# Team

AI Challenge 2026

Video Retrieval System
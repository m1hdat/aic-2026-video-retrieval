# AIC 2026 Video Retrieval System

A web-based retrieval system developed for the **AI Challenge 2026 (AIC 2026)**.

The system is designed to support all preliminary round retrieval tasks:

- Textual Known-Item Search (Textual KIS)
- Question Answering (Q&A)
- Temporal Reasoning and Knowledge Extraction (TRAKE)

Current frontend is developed with **Gradio** and can run in two modes:

- **Mock Mode** (no dataset required)
- **Real Mode** (Milvus + processed AIC dataset)

---

# Project Structure

```
AIC26_web/

├── assets/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
│
├── frontend/
│   ├── gradio_app.py
│   └── components/
│
├── notebooks/
├── outputs/
├── scripts/
├── src/
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# System Architecture

```
User Query
      │
      ▼
Gradio Frontend
      │
      ▼
Retrieval Service
      │
      ▼
Search Engine
      │
      ▼
CLIP Text Encoder
      │
      ▼
Milvus Vector Database
      │
      ▼
Retrieved Keyframes
      │
      ▼
Frontend Result Browser
      │
      ▼
Answer Queue
      │
      ▼
Submission CSV
```

---

# Installation

Clone repository

```bash
git clone https://github.com/m1hdat/aic-2026-4funXD.git

cd AIC26_web
```

Create virtual environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Run Frontend (Mock Mode)

Mock mode **does not require**

- dataset
- Milvus
- Docker

Enable mock mode

Windows PowerShell

```powershell
$env:USE_MOCK="true"
```

CMD

```cmd
set USE_MOCK=true
```

Run

```bash
python frontend/gradio_app.py
```

Open browser

```
http://127.0.0.1:7860
```

Current features

- Textual KIS
- Q&A
- TRAKE
- Gallery
- Answer Queue
- Export CSV
- Save Session
- Load Session

---

# Run Backend (Real Mode)

After the AIC dataset has been processed.

Start Milvus

```bash
docker compose up -d
```

Disable mock mode

Windows PowerShell

```powershell
$env:USE_MOCK="false"
```

Run frontend

```bash
python frontend/gradio_app.py
```

---

# Dataset Pipeline

```
Raw Dataset
        │
        ▼
scripts/
        │
        ▼
Processed Metadata
Processed Embeddings
        │
        ▼
Milvus Collection
        │
        ▼
Search Engine
        │
        ▼
Frontend
```

---

# Data Structure

```
data/

raw/

    aic/

        videos/

        keyframes/

        metadata/

        objects/

processed/

    aic/

        keyframe_metadata.csv

        keyframe_embeddings.npy
```

---

# Main Components

Frontend

```
frontend/

gradio_app.py

components/

    kis_tab.py

    qa_tab.py

    trake_tab.py

    result_browser.py

    answer_queue.py
```

Backend

```
src/

clip_encoder.py

search_engine.py

milvus_client.py

retrieval_service.py

submission_manager.py

video_service.py
```

---

# Configuration

Main configuration

```
configs/config.yaml
```

Modify

- model
- dataset path
- Milvus collection
- search parameters

without changing source code.

---

# Development Workflow

Stage 1

Frontend development

```
Mock Mode
```

Stage 2

Dataset preprocessing

```
scripts/
```

Stage 3

Milvus indexing

```
indexer.py
```

Stage 4

Retrieval integration

```
search_engine.py
```

Stage 5

Evaluation

```
submission CSV
```

---

# Technologies

- Python
- Gradio
- PyTorch
- Transformers
- Milvus
- Docker
- OpenCV

---

# Team

AI Challenge 2026

Video Retrieval System
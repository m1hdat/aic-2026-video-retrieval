# Milvus Guide

Milvus is the vector database used by this project to store CLIP image embeddings and search them with text embeddings.

## What Is Milvus?

Milvus is a vector database designed to store, index, and search high-dimensional vectors efficiently.

In machine learning systems, raw data such as images, text, audio, or video is often converted into numeric vectors called embeddings. These embeddings capture semantic meaning. For example, two visually similar images should have embeddings that are close to each other in vector space.

Traditional databases are good at exact matching, such as finding a row by ID or filtering by a column value. Milvus is built for similarity search, where the goal is to find the vectors most similar to a query vector.

In this project:

- Each image is encoded by CLIP into a 512-dimensional vector.
- Each text query is also encoded by CLIP into a 512-dimensional vector.
- Milvus stores the image vectors.
- When a user enters a text query, Milvus finds the nearest image vectors.

This is what makes text-to-image retrieval possible.

## Why Use Milvus?

For a small number of images, you could compare vectors with NumPy. For a larger dataset, that approach becomes slow and harder to manage.

Milvus provides:

- Fast vector similarity search.
- Persistent storage for embeddings and metadata.
- Indexing methods such as `IVF_FLAT`.
- Search parameters such as `nprobe`.
- A database-style interface for inserting, deleting, and querying vectors.

## Role in This Project

The retrieval flow is:

1. Images are encoded offline with CLIP on Kaggle.
2. The encoded vectors are saved to `data/processed/image_embeddings.npy`.
3. Metadata is saved to `data/processed/image_metadata.csv`.
4. `scripts/03_index_images.py` loads both files and inserts vectors into Milvus.
5. `scripts/04_test_search.py` and `frontend/gradio_app.py` encode text queries and search Milvus.

## Collection Configuration

The main Milvus settings are in `configs/config.yaml`:

```yaml
milvus:
  host: localhost
  port: 19530
  collection_name: text_image_retrieval
  metric_type: IP
  index_type: IVF_FLAT
  nlist: 128
  nprobe: 10
```

The collection stores:

- `id`: auto-generated primary key.
- `image_path`: path to the local image file.
- `caption`: reference caption for display.
- `embedding`: CLIP image vector.

The embedding dimension is configured as:

```yaml
model:
  embedding_dim: 512
```

This matches `openai/clip-vit-base-patch32`.

## Start Milvus

Milvus must be running before creating or loading a collection:

```bash
docker compose up -d
```

Check service status:

```bash
docker compose ps
```

## Create Collection

Create the configured collection:

```bash
python scripts/02_create_collection.py
```

Expected output:

```text
Collection is ready: text_image_retrieval
```

## Load Precomputed Embeddings

Before indexing, make sure these files exist:

```text
data/processed/image_metadata.csv
data/processed/image_embeddings.npy
```

Then insert vectors into Milvus:

```bash
python scripts/03_index_images.py
```

Expected output:

```text
Inserted 8091 precomputed image embeddings into Milvus.
```

The exact number can differ if you encoded a subset on Kaggle.

## Test Search

Run a terminal search:

```bash
python scripts/04_test_search.py
```

Or pass a custom query:

```bash
python scripts/04_test_search.py "a dog running on the grass"
```

## Reset Collection

If you need to rebuild the collection from scratch:

```bash
python scripts/05_reset_collection.py
python scripts/03_index_images.py
```

This drops and recreates only the configured Milvus collection. It does not delete Docker containers or local files.

## About Deprecation Warnings

You may see warnings like:

```text
PyMilvusDeprecationWarning: ORM-style PyMilvus API will be removed in PyMilvus 3.1
```

These warnings are not runtime errors. If the script prints messages such as:

```text
Collection is ready: text_image_retrieval
Reset collection: text_image_retrieval
```

then the operation succeeded.

The warning means the current code uses the older PyMilvus ORM API. It still works with the installed PyMilvus version, but the code can be migrated to `MilvusClient` later for long-term compatibility.

## Common Problems

### Milvus connection fails

Make sure Docker services are running:

```bash
docker compose ps
```

Then inspect Milvus logs:

```bash
docker compose logs milvus
```

### Metadata and embeddings do not match

`image_metadata.csv` and `image_embeddings.npy` must have the same number of rows and the same order.

Fix:

1. Re-run `scripts/kaggle_encode_images.py` on Kaggle.
2. Download both output files from the same Kaggle run.
3. Replace both local files in `data/processed/`.

### Search returns no results

Check that indexing completed successfully:

```bash
python scripts/03_index_images.py
```

If the collection already contains old or partial data, reset and index again:

```bash
python scripts/05_reset_collection.py
python scripts/03_index_images.py
```

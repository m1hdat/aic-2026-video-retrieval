# Text-Image Retrieval Demo

This demo builds an image search system using text queries with CLIP and Milvus. The default dataset points to Flickr8k, which has already been downloaded in the workspace at `flickr8k/versions/1`.

## Architecture

Main processing flow:

1. Read images and captions from Flickr8k.
2. Normalize the metadata into `data/processed/image_metadata.csv`.
3. Encode images with CLIP offline on Kaggle and export `image_embeddings.npy`.
4. Store precomputed embeddings, image paths, and captions in Milvus.
5. Encode the text query with CLIP and search for the nearest images in Milvus.
6. Display the results using Gradio.

## Installation

```bash
conda create -n text-image-retrieval python=3.11 -y
conda activate text-image-retrieval
pip install -r requirements.txt
```

If PowerShell does not recognize the `conda` command, run the commands above in Anaconda Prompt.

After installation, quickly check the environment:

```bash
python --version
python -c "import torch, transformers, pymilvus, gradio; print('Environment is ready')"
```

## Run Milvus

```bash
docker compose up -d
```

Milvus standalone will be available on port `19530`.

## Encode Images Offline on Kaggle

The local indexing script does not encode images directly. It loads precomputed CLIP image vectors from:

```text
data/processed/image_embeddings.npy
```

Run this script on Kaggle with the Flickr8k dataset attached:

```bash
python scripts/kaggle_encode_images.py
```

You can also use this sample Kaggle notebook as a reference:

```text
https://www.kaggle.com/code/ericnguyen1203/encode-image-offline
```

The Kaggle script writes these files to `/kaggle/working`:

```text
image_metadata.csv
image_embeddings.npy
```

Download both files from Kaggle and place them here:

```text
data/processed/image_metadata.csv
data/processed/image_embeddings.npy
```

Important: `image_metadata.csv` and `image_embeddings.npy` must have the same row order. The Kaggle script already guarantees this when both files come from the same run.

## Run Step by Step

After downloading `image_metadata.csv` and `image_embeddings.npy` from Kaggle into `data/processed`, run:

```bash
python scripts/02_create_collection.py
python scripts/03_index_images.py
python scripts/04_test_search.py
python frontend/gradio_app.py
```

Only run `python scripts/01_prepare_dataset.py` if you want to regenerate metadata locally. Do not run it after downloading Kaggle metadata unless you are sure the local metadata order matches the `.npy` embedding order.

Make sure the Conda environment is activated before running the commands:

```bash
conda activate text-image-retrieval
```

## Configuration

Main configuration file: `configs/config.yaml`

Default paths:

* Images: `flickr8k/versions/1/Images`
* Captions: `flickr8k/versions/1/captions.txt`
* Processed metadata: `data/processed/image_metadata.csv`
* Precomputed image embeddings: `data/processed/image_embeddings.npy`

If you want to change the dataset to `data/raw/flickr8k_sample`, simply update `image_dir` and `captions_path` in `configs/config.yaml`.

## Sample Queries

Some queries you can try:

* `a dog running on the grass`
* `a man riding a bicycle`
* `a child playing outside`
* `two dogs playing together`
* `a soccer player kicking a ball`

A longer list is available in `data/sample_queries.txt`.

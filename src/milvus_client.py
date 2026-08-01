from __future__ import annotations

from typing import Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


VECTOR_FIELD = "embedding"
IMAGE_PATH_FIELD = "image_path"
CAPTION_FIELD = "caption"


def connect_milvus(host: str = "localhost", port: int = 19530, alias: str = "default") -> None:
    """Connect pymilvus to the configured Milvus server."""
    connections.connect(alias=alias, host=host, port=str(port))


def create_collection(config: dict[str, Any]) -> Collection:
    """Create the retrieval collection if it does not already exist."""
    milvus = config["milvus"]
    model = config["model"]
    collection_name = milvus["collection_name"]

    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name=IMAGE_PATH_FIELD, dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name=CAPTION_FIELD, dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name=VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR, dim=model["embedding_dim"]),
        ]
        schema = CollectionSchema(fields=fields, description="CLIP text-image retrieval")
        collection = Collection(name=collection_name, schema=schema)

    if not collection.indexes:
        index_params = {
            "metric_type": milvus.get("metric_type", "IP"),
            "index_type": milvus.get("index_type", "IVF_FLAT"),
            "params": {"nlist": milvus.get("nlist", 128)},
        }
        collection.create_index(field_name=VECTOR_FIELD, index_params=index_params)

    collection.load()
    return collection


def drop_collection(collection_name: str) -> None:
    """Drop a collection when a full rebuild is needed."""
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)


def get_collection(collection_name: str) -> Collection:
    collection = Collection(collection_name)
    collection.load()
    return collection


def insert_embeddings(
    collection: Collection,
    image_paths: list[str],
    captions: list[str],
    embeddings: list[list[float]],
) -> None:
    """Insert image metadata and vectors into Milvus."""
    collection.insert([image_paths, captions, embeddings])
    collection.flush()


def search_vectors(
    collection: Collection,
    query_embedding: list[float],
    top_k: int = 5,
    nprobe: int = 10,
    metric_type: str = "IP",
) -> list[dict[str, Any]]:
    """Search vectors and return a simple list of ranked result dictionaries."""
    search_params = {"metric_type": metric_type, "params": {"nprobe": nprobe}}
    hits = collection.search(
        data=[query_embedding],
        anns_field=VECTOR_FIELD,
        param=search_params,
        limit=top_k,
        output_fields=[IMAGE_PATH_FIELD, CAPTION_FIELD],
    )

    results = []
    for rank, hit in enumerate(hits[0], start=1):
        results.append(
            {
                "rank": rank,
                "score": float(hit.distance),
                "image_path": hit.entity.get(IMAGE_PATH_FIELD),
                "caption": hit.entity.get(CAPTION_FIELD),
            }
        )

    return results

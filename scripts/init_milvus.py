from pymilvus import MilvusClient, DataType
from src.settings import settings

c=MilvusClient(uri=settings.milvus_uri)
if not c.has_collection(settings.collection):
    schema=c.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("video_id", DataType.VARCHAR, max_length=32)
    schema.add_field("keyframe_n", DataType.INT32)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)
    indexes=c.prepare_index_params()
    # IVF_SQ8 nén phần index xuống 8-bit, hợp lý hơn HNSW khi máy ít dung lượng.
    # Milvus vẫn giữ source vector để đảm bảo khả năng rebuild/index consistency.
    indexes.add_index("embedding", index_type="IVF_SQ8", metric_type="COSINE", params={"nlist":4096})
    c.create_collection(settings.collection, schema=schema, index_params=indexes)
description=c.describe_collection(settings.collection)
embedding_field=next((f for f in description.get("fields",[]) if f.get("name")=="embedding"),{})
actual_dim=int(embedding_field.get("params",{}).get("dim",embedding_field.get("dim",0)) or 0)
if actual_dim and actual_dim != settings.embedding_dim:
    raise RuntimeError(
        f"Collection {settings.collection} có dim={actual_dim}, nhưng cấu hình cần "
        f"dim={settings.embedding_dim}. Hãy dùng collection SigLIP2 mới; không nạp đè "
        "vector 768 chiều vào collection CLIP 512 chiều."
    )
print(description)

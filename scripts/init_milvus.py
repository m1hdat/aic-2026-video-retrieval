from pymilvus import MilvusClient, DataType
from src.settings import settings

c=MilvusClient(uri=settings.milvus_uri)
if not c.has_collection(settings.collection):
    schema=c.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("video_id", DataType.VARCHAR, max_length=32)
    schema.add_field("keyframe_n", DataType.INT32)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=512)
    indexes=c.prepare_index_params()
    # IVF_SQ8 nén phần index xuống 8-bit, hợp lý hơn HNSW khi máy ít dung lượng.
    # Milvus vẫn giữ source vector để đảm bảo khả năng rebuild/index consistency.
    indexes.add_index("embedding", index_type="IVF_SQ8", metric_type="COSINE", params={"nlist":4096})
    c.create_collection(settings.collection, schema=schema, index_params=indexes)
print(c.describe_collection(settings.collection))

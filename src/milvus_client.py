from __future__ import annotations

from typing import Any

import numpy as np
from pymilvus import DataType, MilvusClient

VECTOR_FIELD = "embedding"

OUTPUT_FIELDS = [
    "video_id",
    "frame_id",
    "source_part",
    "feature_row",
    "keyframe_relpath",
    "video_relpath",
    "timestamp_sec",
]


class MilvusManager:
    """Unified client for Milvus Lite and Milvus Standalone."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        milvus = config["milvus"]

        kwargs: dict[str, Any] = {
            "uri": milvus["resolved_uri"],
        }

        if milvus.get("token"):
            kwargs["token"] = milvus["token"]

        self.client = MilvusClient(**kwargs)

        self.collection_name = milvus["collection_name"]
        self.metric_type = milvus.get("metric_type", "IP")
        self.nprobe = int(milvus.get("nprobe", 16))

        if not self.collection_exists():
            raise RuntimeError(
                f"Không tìm thấy Milvus collection: "
                f"{self.collection_name}"
            )

        self.load_collection()

    def collection_exists(self) -> bool:
        return self.client.has_collection(
            collection_name=self.collection_name
        )

    def load_collection(self) -> None:
        """
        Đưa collection vào trạng thái loaded để có thể search/query.

        Milvus Lite hoặc Standalone có thể mở collection ở trạng thái
        released sau khi khởi tạo client, vì vậy cần load trước khi search.
        """
        try:
            self.client.load_collection(
                collection_name=self.collection_name
            )
        except Exception as exc:
            raise RuntimeError(
                f"Không thể load collection "
                f"'{self.collection_name}': {exc}"
            ) from exc

    def ensure_collection(self, drop_existing: bool = False) -> None:
        """
        Tạo collection nếu chưa tồn tại.

        Nếu drop_existing=True, collection cũ sẽ bị xóa và tạo lại.
        """
        if self.collection_exists():
            if not drop_existing:
                self.load_collection()
                return

            self.client.drop_collection(
                collection_name=self.collection_name
            )

        dim = int(
            self.config["model"].get(
                "embedding_dim",
                512,
            )
        )

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )

        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
        )

        schema.add_field(
            field_name=VECTOR_FIELD,
            datatype=DataType.FLOAT_VECTOR,
            dim=dim,
        )

        schema.add_field(
            field_name="video_id",
            datatype=DataType.VARCHAR,
            max_length=64,
        )

        schema.add_field(
            field_name="frame_id",
            datatype=DataType.INT64,
        )

        schema.add_field(
            field_name="source_part",
            datatype=DataType.VARCHAR,
            max_length=16,
        )

        schema.add_field(
            field_name="feature_row",
            datatype=DataType.INT64,
        )

        schema.add_field(
            field_name="keyframe_relpath",
            datatype=DataType.VARCHAR,
            max_length=1024,
        )

        schema.add_field(
            field_name="video_relpath",
            datatype=DataType.VARCHAR,
            max_length=1024,
        )

        schema.add_field(
            field_name="timestamp_sec",
            datatype=DataType.FLOAT,
        )

        params = self.client.prepare_index_params()

        index_type = self.config["milvus"].get(
            "index_type",
            "FLAT",
        )

        index_kwargs: dict[str, Any] = {
            "field_name": VECTOR_FIELD,
            "index_type": index_type,
            "metric_type": self.metric_type,
        }

        if index_type.startswith("IVF"):
            index_kwargs["params"] = {
                "nlist": int(
                    self.config["milvus"].get(
                        "nlist",
                        128,
                    )
                )
            }

        params.add_index(**index_kwargs)

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=params,
        )

        self.load_collection()

    def insert(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        if not records:
            return

        if not self.collection_exists():
            raise RuntimeError(
                f"Collection '{self.collection_name}' không tồn tại."
            )

        self.client.insert(
            collection_name=self.collection_name,
            data=records,
        )

    def search(
        self,
        query_vector: np.ndarray | list[float],
        top_k: int = 100,
        filter_expr: str = "",
    ) -> list[dict[str, Any]]:
        if not self.collection_exists():
            raise RuntimeError(
                f"Collection '{self.collection_name}' không tồn tại."
            )

        # Đảm bảo collection luôn được load trước khi search.
        self.load_collection()

        vector = np.asarray(
            query_vector,
            dtype=np.float32,
        ).reshape(-1)

        expected_dim = int(
            self.config["model"].get(
                "embedding_dim",
                512,
            )
        )

        if vector.size != expected_dim:
            raise ValueError(
                f"Query vector có dimension={vector.size}, "
                f"mong đợi dimension={expected_dim}."
            )

        norm = float(np.linalg.norm(vector))

        if norm <= 0:
            raise ValueError(
                "Query vector có norm bằng 0, không thể search."
            )

        vector = vector / norm

        search_params: dict[str, Any] = {
            "metric_type": self.metric_type,
            "params": {},
        }

        index_type = self.config["milvus"].get(
            "index_type",
            "FLAT",
        )

        if index_type.startswith("IVF"):
            search_params["params"]["nprobe"] = self.nprobe

        response = self.client.search(
            collection_name=self.collection_name,
            data=[vector.tolist()],
            anns_field=VECTOR_FIELD,
            limit=int(top_k),
            filter=filter_expr,
            search_params=search_params,
            output_fields=OUTPUT_FIELDS,
        )

        if not response:
            return []

        results: list[dict[str, Any]] = []

        for rank, hit in enumerate(
            response[0],
            start=1,
        ):
            entity = hit.get("entity") or {}

            results.append(
                {
                    "rank": rank,
                    "id": int(hit["id"]),
                    "score": float(
                        hit.get(
                            "distance",
                            hit.get("score", 0.0),
                        )
                    ),
                    **entity,
                }
            )

        return results

    def count(self) -> int | None:
        if not self.collection_exists():
            return 0

        try:
            self.load_collection()

            result = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["count(*)"],
            )

            if not result:
                return 0

            return int(
                result[0].get(
                    "count(*)",
                    0,
                )
            )

        except Exception:
            return None

    def drop(self) -> None:
        if self.collection_exists():
            self.client.drop_collection(
                collection_name=self.collection_name
            )


def connect_milvus(
    config: dict[str, Any],
) -> MilvusManager:
    return MilvusManager(config)
from __future__ import annotations

from mutoracle.retrieval import Passage
from mutoracle.storage.faiss_index import FaissIndex


class ToyEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    float("mutoracle" in lower),
                    float("retrieval" in lower),
                    float("provider" in lower),
                ]
            )
        return vectors


def test_faiss_index_orders_by_embedding_similarity() -> None:
    passages = [
        Passage(id="provider", title="Provider", text="OpenRouter provider"),
        Passage(id="retrieval", title="Retrieval", text="retrieval index"),
    ]
    index = FaissIndex(passages, ToyEmbedder())

    hits = index.search("retrieval", top_k=1)

    assert hits[0].passage.id == "retrieval"

from __future__ import annotations

from mutoracle.retrieval import LexicalRetriever, Passage, load_corpus


def test_fixture_corpus_loads() -> None:
    corpus = load_corpus()

    assert len(corpus) >= 3
    assert {passage.id for passage in corpus} >= {"mutoracle-purpose", "rag-definition"}


def test_lexical_retriever_is_deterministic() -> None:
    corpus = [
        Passage(id="b", title="Beta", text="pipeline mutation retrieval"),
        Passage(id="a", title="Alpha", text="pipeline mutation retrieval"),
        Passage(id="c", title="Gamma", text="unrelated text"),
    ]
    retriever = LexicalRetriever(corpus)

    first = retriever.search("mutation retrieval", top_k=2)
    second = retriever.search("mutation retrieval", top_k=2)

    assert first == second
    assert [hit.passage.id for hit in first] == ["a", "b"]

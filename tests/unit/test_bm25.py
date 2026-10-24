"""BM25 и поиск in-memory VectorMemory (закрытие расхождения с отчётом 6-го семестра)."""

from atomic.memory.bm25 import BM25, tokenize
from atomic.memory.vector import VectorMemory

DOCS = [
    "steam generator tube rupture with primary to secondary leakage",
    "loss of offsite power and diesel generator start",
    "feedwater pipe rupture in turbine building",
]


def test_tokenize_words_and_bigrams():
    t = tokenize("Steam Generator, tube!")
    assert t == ["steam", "generator", "tube", "steam_generator", "generator_tube"]
    assert tokenize("a b", bigrams=False) == []  # слова короче 2 символов отбрасываются


def test_ranking_and_zero_scores_excluded():
    idx = BM25(DOCS)
    top = idx.top("steam generator tube rupture", k=3)
    assert top[0][0] == 0
    assert all(s > 0 for _, s in top)
    assert idx.top("reactivity boron", k=3) == []


def test_bigram_phrase_beats_scattered_words():
    idx = BM25(["generator steam pump tube", "steam generator tube"])
    assert idx.top("steam generator", k=1)[0][0] == 1


def test_matched_terms():
    idx = BM25(DOCS)
    assert "pipe_rupture" in idx.matched_terms("feedwater pipe rupture", 2)


def test_vector_memory_ids_and_bm25_search():
    vm = VectorMemory()
    vm.add("atomic — AI-агентная система с роутингом.")
    vm.add("CommandAgent выполняет код и действия.")
    res = vm.search("найди информацию об atomic")
    assert res and res[0]["id"] == "mem:1" and "atomic" in res[0]["content"]
    assert vm.has_id("mem:2") and not vm.has_id("mem:3")
    assert VectorMemory().search("что угодно") == []

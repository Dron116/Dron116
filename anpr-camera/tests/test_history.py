from pathlib import Path

from app.history import HistoryStore


def test_history_add_and_dedup(tmp_path: Path):
    store = HistoryStore(tmp_path / "d.db")
    first = store.add("А123ВС777", "А 123 ВС 777", 0.9, "A123BC777", dedup_seconds=30)
    second = store.add("А123ВС777", "А 123 ВС 777", 0.8, "A123BC777", dedup_seconds=30)
    other = store.add("К456ЕМ199", "К 456 ЕМ 199", 0.7, "K456EM199", dedup_seconds=30)

    assert first is not None
    assert second is None
    assert other is not None
    items = store.list()
    assert len(items) == 2
    stats = store.stats()
    assert stats["total"] == 2
    assert stats["unique"] == 2


def test_history_search(tmp_path: Path):
    store = HistoryStore(tmp_path / "d.db")
    store.add("А123ВС777", "А 123 ВС 777", 0.9, "", dedup_seconds=0)
    store.add("К456ЕМ199", "К 456 ЕМ 199", 0.9, "", dedup_seconds=0)
    found = store.list(query="123")
    assert len(found) == 1
    assert found[0]["plate"] == "А123ВС777"

import app
from index_store import DismissedPairs


def make_pair(tmp_path):
    a, b = str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")
    for path in (a, b):
        with open(path, "wb") as handle:
            handle.write(b"x")
    return a, b


def test_dismissed_pair_is_hidden_in_either_order_until_restored(tmp_path):
    a, b = make_pair(tmp_path)
    store = DismissedPairs(str(tmp_path / "db"))
    pairs = [(0.99, a, b)]

    store.dismiss(b, a)
    assert store.is_dismissed(a, b)
    assert store.undismissed(pairs) == []

    store.restore(a, b)
    assert store.undismissed(pairs) == pairs


def test_changed_file_brings_dismissed_pair_back(tmp_path):
    a, b = make_pair(tmp_path)
    store = DismissedPairs(str(tmp_path / "db"))
    store.dismiss(a, b)

    with open(b, "ab") as handle:
        handle.write(b"edited")

    assert not store.is_dismissed(a, b)
    assert store.undismissed([(0.99, a, b)]) == [(0.99, a, b)]


def test_not_duplicates_button_hides_pair_unless_showing_dismissed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(app, "DB_ROOT", str(tmp_path / "db"))
    a, b = make_pair(tmp_path)
    c = str(tmp_path / "c.jpg")
    with open(c, "wb") as handle:
        handle.write(b"x")
    pairs = [(0.99, a, b), (0.98, a, c)]

    shown = app.handle_toggle_dismissed(0, pairs, [], "", True, 10)
    assert shown[1] == pairs
    assert shown[-1].value == "Restore Pair"

    app.handle_toggle_dismissed(0, pairs, [], "", True, 10)  # restore
    hidden = app.handle_toggle_dismissed(0, pairs, [], "", False, 10)
    assert hidden[1] == [(0.98, a, c)]
    assert [path for path, _ in hidden[0]] == [a, c]

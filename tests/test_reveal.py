from index_store import reveal_command, reveal_in_file_manager


def test_reveal_command_per_platform():
    assert reveal_command("/a b/c,d.jpg", "darwin") == ["open", "-R", "/a b/c,d.jpg"]
    linux = reveal_command("/a b/c,d.jpg", "linux")
    assert linux[0] == "dbus-send"
    assert "array:string:file:///a%20b/c%2Cd.jpg" in linux


def test_reveal_missing_file_does_not_launch(tmp_path, monkeypatch):
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: 1 / 0)
    assert reveal_in_file_manager(str(tmp_path / "missing.jpg")) is False

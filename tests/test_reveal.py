import subprocess
import sys

from index_store import reveal_command, reveal_in_file_manager


def test_reveal_command_per_platform():
    assert reveal_command("/a b/c,d.jpg", "darwin") == ["open", "-R", "/a b/c,d.jpg"]
    linux = reveal_command("/a b/c,d.jpg", "linux")
    assert linux[0] == "dbus-send"
    assert "array:string:file:///a%20b/c%2Cd.jpg" in linux


def test_reveal_missing_file_does_not_launch(tmp_path, monkeypatch):
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: 1 / 0)
    assert reveal_in_file_manager(str(tmp_path / "missing.jpg")) is False


class FakePopen:
    launched = []
    exit_code = 0

    def __init__(self, command):
        self.launched.append(command)

    def wait(self, timeout):
        return self.exit_code


def reveal_on_linux(tmp_path, monkeypatch, dbus_stderr, xdg_exit_code=0):
    path = tmp_path / "a.jpg"
    path.touch()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, b"", dbus_stderr),
    )
    popen = type("Popen", (FakePopen,), {"launched": [], "exit_code": xdg_exit_code})
    monkeypatch.setattr(subprocess, "Popen", popen)
    return reveal_in_file_manager(str(path)), popen.launched


def test_slow_file_manager_is_not_opened_twice(tmp_path, monkeypatch):
    stderr = b"Error org.freedesktop.DBus.Error.NoReply: Did not receive a reply."
    assert reveal_on_linux(tmp_path, monkeypatch, stderr) == (True, [])


def test_failed_xdg_open_reports_failure(tmp_path, monkeypatch):
    stderr = b"Error org.freedesktop.DBus.Error.ServiceUnknown: not provided"
    ok, launched = reveal_on_linux(tmp_path, monkeypatch, stderr, xdg_exit_code=3)
    assert not ok and launched == [["xdg-open", str(tmp_path)]]

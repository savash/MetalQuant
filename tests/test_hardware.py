from __future__ import annotations

import subprocess

from metalquant.hardware import detect_hardware


def test_detect_hardware_collects_shell_command_output(monkeypatch) -> None:
    calls = []

    def fake_check_output(command, text):
        calls.append((command, text))
        if command[-1] == "sw_vers":
            return "ProductName: macOS\n"
        return "Darwin test-host\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    info = detect_hardware()

    assert info["sw_vers"] == "ProductName: macOS"
    assert info["uname"] == "Darwin test-host"
    assert "platform" in info
    assert "machine" in info
    assert calls == [
        (["/bin/zsh", "-lc", "sw_vers"], True),
        (["/bin/zsh", "-lc", "uname -a"], True),
    ]


def test_detect_hardware_records_command_failures(monkeypatch) -> None:
    def fake_check_output(command, text):
        raise subprocess.CalledProcessError(returncode=1, cmd=command, output="boom")

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    info = detect_hardware()

    assert info["sw_vers"].startswith("unavailable:")
    assert info["uname"].startswith("unavailable:")

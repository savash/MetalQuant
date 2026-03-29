import platform
import subprocess


def detect_hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    commands = [
        ("sw_vers", "sw_vers"),
        ("uname -a", "uname"),
    ]

    for command, key in commands:
        try:
            output = subprocess.check_output(["/bin/zsh", "-lc", command], text=True).strip()
            info[key] = output
        except Exception as exc:
            info[key] = f"unavailable: {exc}"

    return info

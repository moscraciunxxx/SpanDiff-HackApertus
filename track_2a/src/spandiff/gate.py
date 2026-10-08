"""Read config.json only. A refusal here means weights did not load."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from spandiff import MODEL_ID


def _token() -> str:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    path = Path.home() / ".cache" / "huggingface" / "token"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return ""


def config_is_apertus(payload: bytes) -> bool:
    """True only for a 32-layer Apertus text config. An HTML page is not one."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return False
    from spandiff.score import _config_value, hidden_state_count

    model_type = _config_value(data, "model_type")
    text_type = _config_value(_config_value(data, "text_config"), "model_type")
    if model_type != "apertus1p5" and text_type != "apertus1p5_text":
        return False
    try:
        return hidden_state_count(data) == 33
    except (RuntimeError, TypeError, ValueError):
        return False


def _cached_config_bytes() -> bytes | None:
    hub = Path.home() / ".cache/huggingface/hub/models--swiss-ai--Apertus-v1.5-8B/snapshots"
    if not hub.is_dir():
        return None
    found = sorted(hub.glob("*/config.json"))
    if not found:
        return None
    try:
        return found[-1].read_bytes()
    except OSError:
        return None


def probe_config() -> tuple[bool, str]:
    """Return (loaded, exact error). `loaded` is true only for the Apertus text config."""
    cached = _cached_config_bytes()
    if cached is not None and config_is_apertus(cached):
        return True, ""
    url = f"https://huggingface.co/{MODEL_ID}/resolve/main/config.json"
    headers = {"User-Agent": "spandiff-probe"}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
            if response.status != 200 or not config_is_apertus(body):
                return False, f"HTTP {response.status}\nconfig.json is not the Apertus text config"
            return True, ""
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", "replace").strip()
        code = exc.headers.get("x-error-code", "")
        message = exc.headers.get("x-error-message", "")
        lines = [
            f"HTTP {exc.code}",
            f"x-error-code: {code}",
            f"x-error-message: {message}",
            f"body: {payload}",
            f"url: {url}",
        ]
        if token:
            lines.append("credential: local Hugging Face token was sent; it is not in the authorized list")
        else:
            lines.append("credential: no Hugging Face token was available")
        return False, "\n".join(lines)
    except urllib.error.URLError as exc:
        return False, f"config.json was not readable\n{exc.reason}"

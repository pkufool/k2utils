#!/usr/bin/env python3
# Copyright  2026 Wei Kang (wkang@pku.edu.cn)
#
# See ../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest
from packaging.tags import Tag, sys_tags

from k2utils.commands.install import common
from k2utils.commands.install.common import InstallError

FIXTURES = Path(__file__).parent / "fixtures"
CPU_PAGE_URL = "https://k2-fsa.github.io/k2/cpu.html"


class FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _fake_urlopen(body: bytes):
    def urlopen(request, timeout=None):
        return FakeResponse(body)

    return urlopen


def test_fetch_wheel_links_extracts_https_wheels(monkeypatch):
    body = (FIXTURES / "k2_cpu.html").read_bytes()
    monkeypatch.setattr(common, "urlopen", _fake_urlopen(body))

    links = common.fetch_wheel_links(CPU_PAGE_URL)

    assert links == [
        "https://k2-fsa.github.io/k2/cpu/1.24.4.dev20260625/linux-x64/"
        "k2-1.24.4.dev20260710+cpu.torch2.8.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "https://huggingface.co/csukuangfj2/k2/resolve/main/windows-cpu/1.24.4.dev20260625/"
        "k2-1.24.4.dev20260710+cpu.torch2.8.0-cp311-cp311-win_amd64.whl",
        "https://huggingface.co/csukuangfj2/k2/resolve/main/macos/1.24.4.dev20260625/"
        "k2-1.24.4.dev20260710+cpu.torch2.7.1-cp311-cp311-macosx_11_0_arm64.whl",
        "https://example.com/k2-1.24.4.dev20260710%2Bcpu.torch2.8.0-cp311-cp311-win_amd64.whl",
    ]


def test_fetch_wheel_links_network_error(monkeypatch):
    def urlopen(request, timeout=None):
        raise URLError("connection refused")

    monkeypatch.setattr(common, "urlopen", urlopen)
    with pytest.raises(InstallError, match="Failed to fetch wheel index"):
        common.fetch_wheel_links(CPU_PAGE_URL)


def test_fetch_wheel_links_http_error(monkeypatch):
    def urlopen(request, timeout=None):
        raise HTTPError(CPU_PAGE_URL, 404, "Not Found", None, None)

    monkeypatch.setattr(common, "urlopen", urlopen)
    with pytest.raises(InstallError, match="HTTP 404"):
        common.fetch_wheel_links(CPU_PAGE_URL)


def test_parse_wheel_valid():
    url = (
        "https://example.com/k2-1.24.4.dev20260710%2Bcpu.torch2.8.0-"
        "cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    )
    wheel = common.parse_wheel(url)
    assert wheel is not None
    assert wheel.distribution == "k2"
    assert wheel.version.public == "1.24.4.dev20260710"
    assert wheel.version.local == "cpu.torch2.8.0"
    assert wheel.url == url


def test_parse_wheel_invalid_filename():
    assert common.parse_wheel("https://example.com/not-a-wheel.whl") is None


def test_is_compatible_with_current_python():
    current = frozenset({next(iter(sys_tags()))})
    assert common.is_compatible_with_current_python(current)
    assert not common.is_compatible_with_current_python(
        frozenset({Tag("cp99", "cp99", "win_nope")})
    )


def test_run_install_uses_pip_when_available(monkeypatch):
    monkeypatch.setattr(common, "find_spec", lambda name: object())
    captured = {}

    def fake_run(command):
        captured["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(common.subprocess, "run", fake_run)
    common.run_install("https://example.com/a.whl")

    command = captured["command"]
    assert command[0] == sys.executable
    assert command[1:] == [
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-cache-dir",
        "https://example.com/a.whl",
    ]


def test_run_install_uses_uv_when_pip_missing(monkeypatch):
    monkeypatch.setattr(common, "find_spec", lambda name: None)
    monkeypatch.setattr(common.shutil, "which", lambda name: "/usr/local/bin/uv")
    captured = {}

    def fake_run(command):
        captured["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(common.subprocess, "run", fake_run)
    common.run_install("https://example.com/a.whl")

    assert captured["command"] == [
        "/usr/local/bin/uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--upgrade",
        "https://example.com/a.whl",
    ]


def test_run_install_errors_without_pip_or_uv(monkeypatch):
    monkeypatch.setattr(common, "find_spec", lambda name: None)
    monkeypatch.setattr(common.shutil, "which", lambda name: None)
    with pytest.raises(InstallError, match="Neither pip nor uv"):
        common.run_install("https://example.com/a.whl")


def test_run_install_dry_run_skips_subprocess(monkeypatch):
    monkeypatch.setattr(common, "find_spec", lambda name: object())

    def fake_run(command):
        raise AssertionError("subprocess must not run in dry-run")

    monkeypatch.setattr(common.subprocess, "run", fake_run)
    common.run_install("https://example.com/a.whl", dry_run=True)


def test_run_install_failure_raises(monkeypatch):
    monkeypatch.setattr(common, "find_spec", lambda name: object())
    monkeypatch.setattr(
        common.subprocess,
        "run",
        lambda command: SimpleNamespace(returncode=3),
    )
    with pytest.raises(InstallError, match="exit code 3"):
        common.run_install("https://example.com/a.whl")

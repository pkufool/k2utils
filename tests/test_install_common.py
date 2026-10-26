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
from packaging.version import Version

from conftest import (
    current_tag,
    fake_torch_module,
    fake_verify_run,
    make_package,
    torch_info,
    verify_payload,
    wheel_url,
)
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


# ---------------------------------------------------------------- index page


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


def test_fetch_wheel_links_retries_transient_failure(monkeypatch):
    body = (FIXTURES / "k2_cpu.html").read_bytes()
    calls = []

    def urlopen(request, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise URLError("connection reset by peer")
        return FakeResponse(body)

    monkeypatch.setattr(common, "urlopen", urlopen)
    links = common.fetch_wheel_links(CPU_PAGE_URL)
    assert len(calls) == 2
    assert links[0].endswith(
        "k2-1.24.4.dev20260710+cpu.torch2.8.0-cp311-cp311-"
        "manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    )


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


# ---------------------------------------------------------------- torch build


def test_detect_torch_cuda(monkeypatch):
    monkeypatch.setattr(common, "import_module", fake_torch_module("2.8.0+cu128", cuda="12.8"))
    assert common.detect_torch() == common.TorchInfo(
        version=Version("2.8.0"), backend="cuda", cuda_version="12.8"
    )


def test_detect_torch_cpu(monkeypatch):
    monkeypatch.setattr(common, "import_module", fake_torch_module("2.8.0+cpu"))
    assert common.detect_torch() == common.TorchInfo(
        version=Version("2.8.0"), backend="cpu", cuda_version=None
    )


def test_detect_torch_missing(monkeypatch):
    def raise_import(name):
        raise ImportError(name)

    monkeypatch.setattr(common, "import_module", raise_import)
    with pytest.raises(InstallError, match="PyTorch is not installed"):
        common.detect_torch()


def test_detect_torch_rocm_rejected(monkeypatch):
    monkeypatch.setattr(common, "import_module", fake_torch_module("2.8.0", hip="6.4"))
    with pytest.raises(InstallError, match="ROCm"):
        common.detect_torch()


def test_detect_torch_cuda_and_hip_rejected(monkeypatch):
    monkeypatch.setattr(
        common, "import_module", fake_torch_module("2.8.0", cuda="12.8", hip="6.4")
    )
    with pytest.raises(InstallError, match="both CUDA and ROCm"):
        common.detect_torch()


def test_detect_torch_bad_version(monkeypatch):
    monkeypatch.setattr(common, "import_module", fake_torch_module("latest"))
    with pytest.raises(InstallError, match="Unrecognized PyTorch version"):
        common.detect_torch()


def test_parse_torch_version():
    assert common.parse_torch_version("2.8.0+cu128") == Version("2.8.0")
    assert common.parse_torch_version("2.8.0") == Version("2.8.0")
    with pytest.raises(InstallError):
        common.parse_torch_version("2.8")


def test_index_url_for():
    package = make_package()
    assert common.index_url_for(package, torch_info()) == package.cpu_index
    assert common.index_url_for(package, torch_info(cuda="12.8")) == package.cuda_index
    assert common.index_url_for(package, torch_info(), cn=True) == package.cpu_cn_index
    assert (
        common.index_url_for(package, torch_info(cuda="12.8"), cn=True)
        == package.cuda_cn_index
    )


# ---------------------------------------------------------------- wheel markers


def test_parse_wheel_markers_cpu():
    markers = common.parse_wheel_markers("cpu.torch2.8.0")
    assert markers == common.WheelMarkers(
        torch_version=Version("2.8.0"), backend="cpu", backend_version=None
    )


def test_parse_wheel_markers_cuda():
    markers = common.parse_wheel_markers("cuda13.2.torch2.13.0")
    assert markers == common.WheelMarkers(
        torch_version=Version("2.13.0"), backend="cuda", backend_version="13.2"
    )


@pytest.mark.parametrize(
    "local",
    [
        None,
        "",
        "torch2.8.0",
        "cpu.torch2.8.0.torch2.9.0",
        "cpu",
        "cpu.cuda12.8.torch2.8.0",
        "rocm6.4.torch2.8.0",
    ],
)
def test_parse_wheel_markers_invalid(local):
    with pytest.raises(ValueError):
        common.parse_wheel_markers(local)


# ---------------------------------------------------------------- wheel selection


def test_selects_newest_exact_match():
    links = [
        wheel_url(version="1.24.3.dev20260101", local="cpu.torch2.8.0"),
        wheel_url(version="1.24.4.dev20260710", local="cpu.torch2.8.0"),
        wheel_url(version="1.24.5.dev20260801", local="cpu.torch2.7.1"),
    ]
    best = common.select_wheel(links, torch_info(), make_package())
    assert best.version.public == "1.24.4.dev20260710"


def test_no_fallback_across_torch_versions():
    links = [wheel_url(local="cpu.torch2.7.1"), wheel_url(local="cpu.torch2.8.1")]
    with pytest.raises(InstallError, match="No pre-built k2 wheel"):
        common.select_wheel(links, torch_info(), make_package())


def test_no_fallback_across_cuda_versions():
    links = [wheel_url(local="cuda12.6.torch2.8.0"), wheel_url(local="cuda12.9.torch2.8.0")]
    with pytest.raises(InstallError, match="No pre-built k2 wheel"):
        common.select_wheel(links, torch_info(cuda="12.8"), make_package())


def test_selects_exact_cuda_version():
    links = [
        wheel_url(local="cuda12.6.torch2.8.0"),
        wheel_url(local="cuda12.8.torch2.8.0"),
    ]
    best = common.select_wheel(links, torch_info(cuda="12.8"), make_package())
    assert "cuda12.8" in best.version.local


def test_cuda_environment_rejects_cpu_wheels():
    links = [wheel_url(local="cpu.torch2.8.0")]
    with pytest.raises(InstallError, match="No pre-built k2 wheel"):
        common.select_wheel(links, torch_info(cuda="12.8"), make_package())


def test_cpu_environment_rejects_cuda_wheels():
    links = [wheel_url(local="cuda12.8.torch2.8.0")]
    with pytest.raises(InstallError, match="No pre-built k2 wheel"):
        common.select_wheel(links, torch_info(), make_package())


def test_rejects_incompatible_python_platform():
    bad_tag = str(Tag("cp99", "cp99", "win_nope"))
    links = [
        wheel_url(local="cpu.torch2.8.0", tag=bad_tag),
        wheel_url(local="cpu.torch2.8.0"),
    ]
    best = common.select_wheel(links, torch_info(), make_package())
    assert "win_nope" not in best.url


def test_skips_other_distributions_and_malformed_names():
    links = [
        wheel_url(name="kaldifeat", version="1.25.0", local="cpu.torch2.8.0"),
        f"https://example.com/not-a-wheel.whl",
        wheel_url(),
    ]
    best = common.select_wheel(links, torch_info(), make_package())
    assert best.distribution == "k2"


def test_exact_match_message_mentions_environment():
    with pytest.raises(InstallError, match="does not fall back"):
        common.select_wheel([], torch_info(cuda="12.8"), make_package())


# ---------------------------------------------------------------- installer


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


# ---------------------------------------------------------------- verification


def test_verify_install_ok(monkeypatch):
    monkeypatch.setattr(common.subprocess, "run", fake_verify_run(verify_payload()))
    common.verify_install("k2", torch_info())


def test_verify_install_import_failure(monkeypatch):
    monkeypatch.setattr(
        common.subprocess, "run", fake_verify_run("", returncode=1, stderr="ImportError: boom")
    )
    with pytest.raises(InstallError, match="importing it failed"):
        common.verify_install("k2", torch_info())


def test_verify_install_torch_version_changed(monkeypatch):
    stdout = verify_payload(torch_version="2.7.1")
    monkeypatch.setattr(common.subprocess, "run", fake_verify_run(stdout))
    with pytest.raises(InstallError, match="version changed"):
        common.verify_install("k2", torch_info())


def test_verify_install_cuda_mismatch(monkeypatch):
    stdout = verify_payload(torch_cuda="12.6")
    monkeypatch.setattr(common.subprocess, "run", fake_verify_run(stdout))
    with pytest.raises(InstallError, match="CUDA runtime changed"):
        common.verify_install("k2", torch_info(cuda="12.8"))

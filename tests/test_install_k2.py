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

from types import SimpleNamespace

import pytest
from packaging.tags import Tag, sys_tags
from packaging.version import Version

from k2utils.commands.install import k2 as k2_install
from k2utils.commands.install.common import InstallError

HOST = "https://example.com"


def current_tag() -> str:
    return str(next(iter(sys_tags())))


def wheel_url(
    version: str = "1.24.4.dev20260710",
    local: str = "cpu.torch2.8.0",
    tag: str = None,
    name: str = "k2",
) -> str:
    tag = tag or current_tag()
    return f"{HOST}/{name}-{version}+{local}-{tag}.whl"


def cpu_torch(version: str = "2.8.0", cuda=None, hip=None):
    return k2_install.TorchInfo(
        version=Version(version),
        backend="cpu" if cuda is None else "cuda",
        cuda_version=cuda,
    )


class FakeTorchVersion:
    def __init__(self, cuda=None, hip=None):
        self.cuda = cuda
        self.hip = hip


class FakeTorch:
    def __init__(self, version, cuda=None, hip=None):
        self.__version__ = version
        self.version = FakeTorchVersion(cuda=cuda, hip=hip)


def fake_torch_module(version, cuda=None, hip=None):
    return lambda name: FakeTorch(version, cuda=cuda, hip=hip)


# ---------------------------------------------------------------- torch detection


def test_detect_torch_cuda(monkeypatch):
    monkeypatch.setattr(k2_install, "import_module", fake_torch_module("2.8.0+cu128", cuda="12.8"))
    assert k2_install.detect_torch() == k2_install.TorchInfo(
        version=Version("2.8.0"), backend="cuda", cuda_version="12.8"
    )


def test_detect_torch_cpu(monkeypatch):
    monkeypatch.setattr(k2_install, "import_module", fake_torch_module("2.8.0+cpu"))
    assert k2_install.detect_torch() == k2_install.TorchInfo(
        version=Version("2.8.0"), backend="cpu", cuda_version=None
    )


def test_detect_torch_missing(monkeypatch):
    def raise_import(name):
        raise ImportError(name)

    monkeypatch.setattr(k2_install, "import_module", raise_import)
    with pytest.raises(InstallError, match="PyTorch is not installed"):
        k2_install.detect_torch()


def test_detect_torch_rocm_rejected(monkeypatch):
    monkeypatch.setattr(k2_install, "import_module", fake_torch_module("2.8.0", hip="6.4"))
    with pytest.raises(InstallError, match="ROCm"):
        k2_install.detect_torch()


def test_detect_torch_cuda_and_hip_rejected(monkeypatch):
    monkeypatch.setattr(
        k2_install, "import_module", fake_torch_module("2.8.0", cuda="12.8", hip="6.4")
    )
    with pytest.raises(InstallError, match="both CUDA and ROCm"):
        k2_install.detect_torch()


def test_detect_torch_bad_version(monkeypatch):
    monkeypatch.setattr(k2_install, "import_module", fake_torch_module("latest"))
    with pytest.raises(InstallError, match="Unrecognized PyTorch version"):
        k2_install.detect_torch()


def test_parse_torch_version():
    assert k2_install.parse_torch_version("2.8.0+cu128") == Version("2.8.0")
    assert k2_install.parse_torch_version("2.8.0") == Version("2.8.0")
    with pytest.raises(InstallError):
        k2_install.parse_torch_version("2.8")


# ---------------------------------------------------------------- wheel markers


def test_parse_k2_markers_cpu():
    markers = k2_install.parse_k2_markers("cpu.torch2.8.0")
    assert markers == k2_install.K2Markers(
        torch_version=Version("2.8.0"), backend="cpu", backend_version=None
    )


def test_parse_k2_markers_cuda():
    markers = k2_install.parse_k2_markers("cuda13.2.torch2.13.0")
    assert markers == k2_install.K2Markers(
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
def test_parse_k2_markers_invalid(local):
    with pytest.raises(ValueError):
        k2_install.parse_k2_markers(local)


def test_index_url_for():
    assert k2_install.index_url_for(cpu_torch()) == k2_install.CPU_INDEX_URL
    assert (
        k2_install.index_url_for(cpu_torch(cuda="12.8"))
        == k2_install.CUDA_INDEX_URL
    )
    assert k2_install.index_url_for(cpu_torch(), cn=True) == k2_install.CPU_CN_INDEX_URL
    assert (
        k2_install.index_url_for(cpu_torch(cuda="12.8"), cn=True)
        == k2_install.CUDA_CN_INDEX_URL
    )


# ---------------------------------------------------------------- wheel selection


def test_selects_newest_exact_match():
    links = [
        wheel_url(version="1.24.3.dev20260101", local="cpu.torch2.8.0"),
        wheel_url(version="1.24.4.dev20260710", local="cpu.torch2.8.0"),
        wheel_url(version="1.24.5.dev20260801", local="cpu.torch2.7.1"),
    ]
    best = k2_install.find_best_wheel(links, cpu_torch())
    assert best.version.public == "1.24.4.dev20260710"


def test_no_fallback_across_torch_versions():
    links = [wheel_url(local="cpu.torch2.7.1"), wheel_url(local="cpu.torch2.8.1")]
    with pytest.raises(InstallError, match="No pre-built k2 wheel"):
        k2_install.find_best_wheel(links, cpu_torch())


def test_no_fallback_across_cuda_versions():
    links = [wheel_url(local="cuda12.6.torch2.8.0"), wheel_url(local="cuda12.9.torch2.8.0")]
    with pytest.raises(InstallError, match="No pre-built k2 wheel"):
        k2_install.find_best_wheel(links, cpu_torch(cuda="12.8"))


def test_selects_exact_cuda_version():
    links = [
        wheel_url(local="cuda12.6.torch2.8.0"),
        wheel_url(local="cuda12.8.torch2.8.0"),
    ]
    best = k2_install.find_best_wheel(links, cpu_torch(cuda="12.8"))
    assert "cuda12.8" in best.version.local


def test_cuda_environment_rejects_cpu_wheels():
    links = [wheel_url(local="cpu.torch2.8.0")]
    with pytest.raises(InstallError, match="No pre-built k2 wheel"):
        k2_install.find_best_wheel(links, cpu_torch(cuda="12.8"))


def test_cpu_environment_rejects_cuda_wheels():
    links = [wheel_url(local="cuda12.8.torch2.8.0")]
    with pytest.raises(InstallError, match="No pre-built k2 wheel"):
        k2_install.find_best_wheel(links, cpu_torch())


def test_rejects_incompatible_python_platform():
    bad_tag = str(Tag("cp99", "cp99", "win_nope"))
    links = [
        wheel_url(local="cpu.torch2.8.0", tag=bad_tag),
        wheel_url(local="cpu.torch2.8.0"),
    ]
    best = k2_install.find_best_wheel(links, cpu_torch())
    assert "win_nope" not in best.url


def test_skips_other_distributions_and_malformed_names():
    links = [
        wheel_url(name="kaldifeat", version="1.25.0", local="cpu.torch2.8.0"),
        f"{HOST}/not-a-wheel.whl",
        wheel_url(),
    ]
    best = k2_install.find_best_wheel(links, cpu_torch())
    assert best.distribution == "k2"


def test_exact_match_message_mentions_environment():
    with pytest.raises(InstallError, match="does not fall back"):
        k2_install.find_best_wheel([], cpu_torch(cuda="12.8"))


# ---------------------------------------------------------------- install flow


def run_install(monkeypatch, **kwargs):
    fake_torch = cpu_torch()
    monkeypatch.setattr(k2_install, "detect_torch", lambda: fake_torch)
    url = wheel_url(local="cpu.torch2.8.0")
    recorded = {}

    def fake_fetch(index_url):
        recorded["index_url"] = index_url
        return [url]

    monkeypatch.setattr(k2_install, "fetch_wheel_links", fake_fetch)

    def fake_pip(wheel_url, dry_run=False):
        recorded["url"] = wheel_url
        recorded["dry_run"] = dry_run

    def fake_verify(torch):
        recorded["verified"] = True

    monkeypatch.setattr(k2_install, "run_install", fake_pip)
    monkeypatch.setattr(k2_install, "verify_install", fake_verify)
    rc = k2_install.install_k2(**kwargs)
    return rc, recorded


def test_install_k2_dry_run(monkeypatch, capsys):
    rc, recorded = run_install(monkeypatch, dry_run=True)
    assert rc == 0
    assert recorded["dry_run"] is True
    assert "verified" not in recorded
    out = capsys.readouterr().out
    assert "Selected:" in out
    assert recorded["url"] in out


def test_install_k2_verifies_after_install(monkeypatch):
    rc, recorded = run_install(monkeypatch, dry_run=False)
    assert rc == 0
    assert recorded["dry_run"] is False
    assert recorded["verified"] is True


def test_install_k2_cn_uses_cn_index(monkeypatch):
    rc, recorded = run_install(monkeypatch, dry_run=True, cn=True)
    assert rc == 0
    assert recorded["index_url"] == k2_install.CPU_CN_INDEX_URL


def test_install_k2_default_uses_main_index(monkeypatch):
    rc, recorded = run_install(monkeypatch, dry_run=True)
    assert rc == 0
    assert recorded["index_url"] == k2_install.CPU_INDEX_URL


def test_install_k2_rejects_torch_version_mismatch(monkeypatch):
    monkeypatch.setattr(k2_install, "detect_torch", lambda: cpu_torch("2.8.0"))

    def unexpected_fetch(index_url):
        raise AssertionError("must fail before any network request")

    monkeypatch.setattr(k2_install, "fetch_wheel_links", unexpected_fetch)
    with pytest.raises(InstallError, match="does not match the installed"):
        k2_install.install_k2(torch_version="2.7.0")


# ---------------------------------------------------------------- verification


def fake_verify_run(stdout, returncode=0, stderr=""):
    def run(command, **kwargs):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return run


VERIFY_OK = (
    '__K2UTILS_VERIFY__{"k2_version": "1.24.4", '
    '"torch_version": "2.8.0", "torch_cuda": null}\n'
)


def test_verify_install_ok(monkeypatch):
    monkeypatch.setattr(
        k2_install.subprocess, "run", fake_verify_run(VERIFY_OK)
    )
    k2_install.verify_install(cpu_torch())


def test_verify_install_import_failure(monkeypatch):
    monkeypatch.setattr(
        k2_install.subprocess, "run", fake_verify_run("", returncode=1, stderr="ImportError: boom")
    )
    with pytest.raises(InstallError, match="importing it failed"):
        k2_install.verify_install(cpu_torch())


def test_verify_install_torch_version_changed(monkeypatch):
    stdout = (
        '__K2UTILS_VERIFY__{"k2_version": "1.24.4", '
        '"torch_version": "2.7.1", "torch_cuda": null}\n'
    )
    monkeypatch.setattr(k2_install.subprocess, "run", fake_verify_run(stdout))
    with pytest.raises(InstallError, match="version changed"):
        k2_install.verify_install(cpu_torch())


def test_verify_install_cuda_mismatch(monkeypatch):
    stdout = (
        '__K2UTILS_VERIFY__{"k2_version": "1.24.4", '
        '"torch_version": "2.8.0", "torch_cuda": "12.6"}\n'
    )
    monkeypatch.setattr(k2_install.subprocess, "run", fake_verify_run(stdout))
    with pytest.raises(InstallError, match="CUDA runtime changed"):
        k2_install.verify_install(cpu_torch(cuda="12.8"))

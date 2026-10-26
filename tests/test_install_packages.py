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

"""Per-package wiring for k2 and kaldifeat (index URLs and install entry)."""

import pytest

from conftest import torch_info, wheel_url
from k2utils.commands.install import common
from k2utils.commands.install import k2 as k2_install
from k2utils.commands.install import kaldifeat as kaldifeat_install
from k2utils.commands.install.common import InstallError

CASES = [
    pytest.param(k2_install.install_k2, k2_install.K2, id="k2"),
    pytest.param(
        kaldifeat_install.install_kaldifeat, kaldifeat_install.KALDIFEAT, id="kaldifeat"
    ),
]


@pytest.mark.parametrize("entry, package", CASES)
def test_index_urls(entry, package):
    assert package.cpu_index.endswith("/cpu.html")
    assert package.cuda_index.endswith("/cuda.html")
    assert package.cpu_cn_index.endswith("/cpu-cn.html")
    assert package.cuda_cn_index.endswith("/cuda-cn.html")
    assert package.source_url.startswith("https://")
    assert package.name in package.cpu_index


def test_real_index_urls():
    assert k2_install.K2.cpu_index == "https://k2-fsa.github.io/k2/cpu.html"
    assert k2_install.K2.cuda_index == "https://k2-fsa.github.io/k2/cuda.html"
    assert k2_install.K2.cpu_cn_index == "https://k2-fsa.github.io/k2/cpu-cn.html"
    assert k2_install.K2.cuda_cn_index == "https://k2-fsa.github.io/k2/cuda-cn.html"
    assert (
        kaldifeat_install.KALDIFEAT.cpu_index
        == "https://csukuangfj.github.io/kaldifeat/cpu.html"
    )
    assert (
        kaldifeat_install.KALDIFEAT.cuda_index
        == "https://csukuangfj.github.io/kaldifeat/cuda.html"
    )
    assert (
        kaldifeat_install.KALDIFEAT.cpu_cn_index
        == "https://csukuangfj.github.io/kaldifeat/cpu-cn.html"
    )
    assert (
        kaldifeat_install.KALDIFEAT.cuda_cn_index
        == "https://csukuangfj.github.io/kaldifeat/cuda-cn.html"
    )


def run_install(monkeypatch, entry, package, torch=None, **kwargs):
    torch = torch or torch_info()
    monkeypatch.setattr(common, "detect_torch", lambda: torch)
    url = wheel_url(
        local="cpu.torch2.8.0" if torch.backend == "cpu" else "cuda12.8.torch2.8.0",
        name=package.name,
    )
    recorded = {}

    def fake_fetch(index_url):
        recorded["index_url"] = index_url
        # a foreign distribution on the page must not be selected
        return [wheel_url(name="some-other-package"), url]

    def fake_pip(wheel_url, dry_run=False):
        recorded["url"] = wheel_url
        recorded["dry_run"] = dry_run

    def fake_verify(package_name, torch_arg):
        recorded["verified"] = True
        recorded["verified_package"] = package_name

    monkeypatch.setattr(common, "fetch_wheel_links", fake_fetch)
    monkeypatch.setattr(common, "run_install", fake_pip)
    monkeypatch.setattr(common, "verify_install", fake_verify)
    rc = entry(**kwargs)
    return rc, recorded


@pytest.mark.parametrize("entry, package", CASES)
def test_install_dry_run(monkeypatch, capsys, entry, package):
    rc, recorded = run_install(monkeypatch, entry, package, dry_run=True)
    assert rc == 0
    assert recorded["dry_run"] is True
    assert "verified" not in recorded
    assert f"{package.name}-" in recorded["url"]
    out = capsys.readouterr().out
    assert "Selected:" in out
    assert recorded["url"] in out


@pytest.mark.parametrize("entry, package", CASES)
def test_install_verifies_after_install(monkeypatch, entry, package):
    rc, recorded = run_install(monkeypatch, entry, package, dry_run=False)
    assert rc == 0
    assert recorded["dry_run"] is False
    assert recorded["verified"] is True
    assert recorded["verified_package"] == package.name


@pytest.mark.parametrize("entry, package", CASES)
def test_install_default_uses_cpu_index(monkeypatch, entry, package):
    rc, recorded = run_install(monkeypatch, entry, package, dry_run=True)
    assert rc == 0
    assert recorded["index_url"] == package.cpu_index


@pytest.mark.parametrize("entry, package", CASES)
def test_install_cn_uses_cn_index(monkeypatch, entry, package):
    rc, recorded = run_install(monkeypatch, entry, package, dry_run=True, cn=True)
    assert rc == 0
    assert recorded["index_url"] == package.cpu_cn_index


@pytest.mark.parametrize("entry, package", CASES)
def test_install_cuda_env_uses_cuda_index(monkeypatch, entry, package):
    rc, recorded = run_install(
        monkeypatch, entry, package, torch=torch_info(cuda="12.8"), dry_run=True
    )
    assert rc == 0
    assert recorded["index_url"] == package.cuda_index
    assert "cuda12.8" in recorded["url"]


@pytest.mark.parametrize("entry, package", CASES)
def test_install_rejects_torch_version_mismatch(monkeypatch, entry, package):
    monkeypatch.setattr(common, "detect_torch", lambda: torch_info("2.8.0"))

    def unexpected_fetch(index_url):
        raise AssertionError("must fail before any network request")

    monkeypatch.setattr(common, "fetch_wheel_links", unexpected_fetch)
    with pytest.raises(InstallError, match="does not match the installed"):
        entry(torch_version="2.7.0")

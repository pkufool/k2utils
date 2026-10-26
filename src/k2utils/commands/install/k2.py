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

"""Install k2 from its official pre-built wheel pages.

Index pages (flat lists of wheel links):

- CPU:  https://k2-fsa.github.io/k2/cpu.html
- CUDA: https://k2-fsa.github.io/k2/cuda.html

``--cn`` switches to the cpu-cn.html / cuda-cn.html pages, whose wheel links
go through the hf-mirror.com proxy for users who cannot reach Hugging Face.

Wheel filenames carry the build metadata that pip cannot match for us, e.g.::

    k2-1.24.4.dev20260710+cpu.torch2.13.0-cp315-cp315-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
    k2-1.24.4.dev20260710+cuda13.2.torch2.13.0-cp315-cp315-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

Only a wheel whose torch version, CPU/CUDA backend and tags all match the
running environment exactly is ever installed; there is no fallback to a
"close enough" wheel.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module
from typing import List, Optional

from packaging.version import Version

from k2utils.commands.install.common import (
    InstallError,
    Wheel,
    fetch_wheel_links,
    is_compatible_with_current_python,
    parse_wheel,
    run_install,
)

CPU_INDEX_URL = "https://k2-fsa.github.io/k2/cpu.html"
CUDA_INDEX_URL = "https://k2-fsa.github.io/k2/cuda.html"
CPU_CN_INDEX_URL = "https://k2-fsa.github.io/k2/cpu-cn.html"
CUDA_CN_INDEX_URL = "https://k2-fsa.github.io/k2/cuda-cn.html"

_TORCH_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
_TORCH_MARKER_RE = re.compile(r"torch(\d+\.\d+\.\d+)")
_CUDA_MARKER_RE = re.compile(r"cuda(\d+\.\d+)")
_ROCM_MARKER_RE = re.compile(r"rocm(\d+\.\d+)")
_CPU_MARKER_RE = re.compile(r"(^|\.)cpu(\.|$)")

_VERIFY_CODE = """\
import json
import torch
import k2

print("__K2UTILS_VERIFY__" + json.dumps({
    "k2_version": str(getattr(k2, "__version__", "unknown")),
    "torch_version": torch.__version__,
    "torch_cuda": getattr(torch.version, "cuda", None),
}))
"""


@dataclass(frozen=True)
class TorchInfo:
    version: Version  # e.g. 2.13.0
    backend: str  # "cpu" or "cuda"
    cuda_version: Optional[str]  # e.g. "13.2" for a CUDA build


@dataclass(frozen=True)
class K2Markers:
    torch_version: Version
    backend: str  # "cpu" or "cuda"
    backend_version: Optional[str]  # e.g. "13.2" for a CUDA wheel


def parse_torch_version(value: str) -> Version:
    """Extract the three-part torch release from e.g. "2.13.0+cu128"."""
    match = _TORCH_VERSION_RE.search(value)
    if match is None:
        raise InstallError(f"Unrecognized PyTorch version: {value!r}")
    return Version(match.group())


def _major_minor(value: str) -> str:
    match = re.search(r"\d+\.\d+", value)
    if match is None:
        raise InstallError(f"Unrecognized CUDA version: {value!r}")
    return match.group()


def detect_torch() -> TorchInfo:
    try:
        torch = import_module("torch")
    except ImportError:
        raise InstallError(
            "PyTorch is not installed in this Python environment. "
            "Install PyTorch first, then run k2utils again."
        )

    raw_version = getattr(torch, "__version__", "")
    version = parse_torch_version(raw_version)

    torch_version_module = getattr(torch, "version", None)
    cuda = getattr(torch_version_module, "cuda", None)
    hip = getattr(torch_version_module, "hip", None)

    if cuda and hip:
        raise InstallError(
            "This PyTorch build reports both CUDA and ROCm runtimes; "
            "k2utils cannot tell which one it uses."
        )
    if hip:
        raise InstallError(
            "This is a ROCm build of PyTorch; k2utils install k2 currently "
            "supports CPU and CUDA wheels only."
        )
    if cuda:
        return TorchInfo(version=version, backend="cuda", cuda_version=_major_minor(cuda))
    return TorchInfo(version=version, backend="cpu", cuda_version=None)


def index_url_for(torch: TorchInfo, cn: bool = False) -> str:
    if torch.backend == "cuda":
        return CUDA_CN_INDEX_URL if cn else CUDA_INDEX_URL
    return CPU_CN_INDEX_URL if cn else CPU_INDEX_URL


def parse_k2_markers(local: Optional[str]) -> K2Markers:
    """Parse the k2 wheel local version, e.g. "cuda13.2.torch2.13.0"."""
    if not local:
        raise ValueError("missing local build metadata")

    torch_matches = _TORCH_MARKER_RE.findall(local)
    if len(torch_matches) != 1:
        raise ValueError(f"expected exactly one torch marker in {local!r}")
    torch_version = Version(torch_matches[0])

    cuda_matches = _CUDA_MARKER_RE.findall(local)
    rocm_matches = _ROCM_MARKER_RE.findall(local)
    is_cpu = _CPU_MARKER_RE.search(local) is not None

    if len(cuda_matches) + len(rocm_matches) > 1:
        raise ValueError(f"multiple backend markers in {local!r}")

    if cuda_matches:
        if is_cpu:
            raise ValueError(f"both cpu and cuda markers in {local!r}")
        return K2Markers(
            torch_version=torch_version, backend="cuda", backend_version=cuda_matches[0]
        )
    if rocm_matches:
        raise ValueError("ROCm wheels are not supported")
    if not is_cpu:
        raise ValueError(f"no cpu/cuda marker in {local!r}")
    return K2Markers(torch_version=torch_version, backend="cpu", backend_version=None)


def find_best_wheel(links: List[str], torch: TorchInfo) -> Wheel:
    """Pick the newest k2 wheel that matches the environment exactly."""
    compatible = []
    for url in links:
        wheel = parse_wheel(url)
        if wheel is None or wheel.distribution != "k2":
            continue
        try:
            markers = parse_k2_markers(wheel.version.local)
        except ValueError:
            continue
        if markers.torch_version != torch.version:
            continue
        if markers.backend != torch.backend:
            continue
        if torch.backend == "cuda" and markers.backend_version != torch.cuda_version:
            continue
        if not is_compatible_with_current_python(wheel.tags):
            continue
        compatible.append(wheel)

    if not compatible:
        cuda = f" CUDA {torch.cuda_version}" if torch.cuda_version else ""
        raise InstallError(
            f"No pre-built k2 wheel matches PyTorch {torch.version}{cuda} "
            f"on {platform.system()} {platform.machine()} with "
            f"{platform.python_implementation()} {platform.python_version()}. "
            "k2utils does not fall back to a wheel built for another torch, "
            "CUDA, Python or platform; see https://k2-fsa.github.io/k2/installation/from_source.html"
        )
    return max(compatible, key=lambda w: (w.version, w.url))


def verify_install(torch: TorchInfo) -> None:
    """Import k2 in a fresh interpreter and check the torch runtime matches."""
    print("Verification:")
    result = subprocess.run(
        [sys.executable, "-c", _VERIFY_CODE],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise InstallError(
            "k2 was installed but importing it failed:\n" + (result.stderr.strip() or "unknown error")
        )

    payload = None
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("__K2UTILS_VERIFY__"):
            payload = json.loads(line[len("__K2UTILS_VERIFY__"):])
            break
    if payload is None:
        raise InstallError("k2 verification produced no result")

    if parse_torch_version(payload["torch_version"]) != torch.version:
        raise InstallError(
            f"PyTorch version changed from {torch.version} to {payload['torch_version']} "
            "during installation"
        )
    cuda = payload["torch_cuda"]
    if torch.backend == "cuda":
        if not cuda or _major_minor(cuda) != torch.cuda_version:
            raise InstallError(
                f"PyTorch CUDA runtime changed from {torch.cuda_version} to {cuda} "
                "during installation"
            )
    elif cuda:
        raise InstallError(
            f"Expected a CPU PyTorch build but got CUDA {cuda} after installation"
        )

    print(f"  k2: {payload['k2_version']}")
    print(f"  torch: {payload['torch_version']}")


def install_k2(
    *, dry_run: bool = False, torch_version: Optional[str] = None, cn: bool = False
) -> int:
    torch = detect_torch()
    if torch_version is not None:
        requested = parse_torch_version(torch_version)
        if requested != torch.version:
            raise InstallError(
                f"--torch-version {torch_version} does not match the installed "
                f"PyTorch {torch.version}; k2utils only installs k2 for the "
                "torch already present in this environment."
            )

    cuda = f" CUDA {torch.cuda_version}" if torch.cuda_version else ""
    print("Environment:")
    print(f"  OS: {platform.system()} {platform.machine()}")
    print(f"  Python: {platform.python_implementation()} {platform.python_version()}")
    print(f"  PyTorch: {torch.version}")
    print(f"  Backend: {torch.backend}{cuda}")

    index_url = index_url_for(torch, cn=cn)
    print("Resolving:")
    print(f"  Index: {index_url}")
    wheel = find_best_wheel(fetch_wheel_links(index_url), torch)
    print(f"  Selected: {wheel.url}")

    run_install(wheel.url, dry_run=dry_run)
    if not dry_run:
        verify_install(torch)
    return 0

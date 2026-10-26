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

"""Shared engine for the ``k2utils install`` commands.

Every install target is a flat page of wheel links whose filenames carry
build metadata that pip cannot match for us (torch version, CPU/CUDA
backend, CUDA version). This module holds everything the targets share:
fetch the index page, parse wheel filenames, detect the running torch
build, select an exactly matching wheel, run the installer (pip when
available, otherwise uv) and verify the result. A package module
(k2.py, kaldifeat.py, ...) only declares its index URLs and calls
``install_wheels()``.
"""

from __future__ import annotations

import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from importlib import import_module
from importlib.util import find_spec
from typing import FrozenSet, List, Optional
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

from packaging.tags import Tag, sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from k2utils import __version__

DEFAULT_TIMEOUT = 60.0


class InstallError(Exception):
    """An error that can be shown to the user as a short message."""


@dataclass(frozen=True)
class Wheel:
    url: str
    distribution: str
    version: Version
    tags: FrozenSet[Tag]


@dataclass(frozen=True)
class WheelPackage:
    """Where one package's pre-built wheels are published."""

    name: str  # wheel distribution name, e.g. "k2"
    cpu_index: str
    cuda_index: str
    cpu_cn_index: str
    cuda_cn_index: str
    source_url: str  # docs for building from source


@dataclass(frozen=True)
class TorchInfo:
    version: Version  # e.g. 2.13.0
    backend: str  # "cpu" or "cuda"
    cuda_version: Optional[str]  # e.g. "13.2" for a CUDA build


@dataclass(frozen=True)
class WheelMarkers:
    torch_version: Version
    backend: str  # "cpu" or "cuda"
    backend_version: Optional[str]  # e.g. "13.2" for a CUDA wheel


# ---------------------------------------------------------------- index page


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def fetch_wheel_links(
    url: str, timeout: float = DEFAULT_TIMEOUT, attempts: int = 3
) -> List[str]:
    """Download ``url`` and return its absolute HTTPS .whl links.

    ``attempts`` retries absorb transient network failures; slow mirror links
    occasionally drop the connection mid-download.
    """
    request = Request(url, headers={"User-Agent": f"k2utils/{__version__}"})
    body = None
    last_error = None
    for _ in range(attempts):
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read()
            break
        except Exception as e:  # HTTPError/URLError/timeout are all OSError subclasses
            last_error = e
    if body is None:
        reason = getattr(last_error, "reason", last_error)
        code = getattr(last_error, "code", None)
        detail = f"HTTP {code}: {reason}" if code is not None else str(reason)
        raise InstallError(
            f"Failed to fetch wheel index {url} after {attempts} attempts: {detail}"
        ) from last_error

    parser = _LinkParser()
    parser.feed(body.decode("utf-8", errors="replace"))

    links: List[str] = []
    seen = set()
    for href in parser.hrefs:
        absolute = urljoin(url, href)
        parts = urlsplit(absolute)
        if parts.scheme != "https":
            continue
        if not unquote(parts.path).lower().endswith(".whl"):
            continue
        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return links


def parse_wheel(url: str) -> Optional[Wheel]:
    """Parse the wheel filename at the end of ``url``; None if malformed."""
    filename = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    try:
        distribution, version, _build, tags = parse_wheel_filename(filename)
    except (InvalidWheelFilename, InvalidVersion):
        return None
    return Wheel(url=url, distribution=distribution, version=version, tags=tags)


def is_compatible_with_current_python(tags: FrozenSet[Tag]) -> bool:
    """True if any wheel tag is supported by the running Python."""
    return not tags.isdisjoint(sys_tags())


# ---------------------------------------------------------------- torch build

_TORCH_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
_TORCH_MARKER_RE = re.compile(r"torch(\d+\.\d+\.\d+)")
_CUDA_MARKER_RE = re.compile(r"cuda(\d+\.\d+)")
_ROCM_MARKER_RE = re.compile(r"rocm(\d+\.\d+)")
_CPU_MARKER_RE = re.compile(r"(^|\.)cpu(\.|$)")


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
            "This is a ROCm build of PyTorch; k2utils install currently "
            "supports CPU and CUDA wheels only."
        )
    if cuda:
        return TorchInfo(version=version, backend="cuda", cuda_version=_major_minor(cuda))
    return TorchInfo(version=version, backend="cpu", cuda_version=None)


def index_url_for(package: WheelPackage, torch: TorchInfo, cn: bool = False) -> str:
    if torch.backend == "cuda":
        return package.cuda_cn_index if cn else package.cuda_index
    return package.cpu_cn_index if cn else package.cpu_index


# ---------------------------------------------------------------- wheel match


def parse_wheel_markers(local: Optional[str]) -> WheelMarkers:
    """Parse the wheel local version, e.g. "cuda13.2.torch2.13.0"."""
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
        return WheelMarkers(
            torch_version=torch_version, backend="cuda", backend_version=cuda_matches[0]
        )
    if rocm_matches:
        raise ValueError("ROCm wheels are not supported")
    if not is_cpu:
        raise ValueError(f"no cpu/cuda marker in {local!r}")
    return WheelMarkers(torch_version=torch_version, backend="cpu", backend_version=None)


def select_wheel(links: List[str], torch: TorchInfo, package: WheelPackage) -> Wheel:
    """Pick the newest wheel of ``package`` that matches the environment exactly."""
    compatible = []
    for url in links:
        wheel = parse_wheel(url)
        if wheel is None or wheel.distribution != package.name:
            continue
        try:
            markers = parse_wheel_markers(wheel.version.local)
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
            f"No pre-built {package.name} wheel matches PyTorch {torch.version}{cuda} "
            f"on {platform.system()} {platform.machine()} with "
            f"{platform.python_implementation()} {platform.python_version()}. "
            "k2utils does not fall back to a wheel built for another torch, "
            f"CUDA, Python or platform; see {package.source_url}"
        )
    return max(compatible, key=lambda w: (w.version, w.url))


# ---------------------------------------------------------------- install/run

_VERIFY_CODE = """\
import json
import torch
import __PACKAGE__

print("__K2UTILS_VERIFY__" + json.dumps({
    "package_version": str(getattr(__PACKAGE__, "__version__", "unknown")),
    "torch_version": torch.__version__,
    "torch_cuda": getattr(torch.version, "cuda", None),
}))
"""


def build_install_command(url: str) -> List[str]:
    """Install into the running Python's environment.

    Prefer pip. venvs created by ``uv venv`` ship without pip, so fall back to
    ``uv pip install --python <this interpreter>`` when pip is missing.
    """
    if find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", url]
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "pip", "install", "--python", sys.executable, "--upgrade", url]
    raise InstallError(
        "Neither pip nor uv is available in this environment. venvs created "
        "by uv do not include pip; install uv (https://docs.astral.sh/uv/) or "
        "run 'uv pip install pip' to add pip first."
    )


def run_install(url: str, dry_run: bool = False) -> None:
    command = build_install_command(url)
    print("Install command:")
    print(f"  {shlex.join(command)}")
    if dry_run:
        print("Dry run: the install command was not executed.")
        return
    result = subprocess.run(command)
    if result.returncode != 0:
        raise InstallError(f"installer failed with exit code {result.returncode}")


def verify_install(package: str, torch: TorchInfo) -> None:
    """Import ``package`` in a fresh interpreter and check the torch runtime matches."""
    print("Verification:")
    code = _VERIFY_CODE.replace("__PACKAGE__", package)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise InstallError(
            f"{package} was installed but importing it failed:\n"
            + (result.stderr.strip() or "unknown error")
        )

    payload = None
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("__K2UTILS_VERIFY__"):
            payload = json.loads(line[len("__K2UTILS_VERIFY__"):])
            break
    if payload is None:
        raise InstallError(f"{package} verification produced no result")

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

    print(f"  {package}: {payload['package_version']}")
    print(f"  torch: {payload['torch_version']}")


def install_wheels(
    package: WheelPackage,
    *,
    dry_run: bool = False,
    torch_version: Optional[str] = None,
    cn: bool = False,
) -> int:
    """Detect the environment, pick an exact wheel and install it."""
    torch = detect_torch()
    if torch_version is not None:
        requested = parse_torch_version(torch_version)
        if requested != torch.version:
            raise InstallError(
                f"--torch-version {torch_version} does not match the installed "
                f"PyTorch {torch.version}; k2utils only installs {package.name} "
                "for the torch already present in this environment."
            )

    cuda = f" CUDA {torch.cuda_version}" if torch.cuda_version else ""
    print("Environment:")
    print(f"  OS: {platform.system()} {platform.machine()}")
    print(f"  Python: {platform.python_implementation()} {platform.python_version()}")
    print(f"  PyTorch: {torch.version}")
    print(f"  Backend: {torch.backend}{cuda}")

    index_url = index_url_for(package, torch, cn=cn)
    print("Resolving:")
    print(f"  Index: {index_url}")
    wheel = select_wheel(fetch_wheel_links(index_url), torch, package)
    print(f"  Selected: {wheel.url}")

    run_install(wheel.url, dry_run=dry_run)
    if not dry_run:
        verify_install(package.name, torch)
    return 0

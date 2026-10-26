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

"""Helpers shared by the ``k2utils install`` commands.

Every install target in this project is a flat page of wheel links plus
wheel-filename metadata, so only four things are shared: fetch the page,
parse a wheel filename, check wheel tags against the running Python, and
run the installer (pip when available, otherwise uv).
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from importlib.util import find_spec
from typing import FrozenSet, List, Optional
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

from packaging.tags import Tag, sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from k2utils import __version__

DEFAULT_TIMEOUT = 30.0


class InstallError(Exception):
    """An error that can be shown to the user as a short message."""


@dataclass(frozen=True)
class Wheel:
    url: str
    distribution: str
    version: Version
    tags: FrozenSet[Tag]


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def fetch_wheel_links(url: str, timeout: float = DEFAULT_TIMEOUT) -> List[str]:
    """Download ``url`` and return its absolute HTTPS .whl links."""
    request = Request(url, headers={"User-Agent": f"k2utils/{__version__}"})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except Exception as e:  # HTTPError/URLError/timeout are all OSError subclasses
        reason = getattr(e, "reason", e)
        code = getattr(e, "code", None)
        detail = f"HTTP {code}: {reason}" if code is not None else str(reason)
        raise InstallError(f"Failed to fetch wheel index {url}: {detail}") from e

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

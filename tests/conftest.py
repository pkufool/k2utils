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

"""Shared test helpers: fake torch builds, wheel URLs and verify payloads."""

import json
from types import SimpleNamespace

from packaging.tags import sys_tags
from packaging.version import Version

from k2utils.commands.install.common import TorchInfo, WheelPackage

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


def torch_info(version: str = "2.8.0", cuda=None, hip=None) -> TorchInfo:
    return TorchInfo(
        version=Version(version),
        backend="cpu" if cuda is None else "cuda",
        cuda_version=cuda,
    )


def make_package(name: str = "k2") -> WheelPackage:
    return WheelPackage(
        name=name,
        cpu_index=f"{HOST}/cpu.html",
        cuda_index=f"{HOST}/cuda.html",
        cpu_cn_index=f"{HOST}/cpu-cn.html",
        cuda_cn_index=f"{HOST}/cuda-cn.html",
        source_url=f"{HOST}/from_source.html",
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


def fake_verify_run(stdout, returncode=0, stderr=""):
    def run(command, **kwargs):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return run


def verify_payload(
    package_version: str = "1.24.4",
    torch_version: str = "2.8.0",
    torch_cuda=None,
) -> str:
    body = json.dumps(
        {
            "package_version": package_version,
            "torch_version": torch_version,
            "torch_cuda": torch_cuda,
        }
    )
    return f"__K2UTILS_VERIFY__{body}\n"

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

from typing import Optional

from k2utils.commands.install.common import WheelPackage, install_wheels

K2 = WheelPackage(
    name="k2",
    cpu_index="https://k2-fsa.github.io/k2/cpu.html",
    cuda_index="https://k2-fsa.github.io/k2/cuda.html",
    cpu_cn_index="https://k2-fsa.github.io/k2/cpu-cn.html",
    cuda_cn_index="https://k2-fsa.github.io/k2/cuda-cn.html",
    source_url="https://k2-fsa.github.io/k2/installation/from_source.html",
)


def install_k2(
    *, dry_run: bool = False, torch_version: Optional[str] = None, cn: bool = False
) -> int:
    return install_wheels(K2, dry_run=dry_run, torch_version=torch_version, cn=cn)

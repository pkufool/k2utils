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

import argparse
import sys
from typing import Optional, Sequence

from k2utils.commands.install import k2 as k2_installer
from k2utils.commands.install import kaldifeat as kaldifeat_installer
from k2utils.commands.install.common import InstallError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="k2utils",
        description="Utilities for installing packages from the k2 ecosystem.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    install_parser = subparsers.add_parser("install", help="Install a package.")
    install_subparsers = install_parser.add_subparsers(dest="package", required=True)

    k2_parser = install_subparsers.add_parser(
        "k2",
        help="Install k2 from the official pre-built wheel index.",
        description=(
            "Detect the current Python/PyTorch environment, select an exactly "
            "compatible k2 wheel from the official CPU or CUDA index, install "
            "it with this Python's pip, and verify the result."
        ),
    )
    _add_install_options(k2_parser)
    k2_parser.set_defaults(handler=_run_install, installer=k2_installer.install_k2)

    kaldifeat_parser = install_subparsers.add_parser(
        "kaldifeat",
        help="Install kaldifeat from the official pre-built wheel index.",
        description=(
            "Detect the current Python/PyTorch environment, select an exactly "
            "compatible kaldifeat wheel from the official CPU or CUDA index, "
            "install it with this Python's pip, and verify the result."
        ),
    )
    _add_install_options(kaldifeat_parser)
    kaldifeat_parser.set_defaults(
        handler=_run_install, installer=kaldifeat_installer.install_kaldifeat
    )

    return parser


def _add_install_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the selected wheel and pip command without installing.",
    )
    parser.add_argument(
        "--torch-version",
        metavar="VERSION",
        help=(
            "Torch version the wheel must match, e.g. 2.8.0. "
            "Defaults to the installed torch version and must match it if given."
        ),
    )
    parser.add_argument(
        "--cn",
        action="store_true",
        help=(
            "Use the CN mirror wheel index (hf-mirror.com links), for users "
            "who cannot reach Hugging Face."
        ),
    )


def _run_install(args: argparse.Namespace) -> int:
    return args.installer(
        dry_run=args.dry_run,
        torch_version=args.torch_version,
        cn=args.cn,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except InstallError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

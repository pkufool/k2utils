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

import os
import subprocess
import sys
from pathlib import Path

import pytest

from k2utils import cli
from k2utils.commands.install.common import InstallError


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["install", "--help"],
        ["install", "k2", "--help"],
        ["install", "kaldifeat", "--help"],
    ],
)
def test_help_exits_zero(argv):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 0


@pytest.mark.parametrize("argv", [["bogus"], ["install"], ["install", "bogus"]])
def test_bad_usage_exits_two(argv):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "installer_module, command, entry_name",
    [
        (cli.k2_installer, "k2", "install_k2"),
        (cli.kaldifeat_installer, "kaldifeat", "install_kaldifeat"),
    ],
    ids=["k2", "kaldifeat"],
)
def test_install_routes_to_installer(monkeypatch, installer_module, command, entry_name):
    recorded = {}

    def fake_install(*, dry_run=False, torch_version=None, cn=False):
        recorded["dry_run"] = dry_run
        recorded["torch_version"] = torch_version
        recorded["cn"] = cn
        return 0

    monkeypatch.setattr(installer_module, entry_name, fake_install)
    assert cli.main(["install", command, "--dry-run", "--torch-version", "2.8.0"]) == 0
    assert recorded == {"dry_run": True, "torch_version": "2.8.0", "cn": False}
    assert cli.main(["install", command, "--cn"]) == 0
    assert recorded["cn"] is True


def test_install_error_returns_one(monkeypatch, capsys):
    def fake_install(*, dry_run=False, torch_version=None, cn=False):
        raise InstallError("something went wrong")

    monkeypatch.setattr(cli.k2_installer, "install_k2", fake_install)
    assert cli.main(["install", "k2"]) == 1
    assert "error: something went wrong" in capsys.readouterr().err


def test_keyboard_interrupt_returns_130(monkeypatch):
    def fake_install(*, dry_run=False, torch_version=None, cn=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.k2_installer, "install_k2", fake_install)
    assert cli.main(["install", "k2"]) == 130


def test_python_m_entry_point():
    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(src)}
    result = subprocess.run(
        [sys.executable, "-m", "k2utils", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "install" in result.stdout

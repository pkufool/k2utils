# k2utils

Utilities for installing packages from the [k2](https://github.com/k2-fsa/)
ecosystem from their official pre-built wheel pages.

## Install

Install from PyPI:

```bash
pip install k2utils
```

Or install from a local source checkout:

```bash
cd k2utils
pip install .
```

For development, use an editable install together with the test dependencies:

```bash
cd k2utils
pip install -e ".[dev]"
```

## Usage

```bash
k2utils install k2 [--dry-run] [--torch-version VERSION] [--cn]
```

`k2utils install k2` detects the running Python/PyTorch environment, selects an
exactly compatible k2 wheel from the official index, installs it with the same
Python's pip, and verifies that `import k2` works:

```bash
$ k2utils install k2
Environment:
  OS: Linux x86_64
  Python: CPython 3.11.9
  PyTorch: 2.8.0
  Backend: CUDA 12.8
Resolving:
  Index: https://k2-fsa.github.io/k2/cuda.html
  Selected: https://huggingface.co/.../k2-1.24.4.dev20260710+cuda12.8.torch2.8.0-...whl
Install command:
  /usr/bin/python3 -m pip install --upgrade --no-cache-dir https://huggingface.co/...whl
Verification:
  k2: 1.24.4.dev20260710
  torch: 2.8.0
```

Options:

- `--dry-run` — resolve and show the selected wheel and pip command without
  installing anything.
- `--torch-version VERSION` — assert the target torch version (e.g. `2.8.0`).
  It must match the installed torch; the option exists for scripts that want
  to pin the expectation.
- `--cn` — use the CN mirror index pages (`cpu-cn.html` / `cuda-cn.html`),
  whose wheel links go through the <https://hf-mirror.com> proxy, for users
  who cannot reach Hugging Face.

## Wheels

The index pages are flat lists of pre-built wheels:

- CPU (Linux, macOS, Windows): <https://k2-fsa.github.io/k2/cpu.html>
- CUDA (Linux): <https://k2-fsa.github.io/k2/cuda.html>
- CN mirrors: <https://k2-fsa.github.io/k2/cpu-cn.html> /
  <https://k2-fsa.github.io/k2/cuda-cn.html> (selected with `--cn`)

The CPU or CUDA page is chosen automatically from the installed PyTorch build
(`torch.version.cuda`). PyTorch must already be installed — k2 wheels are
built for an exact PyTorch version and backend, so k2utils never guesses one.

## Matching rules

A wheel is installed only when **all** of the following match exactly:

- k2 wheel's `torchX.Y.Z` marker equals the installed torch version;
- CPU/CUDA backend equals the torch build, and for CUDA the `cudaX.Y` marker
  equals `torch.version.cuda` (major.minor);
- wheel Python/ABI/platform tags are supported by the running interpreter
  (checked with `packaging.tags`).

There is **no fallback** to a wheel built for another torch version, CUDA
version, backend, Python or platform. If nothing matches, the command fails
and prints the environment it searched for. To build k2 from source, see the
[k2 installation docs](https://k2-fsa.github.io/k2/installation/from_source.html).

ROCm PyTorch builds are not supported yet and are rejected explicitly.

## Network

Index pages are fetched with the Python standard library, so `HTTP_PROXY`,
`HTTPS_PROXY` and `NO_PROXY` are honored. pip inherits the same proxy
settings and its own configuration.

## uv-managed environments

venvs created by `uv venv` do not include pip. k2utils detects this and falls
back to `uv pip install --python <current interpreter>` automatically.

Install k2utils into the environment where you want k2 — e.g.
`uv pip install k2utils`, or `uv add k2utils` followed by
`uv run k2utils install k2`. Do not run it via `uvx` or `uv tool install`:
those put k2utils in its own isolated environment, and k2 would end up there
instead of your project.

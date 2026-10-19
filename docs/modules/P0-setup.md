# P0 · Setup: repo and environment (~45 min)

> **Project type: 1 · Real-world application of GNNs** (Medium blog post + Google Colab), not type 2 (PyG tutorial) or type 3 (paper re-implementation as a PyG PR).
>
> **Proposal due Tue Oct 20, 2026, 11:59pm PT** · milestone Tue Nov 10 · final Thu Dec 10.

Modules P1–P8 each cover one bullet of the handout's proposal instructions (§1.3), and this module sets up what they all need. Everything below was tested on this laptop on Oct 16. `PowerShell` blocks run from the repo root. Each step ends with **PASS** or **STOP** (paste the output to me).

## 0.1 Commit the plan

```powershell
git status                 # README.md, .gitignore, docs/ — no .pem or *.local.json files
git add README.md .gitignore docs
git commit -m "Pivot to Worth the Lookup: plan, related work, proposal modules; archive AML plans"
git push
```

## 0.2 Skeleton

```powershell
# Code, scripts, tests, proposal source, AWS files, and results (outputs/ is Git-ignored).
New-Item -ItemType Directory -Force src\wtl, scripts, tests, proposal\figures, infra\aws, outputs | Out-Null
# Git doesn't track empty folders; a .gitkeep keeps them until they have content.
foreach ($d in "proposal\figures", "infra\aws") { New-Item -ItemType File -Force "$d\.gitkeep" | Out-Null }
```

`pyproject.toml` (repo root). It makes `src/wtl` an installable package. Libraries are installed in 0.3, because PyTorch and `pyg-lib` come from special indexes a pyproject can't pin cleanly.

```toml
[project]
name = "wtl"
version = "0.1.0"
description = "Worth the Lookup: budgeted relational retrieval for call-list prioritization"
requires-python = ">=3.10"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

`src/wtl/__init__.py`:

```python
"""Worth the Lookup: budgeted relational retrieval for call-list prioritization."""
```

## 0.3 Environment: x64 Python on your ARM laptop

`pyg-lib` (PyG's compiled sampler, required for time-aware sampling) has no Windows-ARM build. It does have an x64 Windows build, and Windows 11 runs x64 Python through emulation. Imports are slower (~20 s the first time), but the whole stack works on CPU. The pinned versions match RelBench's own tested environment.

```powershell
# 1) Virtual env with x64 Python 3.12 (uv downloads it once).
uv venv --python cpython-3.12-windows-x86_64-none .venv
# 2) Activate (your RemoteSigned policy allows it).
.\.venv\Scripts\Activate.ps1
# 3) PyTorch 2.9.1, CPU build, first, so nothing later pulls a different build.
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
# 4) RelBench 3.0.2 + modeling extras (PyG, pytorch-frame, sentence-transformers, LightGBM).
#    Never install "torch-frame" from PyPI (unrelated project); "pytorch-frame" comes in here.
uv pip install "relbench[example]==3.0.2"
# 5) pyg-lib built for torch 2.9 on CPU, from PyG's wheel index (not on PyPI).
uv pip install pyg-lib --find-links https://data.pyg.org/whl/torch-2.9.0+cpu.html
# 6) Plotting, notebooks, tests.
uv pip install matplotlib ipykernel pytest
# 7) Your package, editable: edits in src/wtl take effect without reinstalling.
uv pip install -e .
# Optional: silence a harmless Hugging Face symlink warning in new windows.
[Environment]::SetEnvironmentVariable("HF_HUB_DISABLE_SYMLINKS_WARNING", "1", "User")
```

In VS Code: **Ctrl+Shift+P → Python: Select Interpreter → `.venv\Scripts\python.exe`**.

> Keep this repo where it is. Some installed files have long names, and this path keeps all of them under Windows' 260-character limit.

## 0.4 `scripts/check_env.py`

```python
"""Environment check: confirm the Python build, the CPU/GPU stack, and every library the project needs.

Run it on the laptop now, and again on the GPU instance later (build track B1).
"""
import sysconfig
from importlib.metadata import version  # reads installed versions without importing the packages

import torch
import torch_geometric.typing as pyg_typing  # flags for which compiled PyG extensions loaded

# Platform of the Python build: "win-amd64" on the laptop (x64 Python on an ARM chip), "linux-x86_64" on AWS.
print("python build:", sysconfig.get_platform())

# Torch build, and whether it sees a GPU (False on the laptop is expected).
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# The critical flag: pyg-lib loaded => time-aware neighbor sampling works.
print("torch_geometric", version("torch_geometric"), "| pyg-lib loaded:", pyg_typing.WITH_PYG_LIB)

# The relational stack: pytorch-frame encodes table columns; relbench supplies data and tasks.
print("pytorch-frame", version("pytorch-frame"), "| relbench", version("relbench"))
```

```powershell
python scripts\check_env.py
```

**PASS:**

```text
python build: win-amd64
torch 2.9.1+cpu | CUDA available: False
torch_geometric 2.8.1 | pyg-lib loaded: True
pytorch-frame 0.3.0 | relbench 3.0.2
```

**STOP:**

- `win-arm64`: delete `.venv` and redo step 1.
- `pyg-lib loaded: False`: paste the step-5 output.

**Note:** keep printed text ASCII. Windows' console encoding crashes on characters like `→` when output is redirected.

## 0.5 Commit

```powershell
git add pyproject.toml src scripts proposal infra   # tests/ is still empty; Git can't add an empty folder
git commit -m "P0: package skeleton and environment check"
git push
```

**Next: [P1 · Application domain](P1-application-domain.md)**

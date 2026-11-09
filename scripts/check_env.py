"""Environment check: confirm the Python build, the CPU/GPU stack, and every library the project needs.

Question it answers: is this machine set up so the rest of the project can run?
It only prints version numbers and flags. It does not download data or train anything.

How to run (from the repo root, with the project's virtual environment active):
    python scripts/check_env.py

A healthy output, in words:
- "python build" is a 64-bit x86 build: win-amd64 on the Windows laptop, linux-x86_64 on a Linux GPU
  machine. win-arm64 means the wrong Python was installed (see setup step P0).
- "CUDA available" is False on a laptop without an NVIDIA GPU (expected), and True on the GPU machine,
  where an extra line names the GPU.
- "pyg-lib loaded" is True. This is the line that matters most: pyg-lib is what lets the neighbor sampler
  respect time, and the whole project depends on that.
- pytorch-frame and relbench print their version numbers.

Where it fits: setup step P0. Run it first on every new machine (laptop, cloud GPU instance, Colab).
"""

# sysconfig: facts about how this Python itself was built (platform, install paths)
import sysconfig 
# reads installed versions without importing the packages
from importlib.metadata import version

import torch
import torch_geometric.typing as pyg_typing # flags for which compiled PyG extensions loaded

# Step 1: platform of the Python build.
# sysconfig.get_platform() returns a short tag: the operating system plus the chip type Python was compiled for.
print("python build:", sysconfig.get_platform())

# Step 2: the torch build, and whether it sees a GPU (False is expected on this laptop).
# torch.cuda.is_available() is True only if torch was built with CUDA support AND a working NVIDIA GPU is present.
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    # GPUs are numbered from 0; this prints the first one's name.
    print("GPU", torch.cuda.get_device_name(0))

# Step 3: PyTorch Geometric (PyG), the graph library.
# WITH_PYG_LIB is True when pyg-lib, PyG's compiled helper library, loaded. pyg-lib provides time-aware
# neighbor sampling: each target only sees rows dated on or before its prediction time. False here is a blocker.
print("torch_geometric", version("torch_geometric"), "| pyg-lib loaded:", pyg_typing.WITH_PYG_LIB)

# Step 4: the relational stack: pytorch-frame encodes table columns; relbench supplies data and tasks.
print("pytorch-frame", version("pytorch-frame"), "| relbench", version("relbench"))
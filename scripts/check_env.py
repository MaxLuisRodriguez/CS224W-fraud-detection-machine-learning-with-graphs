"""Environment check: confirm the Python build, the CPU/GPU stack, and every library the project needs.
"""

import sysconfig 
# reads installed versions without importing the packages
from importlib.metadata import version

import torch
import torch_geometric.typing as pyg_typing # flags for which compiled PyG extensions loaded

# Platform of python build
print("python build:", sysconfig.get_platform())

# Torch build, and whether it sees a GPU (should be false on this laptop
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0))

print("torch_geometric", version("torch_geometric"), "| pyg-lib loaded:", pyg_typing.WITH_PYG_LIB)

# The relational stack: pytorch-frame encodes table columns; relbench supplies data and tasks.
print("pytorch-frame", version("pytorch-frame"), "| relbench", version("relbench"))
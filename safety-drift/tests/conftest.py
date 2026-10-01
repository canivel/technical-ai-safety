# CPU tests: hide the CUDA-only linear-attention kernels so transformers resolves to its pure-torch
# GatedDeltaNet path (hub_kernels.use_kernel_func_from_hub_with_fallback imports these at import time).
import sys

sys.modules["causal_conv1d"] = None
sys.modules["fla"] = None

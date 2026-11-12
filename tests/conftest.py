import os

# Mol-1 and Protenix try to build a fused LayerNorm CUDA extension when CUDA is visible.
os.environ.setdefault("LAYERNORM_TYPE", "torch")

import jax
import pytest
import torch

# Use full float32 on both sides so parity tolerances measure implementation differences.
jax.config.update("jax_default_matmul_precision", "highest")


@pytest.fixture(autouse=True)
def full_float32_torch():
    """Disable TF32, which some upstream code enables mid-run, making parity order-dependent."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

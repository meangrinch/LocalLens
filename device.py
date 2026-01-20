from typing import Optional

import torch


def get_best_device() -> torch.device:
    """Return the optimal device based on platform and available hardware.

    Priority order: CUDA > XPU > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_best_dtype(device: Optional[torch.device] = None) -> torch.dtype:
    """Return the optimal dtype for the given device.

    Args:
        device: The target device. If None, uses get_best_device().

    Returns:
        torch.dtype: bfloat16/float16 for GPU, float32 for CPU.
    """
    if device is None:
        device = get_best_device()

    device_type = device.type

    if device_type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    if device_type == "xpu":
        try:
            if (
                hasattr(torch.xpu, "is_bf16_supported")
                and torch.xpu.is_bf16_supported()
            ):
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    if device_type == "mps":
        return torch.float16

    return torch.float32

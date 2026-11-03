"""Compatibility shim: implementation lives in Megatron-Bridge."""
from megatron.bridge.training.causal_conv1d_backend import install_causal_conv1d_backend

__all__ = ["install_causal_conv1d_backend"]

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Write the built model's structure to a JSON artifact.

``tune_common.sh`` points ``MODEL_STRUCTURE_JSON`` at a file inside the run
directory. The dump replaces the Megatron-FSDP parameter-group block, which
otherwise prints one line per parameter into the training log: the same
inventory (module tree, parameter names, shapes, dtypes) lands in one file that
the run archives. Rank 0 writes it right after the model is built, for both
entrypoints (``pretrain`` and ``pretrain_megatron_mimo``).

Without ``MODEL_STRUCTURE_JSON`` nothing is written and the log block stays.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import torch.distributed as dist

FSDP_LOGGER = "megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer"
FSDP_GROUP_DUMP = "Number of FSDP Parameter Groups:"
SCHEMA = "model-structure/1"

logger = logging.getLogger(__name__)
_installed = False


class _DropFsdpGroupDump(logging.Filter):
    """Drop the FSDP parameter-group block that the JSON now carries."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(FSDP_GROUP_DUMP)


def _chunks(model):
    return list(model) if isinstance(model, (list, tuple)) else [model]


def _size(shape) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _parallel(model) -> dict:
    config = getattr(model, "config", None) or getattr(getattr(model, "module", None), "config", None)
    return {
        "tensor_parallel": getattr(config, "tensor_model_parallel_size", None),
        "pipeline_parallel": getattr(config, "pipeline_model_parallel_size", None),
    }


def describe(model, source: str) -> dict:
    """Return the module tree and the parameter inventory of ``model``."""
    chunks = _chunks(model)
    modules, parameters = [], []
    dtypes: dict[str, int] = {}
    for chunk in chunks:
        for name, module in chunk.named_modules():
            modules.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "numel": sum(_size(p.shape) for p in module.parameters(recurse=False)),
                }
            )
        for name, parameter in chunk.named_parameters():
            shape = [int(dim) for dim in parameter.shape]
            dtype = str(parameter.dtype).replace("torch.", "")
            numel = _size(shape)
            dtypes[dtype] = dtypes.get(dtype, 0) + numel
            parameters.append(
                {
                    "name": name,
                    "shape": shape,
                    "dtype": dtype,
                    "numel": numel,
                    "requires_grad": bool(parameter.requires_grad),
                }
            )
    return {
        "schema": SCHEMA,
        "source": source,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        # Shapes are the ones this rank holds: with TP they are per-rank.
        "parallel": _parallel(chunks[0]) if chunks else {},
        "totals": {
            "parameters": sum(entry["numel"] for entry in parameters),
            "parameter_tensors": len(parameters),
            "dtypes": dtypes,
        },
        "modules": modules,
        "parameters": parameters,
    }


def write(path: str, model, source: str = "") -> None:
    payload = describe(model, source)
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=1)
        stream.write("\n")
    os.replace(temporary, path)


def _wrap(module, name: str, path: str, source: str) -> None:
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        output = original(*args, **kwargs)
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                write(path, output.model, source)
            except Exception as error:  # never fail a training run over the artifact
                logger.warning("model structure dump failed: %s", error)
        return output

    setattr(module, name, wrapper)


def install_model_structure(path: str | None = None, source: str = "") -> None:
    """Dump the model structure to ``path`` once the model is built.

    Must run before pretraining starts: both the log filter and the setup
    wrappers have to be in place before the model is built. Both pretrain
    entrypoints are wrapped, so the entrypoint does not need to know which one
    runs.
    """
    global _installed
    path = path or os.environ.get("MODEL_STRUCTURE_JSON", "")
    if _installed or not path:
        return
    _installed = True
    logging.getLogger(FSDP_LOGGER).addFilter(_DropFsdpGroupDump())

    from megatron.bridge.training import pretrain, pretrain_megatron_mimo

    _wrap(pretrain, "setup", path, source)
    _wrap(pretrain_megatron_mimo, "setup_megatron_mimo", path, source)

#!/usr/bin/env bash
# ============================================================================
# Qwen3.5-35B-A3B weight preparation for the verl GoLongRL recipe.
#
# Unlike the slime/miles lines, verl + Megatron-Bridge needs NO offline
# weight conversion: actor/ref modeling goes through use_mbridge=True +
# vanilla_mbridge=False (the qwen35_vl_provider builds TransformerConfig
# from the HF checkpoint at runtime), and the sglang rollout also loads HF
# safetensors directly. This step only: presence check -> hf download
# fallback -> integrity validation. HF_MODEL_PATH is independent of the
# effective train TP (no per-TP torch_dist directories).
#
# Usage (inside container):
#   bash golongrl_prepare_weights_verl.sh
#   HF_MODEL_PATH=/path/to/models/Qwen3.5-35B-A3B bash golongrl_prepare_weights_verl.sh
# ============================================================================
set -euo pipefail

NAS_ROOT=${NAS_ROOT:-/workspace/golongrl}
HF_MODEL_PATH=${HF_MODEL_PATH:-${NAS_ROOT}/models/Qwen3.5-35B-A3B}
MODEL_REPO=${MODEL_REPO:-Qwen/Qwen3.5-35B-A3B}

mkdir -p "$(dirname "${HF_MODEL_PATH}")"

# ---------- presence check ----------
if [[ -f "${HF_MODEL_PATH}/config.json" && -n "$(ls ${HF_MODEL_PATH}/*.safetensors 2>/dev/null)" ]]; then
    echo "== [W0] weights present at ${HF_MODEL_PATH} — SKIP download =="
else
    echo "== [W0] weights missing — hf download ${MODEL_REPO} =="
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    hf download "${MODEL_REPO}" --local-dir "${HF_MODEL_PATH}"
fi

# ---------- integrity validation ----------
python3 - "${HF_MODEL_PATH}" <<'PY'
import json, os, sys
mp = sys.argv[1]
def need(f):
    p = os.path.join(mp, f)
    assert os.path.isfile(p), f"missing file: {p}"
    return p
cfg = json.load(open(need("config.json")))
# Qwen3.5-35B-A3B architecture fingerprint (config.json: top-level
# qwen3_5_moe, text_config with 40 layers x 256 experts, top-8 routing).
t = cfg.get("text_config", cfg)
assert cfg.get("model_type") == "qwen3_5_moe", cfg.get("model_type")
assert t.get("num_hidden_layers") == 40, t.get("num_hidden_layers")
assert t.get("num_experts", 0) == 256, t.get("num_experts")
assert t.get("num_experts_per_tok", 0) == 8, t.get("num_experts_per_tok")
n_shard = len([f for f in os.listdir(mp) if f.endswith(".safetensors")])
assert n_shard >= 1, "no safetensors shards"
need("tokenizer.json"); need("tokenizer_config.json")
if n_shard > 1:
    need("model.safetensors.index.json")
print(f"[weights OK] {mp}: {t['num_hidden_layers']}L x MoE{t['num_experts']}E(8/tok), {n_shard} safetensors shards, tokenizer OK")
PY

echo "== weight preparation done (${HF_MODEL_PATH}; point HF_MODEL_PATH here when training) =="
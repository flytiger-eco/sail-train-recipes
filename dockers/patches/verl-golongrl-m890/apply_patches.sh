#!/usr/bin/env bash
# ============================================================================
# verl 0.9.1-m890 GoLongRL patch chain (patches_release, idempotent)
#
# Scope: GoLongRL training on 8x M890 — rollout stability, dataset reward,
# memory optimization, PPU platform fixes, plus chunkwise GDN CP + contiguous
# data plane (m890 additions 07/08/09).
#
# Patch list and order (apply order is fixed):
#   verl source tree (/opt/verl, pristine v0.9.1 editable in the base image):
#     01_rollout_stability.patch   # DEFAULT_TIMEOUT 60->3600 + ipc_collect
#     02_golongrl_dataset.patch    # GoLongRL dataset (reward_score routing, 7 files)
#     03_memory_opt.patch          # TE parallel_cross_entropy CE + watermark gating
#     09_verl_chunkwise_wiring.patch  # util.py allgather_cp injection (2 sites);
#                                   # outer data plane stays zigzag (GDN converts internally)
#   site-packages (a/ header patches, patch -d <site-packages> -p1):
#     04_megatron_tms_hookmode.patch       # megatron TMS hook_mode conditional
#     05_sglang_ipc_barrier.patch          # sglang weight_sync bucket barrier
#                                          # (SKIP-baked: official 20261113 base
#                                          #  already carries VERL_PATCH_IPC_BARRIER)
#     06_sglang_kv_resume_serialize.patch  # sglang KV resume per-rank serialize (256k)
#     07_megatron_gdn_chunkwise_cp.patch   # GDN chunkwise CP + context_parallel_layout
#                                          # module + TransformerConfig/PackedSeqParams
#                                          # fields (megatron-core 0.18.0 single file)
#     08_megatron_a2a_contiguous.patch     # full-attn entry/exit single a2a + TE ext
#                                          # kept_packed_seq_params discard
#
# Idempotency: each patch is reverse-dry-run first (already applied -> SKIP);
# 05 additionally accepts the baked-in anchor. For a different patch set use a
# clean tree / fresh container; this script does not revert foreign changes.
#
# Hard prerequisite: transformer_engine==2.16+v0.1.0.ppu2.2.0 (patch 03 CE path
# and CP>1 fused attention backend). The official 20261113 base ships it.
#
# Usage (inside container):
#   bash <release>/patches_release/apply_patches.sh                # /opt/verl
#   VERL_ROOT=/root/verl-clean bash .../apply_patches.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${SCRIPT_DIR}"
VERL_ROOT=${VERL_ROOT:-/opt/verl}
SP="$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null | grep -m1 '^/')"

# ---------- [0] pre-flight ----------
[[ -d "${VERL_ROOT}/verl" ]] || { echo "ERROR: verl source tree missing: ${VERL_ROOT}" >&2; exit 1; }
for p in 01_rollout_stability 02_golongrl_dataset 03_memory_opt \
         04_megatron_tms_hookmode 05_sglang_ipc_barrier \
         06_sglang_kv_resume_serialize 07_megatron_gdn_chunkwise_cp \
         08_megatron_a2a_contiguous 09_verl_chunkwise_wiring; do
    [[ -f "${PATCH_DIR}/${p}.patch" ]] || { echo "ERROR: missing ${PATCH_DIR}/${p}.patch" >&2; exit 1; }
done
TE_VER=$(python3 -c "import importlib.metadata as m; print(m.version('transformer_engine'))" 2>/dev/null | grep -m1 '^[0-9]')
if [[ "${TE_VER}" != "2.16+v0.1.0.ppu2.2.0" ]]; then
    echo "ERROR: transformer_engine=${TE_VER}, need 2.16+v0.1.0.ppu2.2.0 (patch 03 hard dep; CP>1 fused backend)." >&2
    echo "       Use the m890 docker/Dockerfile (TE-2.16-ready base) and rerun." >&2
    exit 1
fi
echo "== [0] pre-flight OK: VERL_ROOT=${VERL_ROOT} SP=${SP} TE=${TE_VER} =="

# Idempotent apply: reverse dry-run (already applied -> SKIP), then forward
# dry-run (clean -> apply), otherwise fail loudly.
apply() {  # apply <patch file> <target dir> <note>
    local pf="$1" tgt="$2" note="$3"
    if patch -d "${tgt}" -p1 -R --dry-run -s < "${pf}" >/dev/null 2>&1; then
        echo "  [SKIP] ${note} (already applied)"
    elif patch -d "${tgt}" -p1 --dry-run -s < "${pf}" >/dev/null 2>&1; then
        patch -d "${tgt}" -p1 -s < "${pf}"
        echo "  [APPLIED] ${note}"
    else
        echo "ERROR: ${note} neither applied nor cleanly applicable; inspect target tree" >&2; exit 1
    fi
}

# ---------- [1-3] verl tree: 01 -> 02 -> 03 ----------
echo "== [1-3] verl tree patches (order 01 -> 02 -> 03) =="
apply "${PATCH_DIR}/01_rollout_stability.patch"  "${VERL_ROOT}" "01 rollout stability (timeout 3600 + ipc_collect)"
apply "${PATCH_DIR}/02_golongrl_dataset.patch"   "${VERL_ROOT}" "02 GoLongRL dataset (reward_score routing)"
apply "${PATCH_DIR}/03_memory_opt.patch"         "${VERL_ROOT}" "03 memory opt (TE CE + watermark gating)"

# ---------- [4-6] site-packages: 04 -> 05 -> 06 ----------
echo "== [4-6] site-packages patches (04 megatron -> 05/06 sglang) =="
apply "${PATCH_DIR}/04_megatron_tms_hookmode.patch"  "${SP}" "04 megatron TMS hookmode conditional"
# 05: the official 20261113 base bakes an equivalent per-bucket barrier
# (VERL_PATCH_IPC_BARRIER); accept the anchor as already-applied.
if grep -q 'dist.barrier(group=device_mesh' "${SP}/sglang/srt/weight_sync/utils.py"; then
    echo "  [SKIP-baked] 05 sglang IPC barrier (anchor present in base or prior apply)"
else
    apply "${PATCH_DIR}/05_sglang_ipc_barrier.patch" "${SP}" "05 sglang IPC barrier"
fi
apply "${PATCH_DIR}/06_sglang_kv_resume_serialize.patch" "${SP}" "06 sglang KV resume per-rank serialize (Xid 16258)"

# ---------- [7-8] site-packages: chunkwise CP + contiguous data plane ----------
echo "== [7-8] megatron chunkwise CP (07) + contiguous a2a (08) =="
apply "${PATCH_DIR}/07_megatron_gdn_chunkwise_cp.patch" "${SP}" "07 GDN chunkwise CP + CP-layout module + config fields"
apply "${PATCH_DIR}/08_megatron_a2a_contiguous.patch"   "${SP}" "08 full-attn contiguous a2a + TE ext discard"

# ---------- [9] verl tree: chunkwise wiring (after 03; disjoint hunks) ----------
echo "== [9] verl chunkwise wiring =="
apply "${PATCH_DIR}/09_verl_chunkwise_wiring.patch" "${VERL_ROOT}" "09 allgather_cp injection (outer zigzag)"

# ---------- anchors + compile probes ----------
echo "== [probe] functional anchors =="
grep -q 'torch.cuda.ipc_collect()' "${VERL_ROOT}/verl/workers/rollout/sglang_rollout/sglang_rollout.py" || { echo "ERROR: ipc_collect anchor missing" >&2; exit 1; }
grep -q 'parallel_cross_entropy(logits' "${VERL_ROOT}/verl/utils/megatron/tensor_parallel.py" || { echo "ERROR: TE CE anchor missing" >&2; exit 1; }
grep -q 'Release fragmented cache blocks' "${VERL_ROOT}/verl/workers/engine/megatron/transformer_impl.py" || { echo "ERROR: watermark anchor missing" >&2; exit 1; }
grep -q 'math_longcot' "${VERL_ROOT}/verl/utils/reward_score/__init__.py" || { echo "ERROR: dataset routing anchor missing" >&2; exit 1; }
grep -q 'incompatible with expandable_segments' "${SP}/megatron/core/inference/contexts/dynamic_context.py" || { echo "ERROR: patch 04 anchor missing" >&2; exit 1; }
grep -q 'dist.barrier(group=device_mesh' "${SP}/sglang/srt/weight_sync/utils.py" || { echo "ERROR: patch 05 barrier anchor missing" >&2; exit 1; }
grep -q 'for _r in range(self.tp_size)' "${SP}/sglang/srt/managers/scheduler_update_weights_mixin.py" || { echo "ERROR: patch 06 anchor missing" >&2; exit 1; }
grep -q '_forward_chunkwise' "${SP}/megatron/core/ssm/gated_delta_net.py" || { echo "ERROR: patch 07 chunkwise anchor missing" >&2; exit 1; }
grep -q 'a2a_cp_back_converter' "${SP}/megatron/core/transformer/attention.py" || { echo "ERROR: patch 08 a2a anchor missing" >&2; exit 1; }
[[ "$(grep -c 'extra_packed_args\["allgather_cp"\] = True' "${VERL_ROOT}/verl/models/mcore/util.py")" == "2" ]] || { echo "ERROR: patch 09 allgather_cp anchor missing" >&2; exit 1; }
python3 -m py_compile \
    "${VERL_ROOT}/verl/workers/rollout/sglang_rollout/sglang_rollout.py" \
    "${VERL_ROOT}/verl/utils/megatron/tensor_parallel.py" \
    "${VERL_ROOT}/verl/workers/engine/megatron/transformer_impl.py" \
    "${VERL_ROOT}/verl/models/mcore/util.py" \
    "${VERL_ROOT}/verl/utils/reward_score/__init__.py" \
    "${VERL_ROOT}/verl/utils/reward_score/golongrl/"*.py \
    "${SP}/megatron/core/inference/contexts/dynamic_context.py" \
    "${SP}/megatron/core/packed_seq_params.py" \
    "${SP}/megatron/core/ssm/gated_delta_net.py" \
    "${SP}/megatron/core/transformer/transformer_config.py" \
    "${SP}/megatron/core/transformer/attention.py" \
    "${SP}/megatron/core/extensions/transformer_engine.py" \
    "${SP}/megatron/core/context_parallel_layout/"*.py \
    "${SP}/sglang/srt/weight_sync/utils.py" \
    "${SP}/sglang/srt/managers/scheduler_update_weights_mixin.py"
echo "  all functional anchors present, py_compile OK"

# ---------- import probes (cross-package changes need a full import smoke) ----------
echo "== [probe] import smoke =="
python3 - <<'PY'
import dataclasses
from megatron.core.packed_seq_params import PackedSeqParams, resolve_cp_group
f = {x.name for x in dataclasses.fields(PackedSeqParams)}
assert {"cp_partition_mode", "cp_partition_route", "allgather_cp"} <= f, f
from megatron.core.context_parallel_layout import (
    CpPartitionModeConverter, convert_module_input_tensors_cp_partition_mode)
import megatron.core.ssm.gated_delta_net as gdn
assert hasattr(gdn.GatedDeltaNet, "_forward_chunkwise")
assert gdn.build_cp_context is not None, "fla build_cp_context unavailable"
from megatron.core.transformer.transformer_config import TransformerConfig
tf = {x.name for x in dataclasses.fields(TransformerConfig)}
assert {"linear_cp_mode", "cp_partition_mode"} <= tf
import megatron.core.transformer.attention  # noqa: F401
import types
from verl.workers.engine.megatron.transformer_impl import MegatronEngine
sel = MegatronEngine._get_context_parallel_layout
mk = lambda **kw: types.SimpleNamespace(config=types.SimpleNamespace(**kw))
assert sel(mk()) == "zigzag"
# outer data plane stays zigzag for chunkwise (GDN converts zigzag->contiguous internally);
# only the dsv4_hybrid attention variant requires a contiguous outer split.
assert sel(mk(linear_cp_mode="chunkwise", context_parallel_size=4)) == "zigzag"
assert sel(mk(linear_cp_mode="headwise", context_parallel_size=4)) == "zigzag"
assert sel(mk(experimental_attention_variant="dsv4_hybrid")) == "contiguous"
from verl.models.mcore.util import preprocess_thd_engine  # noqa: F401
print("  import smoke OK (PackedSeqParams/CP-layout/GDN/config/attention/verl wiring)")
PY

echo "== patches_release applied (01-09 full chain); rerun to verify idempotency =="
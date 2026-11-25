#!/bin/bash
# ============================================================================
# GoLongRL data preparation: HF download -> slime jsonl, four bands, idempotent.
#
# Five-step chain (skips steps whose outputs already exist; resumable):
#   B0 dataset download fallback: hf download Kwai-Klear/GoLongRL (HF_ENDPOINT
#      defaults to hf-mirror.com)
#   B1 full prompt token counting (Qwen3.5 tokenizer, threshold 24k) ->
#      prompts_over_24k.jsonl
#   B2 category filter (exclude multitableqa_pretraining) + per-band window split
#   B3 slime jsonl conversion (prompt/label/metadata keys, 95:5 train/val split)
#   B4 validation (row count / three-key structure / window bounds)
#
# Band windows (prompt token length):
#   32k  [24576, 28600]  resp 4096  context 32768
#   64k  [57344, 61380]  resp 4096  context 65536
#   128k [102400, 122800] resp 8192 context 131072
#   256k [204800, 253900] resp 8192 context 262144
#
# Output: ${SLIME_DATA_ROOT}/band${BAND}${DATA_SUFFIX}/{train,val}.jsonl + summary.json
#   each line: {"prompt":[{role,content}...], "label":<golden>, "metadata":{...}}
#
# Prereq: tokenizer available (TOKENIZER overridable, defaults to HF_CKPT);
#   transformers / pyarrow installed.
# Usage (inside container):
#   bash prepare_data.sh                       # all four bands
#   BAND=32k bash prepare_data.sh              # only 32k
#   DATA_SUFFIX=_user1 bash prepare_data.sh    # per-user isolation suffix
# Note: B1 full tokenization takes ~1-2h (22965 rows); delete prompts_over_24k.jsonl
#   to force a rerun.
# ============================================================================
set -euo pipefail
# Shared config — env-overridable; defaults are neutral placeholders (set to your
# storage layout). GCS_PORT/DASH_PORT avoid the ray worker range 10002-19999;
# RAY_TMP_DIR kept short (AF_UNIX 107-byte limit). Conversion TP must match training TP.
NAS_ROOT=${NAS_ROOT:-/workspace/golongrl}
MODEL_ROOT=${MODEL_ROOT:-/workspace/models}
HF_CKPT=${HF_CKPT:-${MODEL_ROOT}/Qwen3.5-35B-A3B}
GDN_TORCH_DIST=${GDN_TORCH_DIST:-${MODEL_ROOT}/Qwen3.5-35B-A3B_slime_torch_dist_gdn_tp2}
SLIME_DATA_ROOT=${SLIME_DATA_ROOT:-${NAS_ROOT}/data}
DATA_SUFFIX=${DATA_SUFFIX:-}
CKPT_ROOT=${CKPT_ROOT:-${NAS_ROOT}/checkpoints}
NUM_GPUS=${NUM_GPUS:-8}
GCS_PORT=${GCS_PORT:-26580}
DASH_PORT=${DASH_PORT:-28366}
RAY_TMP_DIR=${RAY_TMP_DIR:-/tmp/ray_golongrl}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
SLIME_ROOT=${SLIME_ROOT:-$(pip show slime 2>/dev/null | awk '/Editable project location:/{print $NF}')}
MEGATRON_ROOT=${MEGATRON_ROOT:-$(pip show megatron-core 2>/dev/null | awk '/Editable project location:/{print $NF}')}
export NAS_ROOT MODEL_ROOT HF_CKPT GDN_TORCH_DIST SLIME_DATA_ROOT DATA_SUFFIX \
       CKPT_ROOT NUM_GPUS GCS_PORT DASH_PORT RAY_TMP_DIR MASTER_ADDR SLIME_ROOT MEGATRON_ROOT

BANDS=${BAND:-"32k 64k 128k 256k"}
TOKENIZER=${TOKENIZER:-${HF_CKPT}}
DATA_DIR=${SLIME_DATA_ROOT}
PARQUET_GLOB="${DATA_DIR}/GoLongRL/data/*.parquet"
RAW_OUT="${DATA_DIR}/prompts_over_24k.jsonl"
EXCLUDE_DS="multitableqa_pretraining"

mkdir -p "${DATA_DIR}"

# ---------- B0 dataset download (fallback when missing) ----------
N_PARQUET=$(ls ${PARQUET_GLOB} 2>/dev/null | wc -l)
if [[ "${N_PARQUET}" -eq 0 ]]; then
    echo "== [B0] GoLongRL parquet missing — hf download Kwai-Klear/GoLongRL =="
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    hf download Kwai-Klear/GoLongRL --repo-type dataset --local-dir "${DATA_DIR}/GoLongRL"
    N_PARQUET=$(ls ${PARQUET_GLOB} 2>/dev/null | wc -l)
    [[ "${N_PARQUET}" -gt 0 ]] || { echo "ERROR: still no parquet after download; check network/HF_ENDPOINT"; exit 1; }
else
    echo "== [B0] GoLongRL parquet present (${N_PARQUET} shards) — SKIP =="
fi

# ---------- B1 full token counting (threshold 24k) ----------
echo "== [B1] full prompt token counting (threshold 24k) =="
if [[ -s "${RAW_OUT}" ]]; then
    echo "  ${RAW_OUT} present ($(wc -l < "${RAW_OUT}") rows) — SKIP"
else
    python3 - "${PARQUET_GLOB}" "${RAW_OUT}" "${TOKENIZER}" <<'PY'
import glob, json, sys, time
from transformers import AutoTokenizer
import pyarrow.parquet as pq
parquet_glob, out_path, tok_path = sys.argv[1], sys.argv[2], sys.argv[3]
THRESHOLD = 24 * 1024
BATCH = 32
tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
def prompt_text(p):
    if isinstance(p, str): return p
    return "\n".join(m.get("content","") for m in (p or []) if isinstance(m, dict) and m.get("content"))
n_in = n_out = 0; t0 = time.time()
with open(out_path, "w", encoding="utf-8") as fout:
    for fp in sorted(glob.glob(parquet_glob)):
        table = pq.read_table(fp)
        for start in range(0, table.num_rows, BATCH):
            recs = table.slice(start, BATCH).to_pylist()
            texts = [prompt_text(r.get("prompt")) for r in recs]
            enc = tok(texts, add_special_tokens=False)
            for r, ids in zip(recs, enc["input_ids"]):
                n_in += 1
                if len(ids) >= THRESHOLD:
                    row = dict(r); row["token_len"] = len(ids)
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_out += 1
        print(f"  {fp.rsplit('/',1)[-1]} done: cum_in={n_in} kept={n_out} ({time.time()-t0:.0f}s)", flush=True)
print(f"B1 output: {n_out}/{n_in} rows -> {out_path}")
PY
fi

# ---------- B2+B3 category filter + band window + slime jsonl ----------
echo "== [B2+B3] category filter + banding + slime jsonl (bands: ${BANDS}) =="
python3 - "${RAW_OUT}" "${DATA_DIR}" "${EXCLUDE_DS}" "${DATA_SUFFIX}" ${BANDS} <<'PY'
import json, sys, os, random, collections
src, data_dir, exclude, suffix = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
bands = sys.argv[5:]
BANDS = {
    "32k":  (24*1024,  28600, 4096),
    "64k":  (56*1024,  61380, 4096),
    "128k": (100*1024, 122800, 8192),
    "256k": (200*1024, 253900, 8192),
}
VAL_RATIO = 0.05
random.seed(42)
rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
for name in bands:
    lo, hi, resp = BANDS[name]
    out_dir = f"{data_dir}/band{name}{suffix}"
    os.makedirs(out_dir, exist_ok=True)
    train_path = f"{out_dir}/train.jsonl"
    val_path = f"{out_dir}/val.jsonl"
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"  band{name}: outputs present — SKIP")
        continue
    n_excl = n_lo = n_hi = 0; kept = []; by_ds = collections.Counter()
    for r in rows:
        if r.get("data_source") == exclude: n_excl += 1; continue
        tl = r.get("token_len", 0)
        if tl < lo: n_lo += 1; continue
        if tl > hi: n_hi += 1; continue
        rm = r.get("reward_model") or {}
        gt = rm.get("ground_truth") or {}
        extra = r.get("extra_info") or {}
        record = {
            "prompt": [dict(m) for m in r["prompt"]],
            "label": gt.get("golden_label"),
            "metadata": {
                "data_source": r.get("data_source"),
                "reward_mode": extra.get("reward_mode") or r.get("data_source"),
                "doc_ids": gt.get("doc_ids") or [],
                "summary": gt.get("summary") or "English",
                "index": extra.get("index"),
                "token_len": tl,
                "max_response_len": resp,
                "band": name,
            },
        }
        assert record["label"] is not None and record["prompt"]
        kept.append(record)
        by_ds[r.get("data_source")] += 1
    random.shuffle(kept)
    n_val = max(1, int(len(kept) * VAL_RATIO))
    val_rows, train_rows = kept[:n_val], kept[n_val:]
    for path, data in [(train_path, train_rows), (val_path, val_rows)]:
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {"train": len(train_rows), "val": len(val_rows), "by_ds": dict(by_ds)}
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    lens = sorted(r["metadata"]["token_len"] for r in kept)
    print(f"  band{name}: excl={n_excl} below={n_lo} above={n_hi} -> train={len(train_rows)} val={len(val_rows)} "
          f"(token min/med/max {lens[0] if lens else '-'}/{lens[len(lens)//2] if lens else '-'}/{lens[-1] if lens else '-'}) {dict(by_ds)}")
PY

# ---------- B4 validation ----------
echo "== [B4] three-key structure validation =="
python3 - "${DATA_DIR}" "${DATA_SUFFIX}" ${BANDS} <<'PY'
import json, sys, os
data_dir, suffix = sys.argv[1], sys.argv[2]
keys = {"prompt", "label", "metadata"}
for name in sys.argv[3:]:
    out_dir = f"{data_dir}/band{name}{suffix}"
    total = 0
    for split in ("train", "val"):
        p = f"{out_dir}/{split}.jsonl"
        n = 0
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            assert keys <= set(r) and isinstance(r["prompt"], list)
            assert r["metadata"]["band"] == name
            n += 1
        total += n
        print(f"  band{name}/{split}: {n} rows OK")
    print(f"  band{name} total: {total} rows")
PY
echo "GOLONGRL_DATA_PREP_OK"
#!/usr/bin/env bash
# ============================================================================
# GoLongRL parquet data preparation for the verl recipe (32k/64k/128k/256k).
#
# Same source and band windows as the slime/miles GoLongRL lines, plus a
# final verl 5-column schema conversion:
#   band  prompt window (tokens)       response  engine context ceiling
#   32k   [24576, 28600]  (24k-28k)    4096      32768
#   64k   [57344, 61380]  (56k-60k)    4096      65536
#   128k  [102400, 122800](100k-120k)  8192      131072
#   256k  [204800, 253900](200k-248k)  8192      262144
# Window ceiling = context - response - 52 template margin.
#
# Idempotent chain (existing artifacts are skipped; safe to resume):
#   B0 dataset download fallback: hf download Kwai-Klear/GoLongRL if missing
#   B1 full token count: Qwen3.5 tokenizer over all prompts, threshold 24k
#   B2 category filter: exclude multitableqa_pretraining
#   B3 band windows + 3-key jsonl (prompt/label/metadata)
#   B4 structure validation
#   B5 verl 5-column parquet: data_source / prompt / ability /
#      reward_model{style,ground_truth} / extra_info{index,split,reward_mode,
#      band,max_response_len} -> golongrl-verl-bands/band<N>k/{train,val}.parquet
#      (the layout tune_verl_golongrl.sh consumes)
#
# Prereq: golongrl_prepare_weights_verl.sh first (tokenizer); transformers,
# pyarrow, pandas available. B1 full tokenize takes ~1-2h (22965 rows).
#
# Usage (inside container):
#   DATA_DIR=/path/to/datas bash golongrl_prepare_data_verl.sh
#   BANDS="32k" DATA_DIR=/path/to/datas bash golongrl_prepare_data_verl.sh
# ============================================================================
set -euo pipefail

NAS_ROOT=${NAS_ROOT:-/workspace/golongrl}
BANDS="${BANDS:-32k 64k 128k 256k}"
DATA_DIR=${DATA_DIR:-${NAS_ROOT}/datas}
PARQUET_GLOB="${DATA_DIR}/GoLongRL/data/*.parquet"
TOKENIZER=${TOKENIZER:-${HF_MODEL_PATH:-${NAS_ROOT}/models/Qwen3.5-35B-A3B}}
RAW_OUT="${DATA_DIR}/prompts_over_24k.jsonl"
EXCLUDE_DS="multitableqa_pretraining"
BANDS_ROOT="${DATA_DIR}/golongrl-verl-bands"

mkdir -p "${DATA_DIR}" "${BANDS_ROOT}"

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

# ---------- B1 full token count (threshold 24k) ----------
echo "== [B1] full prompt token count (threshold 24k) =="
if [[ -s "${RAW_OUT}" ]]; then
    echo "  ${RAW_OUT} present ($(wc -l < ${RAW_OUT}) rows) — SKIP"
else
    python3 - "${PARQUET_GLOB}" "${RAW_OUT}" "${TOKENIZER}" <<'PY'
import glob, json, sys, time
from transformers import AutoTokenizer
parquet_glob, out_path, tok_path = sys.argv[1], sys.argv[2], sys.argv[3]
import pyarrow.parquet as pq
THRESHOLD = 24 * 1024            # 24k, lower bound of all four bands
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

# ---------- B2+B3 category filter + band windows + 3-key jsonl ----------
echo "== [B2+B3] category filter + band windows + 3-key jsonl (bands: ${BANDS}) =="
python3 - "${RAW_OUT}" "${DATA_DIR}" "${EXCLUDE_DS}" ${BANDS} <<'PY'
import json, sys, collections
src, data_dir, exclude = sys.argv[1], sys.argv[2], sys.argv[3]
bands = sys.argv[4:]
BANDS = {
    "32k":  (24*1024,        28600, 4096),
    "64k":  (56*1024,        61380, 4096),
    "128k": (100*1024,      122800, 8192),
    "256k": (200*1024,      253900, 8192),
}
rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
for name in bands:
    lo, hi, resp = BANDS[name]
    out = f"{data_dir}/golongrl_band{name}_miles.jsonl"
    n_excl = n_lo = n_hi = n_out = 0; lens = []; by_ds = collections.Counter()
    with open(out, "w", encoding="utf-8") as fout:
        for r in rows:
            if r.get("data_source") == exclude: n_excl += 1; continue
            tl = r.get("token_len", 0)
            if tl < lo: n_lo += 1; continue
            if tl > hi: n_hi += 1; continue
            rm = r.get("reward_model") or {}
            extra = r.get("extra_info") or {}
            row = {
                "prompt":   [dict(m) for m in r["prompt"]],
                "label":    rm.get("ground_truth"),
                "metadata": {"data_source": r.get("data_source"), "ability": r.get("ability"),
                             "reward_mode": extra.get("reward_mode"), "language": extra.get("language"),
                             "band": name, "max_response_len": resp},
            }
            assert row["label"] is not None and row["prompt"]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1; lens.append(tl); by_ds[r.get("data_source")] += 1
    lens.sort()
    print(f"  band{name}: in_excl={n_excl} below={n_lo} above={n_hi} -> {n_out} rows "
          f"(token min/med/max {lens[0] if lens else '-'}/{lens[len(lens)//2] if lens else '-'}/{lens[-1] if lens else '-'}) {dict(by_ds)}")
PY

# ---------- B4 structure validation ----------
echo "== [B4] 3-key structure validation =="
python3 - "${DATA_DIR}" ${BANDS} <<'PY'
import json, sys
keys = {"prompt", "label", "metadata"}
for name in sys.argv[2:]:
    p = f"{sys.argv[1]}/golongrl_band{name}_miles.jsonl"
    n = 0
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        assert keys <= set(r) and isinstance(r["prompt"], list) and isinstance(r["label"], dict)
        assert r["metadata"]["band"] == name
        n += 1
    print(f"  band{name}: {n} rows, 3-key structure OK")
PY

# ---------- B5 3-key jsonl -> verl 5-column parquet (per-band dirs) ----------
echo "== [B5] verl parquet conversion -> ${BANDS_ROOT}/band<N>k/{train,val}.parquet =="
python3 - "${DATA_DIR}" "${BANDS_ROOT}" ${BANDS} <<'PY'
import json, os, sys
import pandas as pd
src_dir, bands_root, bands = sys.argv[1], sys.argv[2], sys.argv[3:]
for band in bands:
    src = os.path.join(src_dir, f"golongrl_band{band}_miles.jsonl")
    rows = []
    with open(src) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj.get("metadata", {})
            # data_source routes the GoLongRL reward (longbench_pro /
            # selection / EM / math_longcot*); ground_truth keeps the label
            # dict as-is (doc_ids / golden_label / summary).
            rows.append({
                "data_source": meta.get("data_source", "longbench_pro"),
                "prompt": obj["prompt"],
                "ability": meta.get("ability", "unknown"),
                "reward_model": {"style": "rule", "ground_truth": obj.get("label")},
                "extra_info": {
                    "index": i, "split": "train",
                    "reward_mode": meta.get("reward_mode", ""),
                    "band": meta.get("band", band),
                    "max_response_len": meta.get("max_response_len", 4096),
                },
            })
    df = pd.DataFrame(rows)
    out_dir = os.path.join(bands_root, f"band{band}")
    os.makedirs(out_dir, exist_ok=True)
    train, val = df, df.iloc[: min(4, max(1, len(df) // 200))]
    train.to_parquet(os.path.join(out_dir, "train.parquet"))
    val.to_parquet(os.path.join(out_dir, "val.parquet"))
    print(f"[band {band}] total={len(df)} train={len(train)} val={len(val)}")
    print("  data_source counts:", df["data_source"].value_counts().to_dict())
PY

# ---------- final schema validation ----------
echo "== [final] verl parquet 5-column schema validation =="
python3 - "${BANDS_ROOT}" ${BANDS} <<'PY'
import sys, pandas as pd
bands_root, bands = sys.argv[1], sys.argv[2:]
need = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
for name in bands:
    for split in ("train", "val"):
        df = pd.read_parquet(f"{bands_root}/band{name}/{split}.parquet")
        assert need == list(df.columns), f"band{name}/{split}: columns {list(df.columns)}"
        assert (df["reward_model"].map(lambda x: x["style"] == "rule")).all()
        assert df["reward_model"].map(lambda x: "ground_truth" in x).all()
        assert (df["extra_info"].map(lambda x: x["band"] == name)).all()
        print(f"  band{name}/{split}.parquet: {len(df)} rows, 5-column schema OK")
PY
echo "== data preparation done (${BANDS} -> ${BANDS_ROOT}/band<N>k/{train,val}.parquet) =="
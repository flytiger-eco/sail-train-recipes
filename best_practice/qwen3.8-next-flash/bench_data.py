#!/usr/bin/env python3
"""Materialize fixed-length synthetic SFT data for Qwen3.8-Flash-Next perf runs.

Padding-free training makes the token count per step depend on the samples, so a
throughput number is only reproducible if every sample is the same length. This
script renders one assistant/user exchange per sample from repeated filler text
and pads/truncates it to exactly --seq-len tokens under the model's chat
template. With --mbs 1 --ga 8 every optimizer step therefore sees
8 * seq_len tokens, whatever the sequence length is.

Both turns carry text so that the response tokens (the ones that produce loss)
are roughly half the sample; a prompt-only sample would train on a handful of
tokens per sequence.
"""

import argparse
import json

from transformers import AutoTokenizer

FILLER = (
    "The history of distributed training systems shows that scaling laws are not only about "
    "parameter counts but also about how communication, memory hierarchy, and scheduling "
    "interact. In this note we revisit the design of a hybrid attention stack that mixes "
    "gated delta networks with block sparse attention, and we report measurements collected "
    "on a large accelerator cluster under realistic batch shapes. "
)


def rendered_len(tok, messages):
    return len(tok(tok.apply_chat_template(messages, tokenize=False), add_special_tokens=False)["input_ids"])


def build_sample(tok, text_ids, seq_len):
    probe = rendered_len(tok, [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}])
    budget = seq_len - (probe - 2)
    if budget < 64:
        raise SystemExit(f"seq_len {seq_len} leaves no room for content")
    half = budget // 2
    messages = [
        {"role": "user", "content": tok.decode(text_ids[:half])},
        {"role": "assistant", "content": tok.decode(text_ids[half: half * 2])},
    ]
    got = rendered_len(tok, messages)
    # The template can add or drop a token at the turn boundary; trim until it
    # lands exactly on seq_len so the step's token count is exact.
    while got > seq_len:
        cut = got - seq_len
        text = messages[1]["content"]
        ids = tok(text, add_special_tokens=False)["input_ids"][:-cut]
        messages[1]["content"] = tok.decode(ids)
        got = rendered_len(tok, messages)
    return messages, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--samples", type=int, default=160)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    filler = tok(FILLER * 2000, add_special_tokens=False)["input_ids"]

    rows, lengths = [], []
    for i in range(args.samples):
        offset = (i * 97) % 512  # vary the text a little so batches are not identical
        messages, got = build_sample(tok, filler[offset:], args.seq_len)
        rows.append({"messages": messages})
        lengths.append(got)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} samples to {args.out}: tokens/sample {sorted(set(lengths))}")


if __name__ == "__main__":
    main()

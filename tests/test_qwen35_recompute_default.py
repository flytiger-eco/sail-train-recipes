"""Recompute default of the 122B tuning script must follow the sequence length.

A manual 122B 8192-seq run that only calls tune_122b_fsdp.sh OOMs in iteration
0's first forward (141.1 GiB allocated, 768 MiB request failed): the script
defaulted RECOMPUTE_MODE to none, while the CI 8k cell passes full explicitly
and peaks at 66.6 GiB. The default now derives from SEQ_LEN, so the shape that
fits does not depend on copying CI's environment by hand.

The dry-run checks execute the script without torchrun; the argv printed by
run_exp is the same one a cluster node would get.
"""
import os
import re
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "best_practice" / "qwen3.5" / "tune_122b_fsdp.sh"

DERIVATION = re.compile(
    r'if \[\[ -n "\$\{RECOMPUTE_MODE:-\}" \]\]; then\n.*?\nelif \(\( SEQ_LEN >= (\d+) \)\); then\n'
    r'.*?RECOMPUTE_MODE="full".*?\nelse\n.*?RECOMPUTE_MODE="none"', re.DOTALL)


def dry_run(seq_len, recompute=None):
    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ.copy()
        for key in ("CP", "SEQ_LEN", "RECOMPUTE_MODE", "RECOMPUTE_MODULES",
                    "TUNE_DRY_RUN", "EXTRA_ARGS"):
            env.pop(key, None)
        env.update({"TUNE_DRY_RUN": "1", "OUTPUT_DIR": tmp, "SEQ_LEN": str(seq_len)})
        if recompute is not None:
            env["RECOMPUTE_MODE"] = recompute
        completed = subprocess.run(["bash", str(SCRIPT)], env=env, cwd=tmp,
                                   capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    line = next(line for line in completed.stdout.splitlines()
                if line.startswith("TUNE_DRY_RUN_CMD:"))
    return shlex.split(line.split(":", 1)[1])


class RecomputeDefaultTests(unittest.TestCase):
    def test_default_is_derived_from_sequence_length(self):
        text = SCRIPT.read_text()
        self.assertIsNone(re.search(r'RECOMPUTE_MODE="\$\{RECOMPUTE_MODE:-', text),
                          "RECOMPUTE_MODE 不应再硬编码默认值")
        match = DERIVATION.search(text)
        self.assertIsNotNone(match, "缺少按 SEQ_LEN 派生 RECOMPUTE_MODE 的分支")
        self.assertEqual(int(match.group(1)), 8192)
        self.assertIn("${RECOMPUTE_ARGS}", text)

    def test_8k_defaults_to_full_recompute(self):
        argv = dry_run(8192)
        self.assertIn("--enable-full-recompute", argv)
        self.assertNotIn("--disable-recompute", argv)

    def test_shorter_sequences_keep_recompute_off(self):
        argv = dry_run(4096)
        self.assertIn("--disable-recompute", argv)
        self.assertNotIn("--enable-full-recompute", argv)
        self.assertNotIn("--enable-recompute", argv)

    def test_explicit_mode_is_not_overridden(self):
        argv = dry_run(8192, "selective")
        self.assertIn("--enable-recompute", argv)
        self.assertIn("--recompute-modules", argv)
        self.assertNotIn("--enable-full-recompute", argv)
        self.assertNotIn("--disable-recompute", argv)


if __name__ == "__main__":
    unittest.main()

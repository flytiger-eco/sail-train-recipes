"""CP linear-attention mode chosen by the 122B tuning script.

122B-A10B cp2/16k deadlocked at the first GDN layer with the chunkwise path
(ranks split between build_cp_context and the MoE allgather, CP all-to-all
watchdog on an 8192x3072 qkvzba transfer). The fix is in the script itself:
CP>1 passes `legacy` (headwise in the VLM entrypoint), CP=1 keeps chunkwise.

The script is sourced in dry-run mode, so no torchrun and no training start;
the argv printed by run_exp is the same one the cluster would get.
"""
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "best_practice" / "qwen3.5" / "tune_122b_fsdp.sh"


def dry_run(cp):
    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ.copy()
        for key in ("CP", "LINEAR_CP_MODE", "TUNE_DRY_RUN", "EXTRA_ARGS"):
            env.pop(key, None)
        env.update({"TUNE_DRY_RUN": "1", "OUTPUT_DIR": tmp, "CP": str(cp)})
        completed = subprocess.run(["bash", str(SCRIPT)], env=env, cwd=tmp,
                                   capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    line = next(line for line in completed.stdout.splitlines()
                if line.startswith("TUNE_DRY_RUN_CMD:"))
    return shlex.split(line.split(":", 1)[1])


def flag_value(argv, flag):
    if flag not in argv:
        raise AssertionError(f"{flag} missing from {argv}")
    return argv[argv.index(flag) + 1]


class LinearCPModeTests(unittest.TestCase):
    def test_cp2_and_above_request_headwise(self):
        for cp in (2, 4, 8):
            with self.subTest(cp=cp):
                argv = dry_run(cp)
                self.assertEqual(flag_value(argv, "--cp"), str(cp))
                self.assertEqual(flag_value(argv, "--linear-cp-mode"), "legacy")
                self.assertNotEqual(flag_value(argv, "--linear-cp-mode"), "chunkwise")
                # Forced routing deadlocks the same GDN path under CP>1.
                self.assertNotIn("--moe-force-load-balancing", argv)

    def test_cp1_keeps_chunkwise(self):
        argv = dry_run(1)
        self.assertEqual(flag_value(argv, "--cp"), "1")
        self.assertEqual(flag_value(argv, "--linear-cp-mode"), "chunkwise")
        self.assertIn("--moe-force-load-balancing", argv)


if __name__ == "__main__":
    unittest.main()

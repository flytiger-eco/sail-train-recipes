"""Static pin for the 122B GDN CP mode (no bash, no GPU, no network).

The cp2/16k deadlock was fixed by making the script default to `legacy`
(headwise in the VLM entrypoint) at CP>1 while CP=1 keeps chunkwise. A dry-run
test already executes the script; this one parses it directly so the smallest
CI image still catches an edit that flips the CP>1 default back to chunkwise.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "best_practice" / "qwen3.5" / "tune_122b_fsdp.sh"

CP_ONE = 'if [[ "$CP" == "1" ]]; then'
CP_DEFAULT = 'CP="${CP:-1}"'
DEFAULT_MODE = re.compile(r'^LINEAR_CP_MODE="\$\{LINEAR_CP_MODE:-([a-z]+)\}"$')
ARGV_MODE = re.compile(r'^\s*--linear-cp-mode\s+"\$LINEAR_CP_MODE"\s*\\?$', re.MULTILINE)


def default_modes(text):
    """Branch of the CP guard that each LINEAR_CP_MODE default sits in."""
    modes, branch = {}, None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == CP_ONE:
            branch = "cp1"
        elif stripped == "else" and branch == "cp1":
            branch = "other"
        elif stripped == "fi":
            branch = None
        else:
            match = DEFAULT_MODE.match(stripped)
            if match and branch:
                modes[branch] = match.group(1)
    return modes


class LinearCPModeStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_cp_defaults_to_single_rank(self):
        self.assertIn(CP_DEFAULT, self.text)

    def test_cp1_keeps_chunkwise_and_cp_above_one_uses_legacy(self):
        self.assertEqual(default_modes(self.text), {"cp1": "chunkwise", "other": "legacy"})

    def test_cp_above_one_never_defaults_to_chunkwise(self):
        modes = default_modes(self.text)
        self.assertNotEqual(modes.get("other"), "chunkwise")
        self.assertEqual(modes.get("other"), "legacy")

    def test_argv_forwards_the_selected_mode(self):
        self.assertRegex(self.text, ARGV_MODE)


if __name__ == "__main__":
    unittest.main()

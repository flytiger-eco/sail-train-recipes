"""Structural checks for the patches the Dockerfiles apply at build time.

A hunk header that disagrees with its body silently breaks the image build:
the merged diff still reads fine, but `git apply` rejects the whole patch.
These checks need no network or Docker, so a PR fails fast instead of on the
next weekly build.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / "dockers" / "patches"
DOCKERFILES = sorted((ROOT / "dockers").glob("Dockerfile*"))

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
PATCH_SOURCE = re.compile(r"COPY\s+dockers/patches/(\S+)")


def _hunks(lines):
    hunks, current = [], None
    for number, line in enumerate(lines, 1):
        match = HUNK.match(line)
        if match:
            current = {"line": number, "header": line,
                       "old": int(match.group(2) or 1), "new": int(match.group(4) or 1),
                       "old_seen": 0, "new_seen": 0}
            hunks.append(current)
            continue
        # Quilt-style patches have no `diff --git`; a file section starts at the
        # `---`/`+++` pair. Removed lines never have a `+++` line right after.
        following = lines[number] if number < len(lines) else ""
        if current is None or line.startswith("diff --git ") or (
                line.startswith("--- ") and following.startswith("+++ ")):
            current = None if current is not None or line.startswith("diff --git ") else current
            continue
        if line.startswith("\\"):
            continue
        if line.startswith(" ") or line == "":
            current["old_seen"] += 1
            current["new_seen"] += 1
        elif line.startswith("-"):
            current["old_seen"] += 1
        elif line.startswith("+"):
            current["new_seen"] += 1
        else:
            current["bad"] = line
    return hunks


def validate(text):
    """Human-readable problems in one patch; empty means it is well formed."""
    if not text.strip():
        return []
    lines = text.splitlines()
    hunks = _hunks(lines)
    if not hunks:
        return [f"没有可识别的 @@ hunk：{lines[:1]}"]
    problems = []
    for hunk in hunks:
        if "bad" in hunk:
            problems.append(f"第 {hunk['line']} 行 hunk 内出现无法归类的内容：{hunk['bad'][:80]!r}")
        if (hunk["old_seen"], hunk["new_seen"]) != (hunk["old"], hunk["new"]):
            problems.append(f"第 {hunk['line']} 行 hunk 头声明 -{hunk['old']}/+{hunk['new']}，"
                            f"实际 -{hunk['old_seen']}/+{hunk['new_seen']}：{hunk['header']}")
    return problems


class PatchStructureTests(unittest.TestCase):
    def test_hunk_headers_match_their_bodies(self):
        patches = sorted(PATCHES.glob("*.patch"))
        self.assertTrue(patches)
        problems = [f"{path.name}: {problem}" for path in patches if path.stat().st_size
                    for problem in validate(path.read_text())]
        self.assertEqual(problems, [], "\n".join(problems))

    def test_patch_drift_is_reported_with_the_offending_hunk(self):
        broken = ("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
                  "@@ -10,3 +10,4 @@\n context\n-old\n+new\n+extra\n")
        self.assertEqual(validate(broken),
                         ["第 4 行 hunk 头声明 -3/+4，实际 -2/+3：@@ -10,3 +10,4 @@"])

    def test_truncated_hunk_is_reported(self):
        truncated = ("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n"
                     "@@ -10,3 +10,2 @@\n context\n-removed\nUNPREFIXED\n")
        problems = validate(truncated)
        self.assertIn("无法归类", problems[0])
        self.assertIn("-3/+2", problems[1])

    def test_empty_patches_are_guarded_by_the_dockerfile(self):
        unguarded = []
        for dockerfile in DOCKERFILES:
            text = dockerfile.read_text()
            for name in PATCH_SOURCE.findall(text):
                path = PATCHES / name
                if path.is_file() and not path.stat().st_size and f"-s /tmp/{name}" not in text:
                    unguarded.append(f"{dockerfile.name} 应用了空补丁却未用 [ -s /tmp/{name} ] 保护")
        self.assertEqual(unguarded, [])

    def test_every_patch_copied_by_a_dockerfile_exists(self):
        missing = []
        for dockerfile in DOCKERFILES:
            for name in PATCH_SOURCE.findall(dockerfile.read_text()):
                if not (PATCHES / name).is_file():
                    missing.append(f"{dockerfile.name} 引用缺失补丁 {name}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()

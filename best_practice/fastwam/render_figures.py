#!/usr/bin/env python3
"""从随文档发布的CSV生成SVG，或检查图表/双仓库资源；仅使用Python标准库。"""

import argparse
import ast
import csv
from decimal import Decimal, ROUND_HALF_EVEN
from html import escape
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

ASSETS = Path(__file__).resolve().parent / "assets"
MODES = ("off", "elementwise", "ffa", "both")
LABELS = ("全部关闭", "仅 elementwise", "仅 FFA", "elementwise + FFA")
COLORS = ("#64748b", "#2563eb", "#7c3aed", "#0d9488")


def read_csv(name):
    with (ASSETS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def import_m890(source):
    """维护入口：逐字保留历史CSV字段，生成正式的50步数据摘录。"""
    with source.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if 1 <= int(r["step"]) <= 50]
    check_series(rows, "mode", MODES)
    target = ASSETS / "robotwin-m890-first-50.csv"
    if target.exists():
        raise ValueError("数据摘录已存在；请核对来源后使用文件编辑工具更新，不自动覆盖")
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("mode", "step", "loss", "loss_action", "loss_video", "lr"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def check_series(rows, group_key=None, groups=(None,)):
    for group in groups:
        selected = [r for r in rows if group_key is None or r[group_key] == group]
        if [int(r["step"]) for r in selected] != list(range(1, 51)):
            raise ValueError(f"{group}: 必须恰好包含按顺序排列的第1至50步")
        for row in selected:
            values = [float(row[key]) for key in ("loss", "loss_action", "loss_video")]
            if not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError(f"非有限或负loss: {row}")
            if abs(values[0] - values[1] - values[2]) > 0.00016:
                raise ValueError(f"总loss与分项不符: {row}")
    if group_key is not None and set(r[group_key] for r in rows) != set(groups):
        raise ValueError("发现未声明的模式")


def audit_data():
    perf = read_csv("performance.csv")
    if len(perf) != 8 or set(r["dataset"] for r in perf) != {"LIBERO", "RoboTwin"}:
        raise ValueError("性能表必须包含两个数据集的四模式")
    for dataset, baseline in (("LIBERO", 0.929), ("RoboTwin", 1.1)):
        rows = [r for r in perf if r["dataset"] == dataset]
        if tuple(r["mode"] for r in rows) != MODES:
            raise ValueError("性能模式/顺序错误")
        off = float(rows[0]["mean_seconds_per_step"])
        for row in rows:
            duration = float(row["mean_seconds_per_step"])
            if abs(128 / duration - float(row["samples_per_second"])) > 0.011:
                raise ValueError("吞吐与global batch=128不符")
            computed = (off / duration - 1) * 100
            reported = float(row["throughput_improvement_pct"])
            # 用户原表的FFA和双开百分比与耗时有轻微出入，保留来源并显式报告。
            exceptions = {("LIBERO", "ffa"): (4.80, 4.81), ("LIBERO", "both"): (23.55, 23.65)}
            if (dataset, row["mode"]) in exceptions:
                expected_reported, expected_computed = exceptions[(dataset, row["mode"])]
                if reported != expected_reported or round(computed, 2) != expected_computed:
                    raise ValueError("LIBERO原表差异与发布注释不符")
                print(f"来源注释：{dataset}/{row['mode']} 原表 +{reported:.2f}%，按耗时重算 +{computed:.2f}%")
            elif abs(computed - reported) > 0.0051:
                raise ValueError("吞吐提升算术错误")
            if float(row["h200_seconds_per_step"]) != baseline:
                raise ValueError("混用了H200基线")
            if abs(baseline / duration - float(row["m890_over_h200"])) > 0.00051:
                raise ValueError("耗时比算术错误")
    m890 = read_csv("robotwin-m890-first-50.csv")
    h200 = read_csv("robotwin-h200-first-50.csv")
    check_series(m890, "mode", MODES)
    check_series(h200)
    off = [r for r in m890 if r["mode"] == "off"]
    ew = [r for r in m890 if r["mode"] == "elementwise"]
    for a, b in zip(off, ew):
        if any(a[k] != b[k] for k in ("loss", "loss_action", "loss_video", "lr")):
            raise ValueError("该历史轮次EW与OFF的50步记录应相同")
    quantize = lambda x: Decimal(x).quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)
    differences = [abs(quantize(a["loss"]) - quantize(b["loss"])) for a, b in zip(off, h200)]
    mae = sum(differences) / 50
    if mae != Decimal("0.033024") or max(differences) != Decimal("0.0928") or differences.index(max(differences)) != 20 or any(d == 0 for d in differences):
        raise ValueError("跨平台50步统计与发布口径不符")
    stability = read_csv("stability.csv")
    if len(stability) != 2:
        raise ValueError("缺少双场景稳定性记录")
    for row, evals in zip(stability, ("200/400/600/800/1000", "500/1000")):
        if row["mode"] != "both" or row["steps"] != "1000" or row["eval_steps"] != evals:
            raise ValueError("稳定性轮次或evaluate步数错误")
    json.loads((ASSETS / "sources.json").read_text(encoding="utf-8"))
    print(f"数据核对通过：四模式性能、200条M890记录、50条H200记录；OFF/H200 MAE={mae}")
    return perf, m890, h200


def begin(width, height, title, desc):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            f'<title>{escape(title)}</title><desc>{escape(desc)}</desc>',
            '<style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei",Arial,sans-serif;fill:#1e293b} .small{font-size:13px} .title{font-size:24px;font-weight:700}</style>',
            '<rect width="100%" height="100%" fill="#ffffff"/>']


def text(parts, x, y, label, size=15, anchor="start"):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(str(label))}</text>')


def line(parts, x1, y1, x2, y2, color="#e2e8f0", extra=""):
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" {extra}/>')


def finish(parts):
    return "\n".join(parts + ["</svg>", ""])


def performance_chart(perf):
    parts = begin(1160, 550, "FastWAM：M890四模式训练性能", "LIBERO和RoboTwin独立H200基线；柱高是M890吞吐占H200的条件比值，不是任务成功率")
    text(parts, 40, 42, "FastWAM · Zhenwu M890 四模式训练性能", 24)
    text(parts, 40, 70, "柱高：M890 / H200（相同 global batch 时的吞吐比）；柱顶：M890 秒/step", 15)
    for panel, (dataset, baseline) in enumerate((("LIBERO", "0.929"), ("RoboTwin", "1.1"))):
        left = 70 + panel * 570
        top, bottom, width = 130, 405, 460
        text(parts, left, 111, f"{dataset}  ·  H200 基线 {baseline} 秒/step", 18)
        for tick in range(0, 81, 20):
            y = bottom - tick / 80 * (bottom - top)
            line(parts, left, y, left + width, y)
            text(parts, left - 12, y + 5, f"{tick}%", 13, "end")
        rows = [r for r in perf if r["dataset"] == dataset]
        for i, row in enumerate(rows):
            ratio = float(row["m890_over_h200"])
            x = left + 24 + i * 115
            height = ratio / 0.8 * (bottom - top)
            parts.append(f'<rect x="{x}" y="{bottom - height:.3f}" width="70" height="{height:.3f}" rx="5" fill="{COLORS[i]}"/>')
            text(parts, x + 35, bottom - height - 28, f'{float(row["mean_seconds_per_step"]):.4f}s', 14, "middle")
            text(parts, x + 35, bottom - height - 8, f"{ratio * 100:.1f}%", 15, "middle")
            text(parts, x + 35, 434, ("关闭", "EW", "FFA", "EW + FFA")[i], 14, "middle")
        text(parts, left, 469, "EW 与 FFA 收益分别报告；不将组合收益全部归因于 FFA。", 13)
    text(parts, 40, 508, "来源：LIBERO 用户表；RoboTwin 历史四模式 CSV。H200 配置尚未完整核对，跨平台比值仅作参考。", 13)
    return finish(parts)


def loss_chart(m890, h200):
    parts = begin(1160, 885, "RoboTwin：M890与H200前50步loss", "总loss、动作loss、视频loss；M890关闭及双开来自同轮百步记录，H200来自用户四位小数摘录；无平滑或拟合")
    text(parts, 40, 40, "RoboTwin · Zhenwu M890 / H200 前 50 步 loss", 24)
    series = ((h200, "H200（用户基线）", "#d97706", ""),
              ([r for r in m890 if r["mode"] == "off"], "M890（全部关闭）", "#2563eb", ""),
              ([r for r in m890 if r["mode"] == "both"], "M890（EW + FFA）", "#0d9488", 'stroke-dasharray="5 3"'))
    for i, (_, label, color, dash) in enumerate(series):
        x = 55 + i * 340
        line(parts, x, 69, x + 32, 69, color, f'stroke-width="3" {dash}')
        text(parts, x + 42, 74, label)
    for panel, (key, label) in enumerate((("loss", "总 loss"), ("loss_action", "动作 loss"), ("loss_video", "视频 loss"))):
        left, right = 90, 1110
        top, bottom = 129 + panel * 235, 298 + panel * 235
        values = [float(r[key]) for rows, _, _, _ in series for r in rows]
        lo, hi = min(values), max(values)
        padding = (hi - lo) * 0.08
        lo, hi = lo - padding, hi + padding
        text(parts, left, top - 15, label, 17)
        for tick in range(5):
            value = lo + tick / 4 * (hi - lo)
            y = bottom - tick / 4 * (bottom - top)
            line(parts, left, y, right, y)
            text(parts, left - 12, y + 5, f"{value:.3f}", 13, "end")
        for step in (1, 10, 20, 30, 40, 50):
            x = left + (step - 1) / 49 * (right - left)
            line(parts, x, top, x, bottom)
            text(parts, x, bottom + 22, step, 13, "middle")
        for rows, _, color, dash in series:
            points = " ".join(f'{left + (int(r["step"]) - 1) / 49 * (right - left):.3f},{bottom - (float(r[key]) - lo) / (hi - lo) * (bottom - top):.3f}' for r in rows)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" {dash}/>')
    text(parts, 600, 812, "训练 step（仅第 1–50 步；无平滑、拟合或移动平均）", 14, "middle")
    text(parts, 40, 846, "OFF / H200：统一 4 位小数后 MAE 0.033024；最大绝对差 0.0928（第 21 步）；相同 0/50 步。", 14)
    text(parts, 40, 870, "两平台并未逐步对齐；配置与随机过程尚未完整核对，不据此判定测试集精度等价或归因于 FFA。", 13)
    return finish(parts)


def architecture_chart():
    parts = begin(1160, 510, "FastWAM训练与动作推理示意", "根据官方论文及当前FastWAM配置整理的概念示意，不是原论文图；推理不生成未来视频但仍迭代去噪动作")
    text(parts, 40, 42, "FastWAM · 保留训练期世界建模，省去推理期未来视频生成", 24)
    def box(x, y, width, label, detail, color):
        parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="82" rx="10" fill="{color}" stroke="#cbd5e1"/>')
        text(parts, x + width / 2, y + 32, label, 17, "middle")
        text(parts, x + width / 2, y + 59, detail, 13, "middle")
    text(parts, 40, 96, "训练：视频与动作联合 Flow Matching；结构化 mask 限定信息依赖", 18)
    box(40, 118, 240, "多相机视频 → VAE", "首帧条件 + 加噪未来 latent", "#eff6ff")
    box(355, 118, 330, "视频 DiT + ActionDiT（MoT）", "30 层 / 分支；注意力 head_dim=128", "#ecfdf5")
    box(760, 118, 340, "视频 loss + 动作 loss", "文本缓存与本体状态提供条件", "#f5f3ff")
    line(parts, 285, 159, 347, 159, "#475569", 'stroke-width="2"')
    text(parts, 308, 152, "→", 21)
    line(parts, 690, 159, 752, 159, "#475569", 'stroke-width="2"')
    text(parts, 713, 152, "→", 21)
    text(parts, 40, 247, "动作推理：只保留当前首帧，不生成未来视频", 18)
    box(40, 270, 240, "当前观测 → VAE", "文本 + 本体状态", "#eff6ff")
    box(355, 270, 330, "视频骨干编码首帧一次", "得到动作专家可复用的世界表征", "#ecfdf5")
    box(760, 270, 340, "动作专家迭代去噪", "输出 action chunk（32 个动作）", "#fff7ed")
    line(parts, 285, 311, 347, 311, "#475569", 'stroke-width="2"')
    text(parts, 308, 304, "→", 21)
    line(parts, 690, 311, 752, 311, "#475569", 'stroke-width="2"')
    text(parts, 713, 304, "→", 21)
    text(parts, 40, 411, "结构化依赖：首帧不读取未来；动作读取首帧和动作 token，不读取未来视频 token。", 16)
    text(parts, 40, 443, "“省去未来视频生成”不等于删除训练视频目标，也不等于动作生成只需一次前向。", 15)
    text(parts, 40, 477, "来源：Fast-WAM 论文 §3；configs/model/fastwam.yaml。本图为概念整理，非原论文图。", 13)
    return finish(parts)


def check_links(document):
    content = document.read_text(encoding="utf-8")
    if content.count("```") % 2:
        raise ValueError("代码围栏未配对")
    for language, block in re.findall(r"```(\w*)\n(.*?)```", content, flags=re.S):
        if language == "bash":
            subprocess.run(["bash", "-n"], input=block, text=True, check=True, capture_output=True)
            for python_block in re.findall(r"python - <<'PY'\n(.*?)\nPY", block, flags=re.S):
                ast.parse(python_block)
    published = re.findall(r"^\| (全部关闭|仅 elementwise|仅 FFA|elementwise \+ FFA) \| ([^\n]+)\|$", content, flags=re.M)
    if published:
        expected = read_csv("performance.csv")
        if len(published) != len(expected):
            raise ValueError(f"文档缺少四模式性能行：{document}")
        for (label, values), row in zip(published, expected):
            i = MODES.index(row["mode"])
            columns = [v.strip() for v in values.split("|")]
            boost = "—" if i == 0 else f"+{float(row['throughput_improvement_pct']):.2f}%"
            correct = [f"{float(row['mean_seconds_per_step']):.4f}", f"{float(row['samples_per_second']):.2f}", boost, f"{float(row['m890_over_h200']):.3f}×"]
            if label != LABELS[i] or columns != correct:
                raise ValueError(f"文档性能行与CSV不一致：{document}: {label}")
    content = re.sub(r"```.*?```", "", content, flags=re.S)
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content):
        if re.match(r"(?:https?://|#|mailto:)", target):
            continue
        path = (document.parent / target.split("#", 1)[0]).resolve()
        if not path.exists():
            raise ValueError(f"失效相对链接：{document}: {target}")


def audit_provenance(root):
    """可选：核对本地历史产物，仅只读，不运行训练或使用设备。"""
    base = root / "scratch/robotwin_perf_trace_20261106/remaining_tail_20261107/robotwin_four_modes_20261107"
    with (base / "loss_all_modes.csv").open(encoding="utf-8", newline="") as stream:
        original = list(csv.DictReader(stream))
    subset = [r for r in original if int(r["step"]) <= 50]
    if subset != read_csv("robotwin-m890-first-50.csv"):
        raise ValueError("M890发布数据不等于原始CSV的50步摘录")
    by_mode = {mode: [r for r in original if r["mode"] == mode] for mode in MODES}
    for a, b in zip(by_mode["off"], by_mode["elementwise"]):
        if any(a[k] != b[k] for k in ("loss", "loss_action", "loss_video", "lr")):
            raise ValueError("百步EW/OFF记录不一致")
    for mode, expected_max in (("ffa", 0.00118017), ("both", 0.000876427)):
        value = max(abs(float(a["loss"]) - float(b["loss"])) for a, b in zip(by_mode["off"], by_mode[mode]))
        if abs(value - expected_max) > 5e-9:
            raise ValueError("百步FFA绝对误差与文档不符")
    with (base / "performance_table.csv").open(encoding="utf-8", newline="") as stream:
        original_perf = list(csv.DictReader(stream))
    published_perf = [r for r in read_csv("performance.csv") if r["dataset"] == "RoboTwin"]
    for a, b in zip(original_perf, published_perf):
        for key in ("mean_seconds_per_step", "samples_per_second", "throughput_improvement_pct"):
            if a[key] != b[key]:
                raise ValueError("RoboTwin性能数据与原CSV不符")
    paths = ("libero", "robotwin_retry1")
    for row, path in zip(read_csv("stability.csv"), paths):
        result = json.loads((root / f"scratch/launchers_both_stability_20261111/{path}/result.json").read_text(encoding="utf-8"))
        if result["stability"] != "PASS" or result["steps"] != 1000 or result["steady_steps"] != [21, 1000]:
            raise ValueError("原千步报告状态或区间不符")
        for column, key in (("mean_seconds_per_step", "mean_seconds"), ("cv_pct", "cv_pct"), ("max_seconds_per_step", "max_seconds")):
            if float(row[column]) != result[key]:
                raise ValueError("千步统计与原报告不符")
        if row["eval_steps"] != "/".join(str(event["step"]) for event in result["evals"]):
            raise ValueError("evaluate步数与原报告不符")
    print("原始产物核对通过：50步摘录、百步误差、四模式性能、两数据集千步报告")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-m890", type=Path, help="首次导入原四模式CSV，不覆盖已有摘录")
    parser.add_argument("--publish-dir", type=Path, help="同步正式图表和CSV到另一文档仓库的资源目录")
    parser.add_argument("--check", action="store_true", help="只核验数据、SVG内容及资源副本，不写入")
    parser.add_argument("--documents", type=Path, nargs="*", default=[], help="核验Markdown链接、性能表和Bash/Python示例语法，不执行示例")
    parser.add_argument("--source-artifacts", type=Path, help="可选的历史产物根目录，用于核对CSV摘录和千步数据来源")
    args = parser.parse_args()
    if args.check and args.import_m890:
        parser.error("只读检查不能导入数据")
    if args.import_m890:
        import_m890(args.import_m890)
    perf, m890, h200 = audit_data()
    if args.source_artifacts:
        audit_provenance(args.source_artifacts)
    figures = {"performance-four-modes.svg": performance_chart(perf),
               "robotwin-loss-first-50.svg": loss_chart(m890, h200),
               "architecture.svg": architecture_chart()}
    for name, content in figures.items():
        ET.fromstring(content)
        path = ASSETS / name
        if args.check:
            if path.read_text(encoding="utf-8") != content:
                raise ValueError(f"图表与CSV不一致：{path}")
        else:
            path.write_text(content, encoding="utf-8")
    names = tuple(figures) + ("performance.csv", "robotwin-h200-first-50.csv", "robotwin-m890-first-50.csv", "stability.csv", "sources.json")
    if args.publish_dir:
        if args.publish_dir.resolve() == ASSETS.resolve():
            parser.error("发布目录不得与源资源目录相同")
        if not args.check:
            args.publish_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            target = args.publish_dir / name
            if args.check:
                if target.read_bytes() != (ASSETS / name).read_bytes():
                    raise ValueError(f"双仓库资源不一致：{target}")
            else:
                shutil.copyfile(ASSETS / name, target)
    for document in args.documents:
        check_links(document)
    print("图表与资源核验通过" if args.check else "SVG生成及正式资源同步完成")


if __name__ == "__main__":
    main()

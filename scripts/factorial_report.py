"""Verify a downloaded formal campaign, archive summaries and write its report."""

import argparse
import json
import shutil
from pathlib import Path

import torch

from .factorial_campaign import paired_effects, schedule, seed_for
from .factorial_pipeline import read_plan
from .kaggle_campaign import file_sha, write_json
from .verify_kaggle_smoke import plan_digest


def collect(download, archive, revision):
    download, archive = Path(download), Path(archive)
    base = download / "ferminet_results"
    plan = read_plan(False)
    campaign = json.loads((base / "campaign.json").read_text())
    if campaign["commit"] != revision or campaign["plan"] != plan:
        raise ValueError("Downloaded campaign source/plan differs")
    artifacts = json.loads((base / "archive_manifest.json").read_text())
    for name, digest in artifacts.items():
        target = base / name
        if not target.is_file() or file_sha(target) != digest:
            raise ValueError(f"Missing or changed artifact: {name}")
    rows = []
    for queue in schedule(plan):
        for case in queue:
            folder = base / case["case_id"]
            result = (
                json.loads((folder / "result.json").read_text())
                if (folder / "result.json").exists()
                else dict(**case, status="missing")
            )
            if result["status"] == "completed":
                checkpoint = folder / "final_checkpoint.pt"
                evaluation = result["evaluation"]
                if (
                    result["epochs"] != 10000
                    or result["parent_checkpoint_sha256"]
                    != plan["fork"]["checkpoint_sha256"][str(case["seed"])]
                    or file_sha(checkpoint) != result["checkpoint_sha256"]
                    or evaluation["time_records"] != 2048
                    or evaluation["seed"] != seed_for(plan, case["seed"], case["arm"])
                    or evaluation["num_invalid"]
                ):
                    raise ValueError(
                        "Final result identity/update/evaluation count differs"
                    )
                trajectory = torch.load(
                    folder / "evaluation/energy.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                if (
                    trajectory.shape != (2048, 512)
                    or not torch.isfinite(trajectory).all()
                ):
                    raise ValueError("Final energy trajectory differs")
                # Recompute statistical summaries from the unmodified trajectory.
                from ferminet_pytorch.vmc_statistics import energy_statistics

                recomputed = energy_statistics(trajectory)
                for key in ("energy_mean", "energy_sem"):
                    if abs(recomputed[key] - evaluation[key]) > 1e-12:
                        raise ValueError("Frozen statistics do not reproduce")
            rows.append(result)
    inference = paired_effects(rows, plan["seeds"], plan["practical_improvement_mha"])
    archive.mkdir(parents=True, exist_ok=False)
    for path in base.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".log"):
            target = archive / "artifacts" / path.relative_to(base)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    for path in download.glob("*.log"):
        shutil.copyfile(path, archive / path.name)
    write_json(
        archive / "verified_summary.json",
        {
            "source_commit": revision,
            "long_plan_sha256": plan_digest(plan),
            "source_download": str(download.resolve()),
            "artifact_count": len(artifacts),
            "cases": rows,
            "inference": inference,
        },
    )
    lines = [
        "# FermiNet 裁剪 × 包络受控实验",
        "",
        f"固定源码：`{revision}`；从三个 50000 轮父检查点分叉，每组 10000 次更新。",
        "本结果仅比较已训练 Gaussian 解的迁移，不代表从头训练或其他主干。",
        "",
        "| 组别 | 种子 | 状态 | 参考偏差 mHa | SEM mHa | 化学精度 | 采样诊断 |",
        "|---|---:|---|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (r["arm"], r["seed"])):
        if row["status"] != "completed":
            lines.append(
                f"| {row['arm']} | {row['seed']} | {row['status']} | — | — | — | — |"
            )
        else:
            e = row["evaluation"]
            lines.append(
                f"| {row['arm']} | {row['seed']} | completed | {e['abs_error_mha']:.6f} | {1000 * e['energy_sem']:.6f} | {e['chemical_accuracy_status']} | {e['diagnostics_status']} |"
            )
    lines += ["", f"交叉实验统计状态：**{inference['status']}**。"]
    if inference["effects"]:
        lines += [
            "",
            "| 效应 | 配对均值 mHa | Bonferroni 近似区间 mHa | 达到预设改善标准 |",
            "|---|---:|---|---|",
        ]
        for name, effect in inference["effects"].items():
            lo, hi = effect["ci_bonferroni_mha"]
            lines.append(
                f"| {name} | {effect['mean_mha']:.6f} | [{lo:.6f}, {hi:.6f}] | {effect['improvement_supported']} |"
            )
    lines += [
        "",
        "四组为 A Gaussian+tanh、B Gaussian+mean_hard、C soft_exponential+tanh、D soft_exponential+mean_hard。",
        "负的主效应表示能量降低；交互项表示非加性，不能独立解释为某个方案的总收益。",
        "三种子的配对区间为近似区间，采用种子间标准误与传播 Monte Carlo 标准误的较大者。",
        "裁剪对照改变一整套规则，不能拆分为尺度或硬裁剪的单独因果结论。",
        "",
        "原始零步评估、最终空间尾部贡献、数值审计和所有状态见 verified_summary.json 及 artifacts/。",
        "缺组、失败或采样诊断不足时不发布完整交叉实验结论。大型检查点与坐标分片保留在 source_download 指定目录。",
    ]
    report = archive / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        archive / "archive_identity.json",
        {
            "source_download": str(download.resolve()),
            "sha256": {
                str(p.relative_to(archive)): file_sha(p)
                for p in archive.rglob("*")
                if p.is_file()
            },
        },
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("download", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(collect(args.download, args.archive, args.revision))


if __name__ == "__main__":
    main()

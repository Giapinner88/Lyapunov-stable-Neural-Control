import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


EPOCH_PATTERN = re.compile(
    r"CEGIS Epoch\s+(?P<epoch>\d+)\s*\|\s*"
    r"Quy\s*m[oô]\s*Box:\s*(?P<box>[0-9.]+)%\s*\|\s*"
    r"Bank:\s*(?P<bank>\d+)\s*\|\s*"
    r"Loss:\s*(?P<loss>[-+0-9.eE]+)\s*\|\s*"
    r"Max Violt:\s*(?P<maxv>[-+0-9.eE]+)\s*\|\s*"
    r"Mean Violt:\s*(?P<meanv>[-+0-9.eE]+)\s*\|\s*"
    r"(?:"
    r"(?P<rho_label>rho|ρ)=(?P<rho>[-+0-9.eE]+)"
    r"|(?P<pending>rho=pending|ρ=pending)"
    r")",
    flags=re.IGNORECASE,
)


@dataclass
class EpochStat:
    epoch: int
    box_percent: float
    bank_size: int
    loss: float
    max_violation: float
    mean_violation: float
    rho: Optional[float]


@dataclass
class EvalSummary:
    total_tests: Optional[int] = None
    convergence_rate: Optional[float] = None
    lyapunov_decrease_rate: Optional[float] = None
    stabilization_rate: Optional[float] = None


@dataclass
class VerifySummary:
    empirical_rho: Optional[float] = None
    verified_rho: Optional[float] = None
    roa_ratio: Optional[float] = None
    roa_volume: Optional[float] = None
    crown_local_radius: Optional[float] = None


@dataclass
class BestCheckpointSelection:
    metric: str
    selected_epoch: Optional[int]
    selected_value: Optional[float]
    controller_path: Optional[str]
    lyapunov_path: Optional[str]
    used_snapshot: bool
    note: str


def _latest_file(paths: list[Path]) -> Optional[Path]:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def find_latest_train_log(repo_root: Path) -> Optional[Path]:
    candidates = sorted((repo_root / "reports").glob("*.log"))
    candidates += sorted(repo_root.glob("*.log"))
    return _latest_file(candidates)


def find_latest_eval_summary(repo_root: Path) -> Optional[Path]:
    candidates = sorted((repo_root / "reports").glob("**/eval_summary.txt"))
    candidates += sorted((repo_root / "evaluation_results").glob("**/eval_summary.txt"))
    return _latest_file(candidates)


def find_latest_verify_summary(repo_root: Path) -> Optional[Path]:
    candidates = sorted((repo_root / "reports").glob("**/verification_summary.txt"))
    candidates += sorted((repo_root / "verification_results").glob("**/verification_summary.txt"))
    return _latest_file(candidates)


def _extract_run_timestamp_from_log_name(path: Path) -> Optional[datetime]:
    m = re.search(r"(\d{8}_\d{6})\.log$", path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _pick_summary_for_run(log_path: Path, candidates: list[Path], window_hours: float = 8.0) -> Optional[Path]:
    if not candidates:
        return None

    log_ts = _extract_run_timestamp_from_log_name(log_path)
    if log_ts is None:
        return _latest_file(candidates)

    selected: Optional[Path] = None
    selected_delta: Optional[float] = None
    for p in candidates:
        file_ts = datetime.fromtimestamp(p.stat().st_mtime)
        delta = (file_ts - log_ts).total_seconds()
        if delta < 0 or delta > window_hours * 3600:
            continue
        if selected is None or delta < selected_delta:
            selected = p
            selected_delta = delta

    if selected is not None:
        return selected
    return _latest_file(candidates)


def find_latest_run_bundle(repo_root: Path) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    train_log = find_latest_train_log(repo_root)
    if train_log is None:
        return None, find_latest_eval_summary(repo_root), find_latest_verify_summary(repo_root)

    eval_candidates = sorted((repo_root / "reports").glob("**/eval_summary.txt"))
    verify_candidates = sorted((repo_root / "reports").glob("**/verification_summary.txt"))
    eval_candidates += sorted((repo_root / "evaluation_results").glob("**/eval_summary.txt"))
    verify_candidates += sorted((repo_root / "verification_results").glob("**/verification_summary.txt"))

    eval_summary = _pick_summary_for_run(train_log, eval_candidates)
    verify_summary = _pick_summary_for_run(train_log, verify_candidates)
    return train_log, eval_summary, verify_summary


def parse_train_log(log_path: Path) -> list[EpochStat]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    # Normalize wrapped lines from terminal width (keep numeric content intact).
    text = text.replace("\n", " ")

    stats_by_epoch: dict[int, EpochStat] = {}
    for m in EPOCH_PATTERN.finditer(text):
        epoch = int(m.group("epoch"))
        box_percent = float(m.group("box"))
        bank_size = int(m.group("bank"))
        loss = float(m.group("loss"))
        max_v = float(m.group("maxv"))
        mean_v = float(m.group("meanv"))
        rho_str = m.group("rho")
        rho = float(rho_str) if rho_str is not None else None

        stats_by_epoch[epoch] = EpochStat(
            epoch=epoch,
            box_percent=box_percent,
            bank_size=bank_size,
            loss=loss,
            max_violation=max_v,
            mean_violation=mean_v,
            rho=rho,
        )

    return [stats_by_epoch[k] for k in sorted(stats_by_epoch)]


def parse_eval_summary(path: Optional[Path]) -> EvalSummary:
    if path is None or not path.exists():
        return EvalSummary()

    txt = path.read_text(encoding="utf-8", errors="ignore")
    out = EvalSummary()

    m = re.search(r"Total tests:\s*(\d+)", txt)
    if m:
        out.total_tests = int(m.group(1))

    m = re.search(r"Convergence rate:\s*([0-9.]+)%", txt)
    if m:
        out.convergence_rate = float(m.group(1))

    m = re.search(r"Lyapunov decrease rate:\s*([0-9.]+)%", txt)
    if m:
        out.lyapunov_decrease_rate = float(m.group(1))

    m = re.search(r"Stabilization rate:\s*([0-9.]+)%", txt)
    if m:
        out.stabilization_rate = float(m.group(1))

    return out


def parse_verify_summary(path: Optional[Path]) -> VerifySummary:
    if path is None or not path.exists():
        return VerifySummary()

    txt = path.read_text(encoding="utf-8", errors="ignore")
    out = VerifySummary()

    m = re.search(r"Empirical\s*(?:rho|ρ):\s*([0-9.eE+-]+)", txt)
    if m:
        out.empirical_rho = float(m.group(1))

    m = re.search(r"Verified\s*(?:rho|ρ):\s*([0-9.eE+-]+)", txt)
    if m:
        out.verified_rho = float(m.group(1))

    m = re.search(r"ROA\s*Ratio\s*Raw:\s*([0-9.eE+-]+)", txt)
    if m:
        out.roa_ratio = float(m.group(1))
    m = re.search(r"ROA Ratio in Box:\s*([0-9.eE+-]+)%", txt)
    if m:
        out.roa_ratio = float(m.group(1)) / 100.0

    m = re.search(r"Estimated\s+ROA\s+Volume\s+Raw:\s*([0-9.eE+-]+)", txt)
    if m:
        out.roa_volume = float(m.group(1))
    m = re.search(r"Estimated ROA Volume:\s*([0-9.eE+-]+)", txt)
    if m:
        out.roa_volume = float(m.group(1))

    m = re.search(r"CROWN Local Certified Radius \(L_inf\):\s*([0-9.eE+-]+)", txt)
    if m:
        out.crown_local_radius = float(m.group(1))

    return out


def save_epoch_csv(stats: list[EpochStat], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "box_percent", "bank_size", "loss", "max_violation", "mean_violation", "rho"])
        for s in stats:
            w.writerow([s.epoch, s.box_percent, s.bank_size, s.loss, s.max_violation, s.mean_violation, s.rho])


def make_plots(stats: list[EpochStat], out_png: Path) -> None:
    if not stats:
        return

    epochs = [s.epoch for s in stats]
    losses = [s.loss for s in stats]
    max_v = [s.max_violation for s in stats]
    mean_v = [s.mean_violation for s in stats]
    rho = [s.rho for s in stats]
    box = [s.box_percent for s in stats]
    bank = [s.bank_size for s in stats]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(epochs, losses, marker="o", linewidth=1.5)
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(epochs, max_v, marker="o", label="max_violation")
    ax.plot(epochs, mean_v, marker="o", label="mean_violation")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_title("Violation Trends")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Violation")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    rho_y = [x if x is not None else float("nan") for x in rho]
    ax.plot(epochs, rho_y, marker="o")
    ax.set_title("Rho Over Time")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("rho")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax2 = ax.twinx()
    p1 = ax.plot(epochs, bank, marker="o", color="tab:blue", label="bank_size")
    p2 = ax2.plot(epochs, box, marker="o", color="tab:orange", label="box_percent")
    ax.set_title("Bank and Curriculum Scale")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bank Size", color="tab:blue")
    ax2.set_ylabel("Box Percent", color="tab:orange")
    ax.grid(alpha=0.25)
    lines = p1 + p2
    labels = [p.get_label() for p in lines]
    ax.legend(lines, labels, loc="best")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def build_summary(stats: list[EpochStat], eval_sum: EvalSummary, verify_sum: VerifySummary) -> dict:
    summary: dict = {
        "num_checkpoints": len(stats),
        "train": {},
        "evaluation": asdict(eval_sum),
        "verification": asdict(verify_sum),
    }

    if stats:
        summary["train"] = {
            "first_epoch": stats[0].epoch,
            "last_epoch": stats[-1].epoch,
            "last_loss": stats[-1].loss,
            "best_loss": min(s.loss for s in stats),
            "last_max_violation": stats[-1].max_violation,
            "most_negative_max_violation": min(s.max_violation for s in stats),
            "last_mean_violation": stats[-1].mean_violation,
            "last_box_percent": stats[-1].box_percent,
            "last_bank_size": stats[-1].bank_size,
            "last_rho": next((s.rho for s in reversed(stats) if s.rho is not None), None),
        }

    return summary


def select_best_checkpoint(
    stats: list[EpochStat],
    metric: str,
    snapshots_dir: Path,
    final_controller_path: Path,
    final_lyapunov_path: Path,
) -> BestCheckpointSelection:
    if not stats:
        return BestCheckpointSelection(
            metric=metric,
            selected_epoch=None,
            selected_value=None,
            controller_path=None,
            lyapunov_path=None,
            used_snapshot=False,
            note="No stats parsed from train log.",
        )

    metric = metric.lower()
    if metric == "loss":
        best = min(stats, key=lambda s: s.loss)
        value = best.loss
    elif metric == "max_violation":
        best = min(stats, key=lambda s: s.max_violation)
        value = best.max_violation
    elif metric == "mean_violation":
        best = min(stats, key=lambda s: s.mean_violation)
        value = best.mean_violation
    elif metric == "rho":
        with_rho = [s for s in stats if s.rho is not None]
        if with_rho:
            best = max(with_rho, key=lambda s: float(s.rho))
            value = float(best.rho)
        else:
            best = stats[-1]
            value = None
    else:
        raise ValueError(f"Unsupported best metric: {metric}")

    ep_tag = f"ep{best.epoch:03d}"
    c_snapshot = snapshots_dir / f"cartpole_controller_{ep_tag}.pth"
    v_snapshot = snapshots_dir / f"cartpole_lyapunov_{ep_tag}.pth"

    if c_snapshot.exists() and v_snapshot.exists():
        return BestCheckpointSelection(
            metric=metric,
            selected_epoch=best.epoch,
            selected_value=value,
            controller_path=str(c_snapshot),
            lyapunov_path=str(v_snapshot),
            used_snapshot=True,
            note="Matched snapshot files by epoch.",
        )

    return BestCheckpointSelection(
        metric=metric,
        selected_epoch=best.epoch,
        selected_value=value,
        controller_path=str(final_controller_path),
        lyapunov_path=str(final_lyapunov_path),
        used_snapshot=False,
        note="Snapshot for selected epoch not found; fallback to final checkpoint.",
    )


def write_summary(
    summary: dict,
    out_json: Path,
    out_txt: Path,
    train_log: Optional[Path],
    eval_path: Optional[Path],
    verify_path: Optional[Path],
    best_selection: Optional[BestCheckpointSelection],
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("Training Analysis Summary")
    lines.append("=" * 40)
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Train log: {train_log}")
    lines.append(f"Eval summary: {eval_path}")
    lines.append(f"Verify summary: {verify_path}")
    lines.append("")

    train = summary.get("train", {})
    if train:
        lines.append("[Train]")
        for k, v in train.items():
            lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("[Evaluation]")
    for k, v in summary.get("evaluation", {}).items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("[Verification]")
    for k, v in summary.get("verification", {}).items():
        lines.append(f"- {k}: {v}")

    if best_selection is not None:
        lines.append("")
        lines.append("[Best Checkpoint Selection]")
        for k, v in asdict(best_selection).items():
            lines.append(f"- {k}: {v}")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze cartpole train/eval/verify outputs")
    parser.add_argument("--train-log", type=str, default=None, help="Path to train log (*.log)")
    parser.add_argument("--eval-summary", type=str, default=None, help="Path to eval_summary.txt")
    parser.add_argument("--verify-summary", type=str, default=None, help="Path to verification_summary.txt")
    parser.add_argument("--output-dir", type=str, default="reports/analysis", help="Output directory")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag for output folder")
    parser.add_argument("--select-best-checkpoint", action="store_true", help="Select best checkpoint from parsed epochs")
    parser.add_argument(
        "--best-metric",
        type=str,
        default="max_violation",
        choices=["loss", "max_violation", "mean_violation", "rho"],
        help="Metric used to choose best checkpoint",
    )
    parser.add_argument("--snapshots-dir", type=str, default="checkpoints/cartpole/snapshots")
    parser.add_argument("--controller-path", type=str, default="checkpoints/cartpole/cartpole_controller.pth")
    parser.add_argument("--lyapunov-path", type=str, default="checkpoints/cartpole/cartpole_lyapunov.pth")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent

    if args.train_log or args.eval_summary or args.verify_summary:
        train_log = Path(args.train_log) if args.train_log else find_latest_train_log(repo_root)
        eval_path = Path(args.eval_summary) if args.eval_summary else find_latest_eval_summary(repo_root)
        verify_path = Path(args.verify_summary) if args.verify_summary else find_latest_verify_summary(repo_root)
    else:
        train_log, eval_path, verify_path = find_latest_run_bundle(repo_root)

    if train_log is not None and not train_log.exists():
        raise FileNotFoundError(f"Train log path does not exist: {train_log}")

    if train_log is None:
        print("[Analyze] Warning: No train log found; proceeding with eval/verify summaries only.")

    stats = parse_train_log(train_log) if train_log is not None else []
    eval_sum = parse_eval_summary(eval_path)
    verify_sum = parse_verify_summary(verify_path)

    if train_log is None and eval_path is None and verify_path is None:
        raise FileNotFoundError(
            "No train/eval/verify artifacts found. Provide --train-log, --eval-summary, or --verify-summary explicitly."
        )

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / tag

    csv_path = out_dir / "epoch_metrics.csv"
    png_path = out_dir / "training_diagnostics.png"
    json_path = out_dir / "analysis_summary.json"
    txt_path = out_dir / "analysis_summary.txt"

    save_epoch_csv(stats, csv_path)
    make_plots(stats, png_path)

    summary = build_summary(stats, eval_sum, verify_sum)

    best_selection: Optional[BestCheckpointSelection] = None
    if args.select_best_checkpoint:
        best_selection = select_best_checkpoint(
            stats,
            metric=args.best_metric,
            snapshots_dir=Path(args.snapshots_dir),
            final_controller_path=Path(args.controller_path),
            final_lyapunov_path=Path(args.lyapunov_path),
        )
        (out_dir / "best_checkpoint.json").write_text(
            json.dumps(asdict(best_selection), indent=2),
            encoding="utf-8",
        )

    write_summary(summary, json_path, txt_path, train_log, eval_path, verify_path, best_selection)

    print("[Analyze] Done")
    print(f"- epoch csv: {csv_path}")
    print(f"- plots: {png_path}")
    print(f"- summary json: {json_path}")
    print(f"- summary txt: {txt_path}")
    if best_selection is not None:
        print(f"- best checkpoint: {out_dir / 'best_checkpoint.json'}")


if __name__ == "__main__":
    main()

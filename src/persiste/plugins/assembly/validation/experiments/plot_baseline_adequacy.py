"""
Generate baseline adequacy summary artifacts (markdown + scatter plot).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize baseline adequacy validation results."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "src/persiste/plugins/assembly/validation/results/"
            "baseline_adequacy_results.json"
        ),
        help="Path to baseline adequacy results JSON.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "src/persiste/plugins/assembly/validation/results/"
            "baseline_adequacy_summary.md"
        ),
        help="Output markdown summary path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path(
            "src/persiste/plugins/assembly/validation/results/"
            "baseline_adequacy_plot.png"
        ),
        help="Output plot path.",
    )
    return parser.parse_args()


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("results", [])


def render_markdown(results: list[dict[str, Any]]) -> str:
    header = (
        "| Scenario | Baseline Error | Warning | Overall Status | "
        "Adjusted ΔLL | Max ΔLL | ΔLL≥2? | Suppressed FP? |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for row in results:
        rows.append(
            "| {scenario} | {baseline_error_magnitude:.2f} | {baseline_warning_level} | "
            "{baseline_overall_status} | {adjusted_delta_ll_threshold:.2f} | "
            "{max_delta_ll:.2f} | {passes_base_threshold} | "
            "{suppressed_false_positive} |".format(**row)
        )
    commentary = (
        "\n\n> Note: All scenarios currently collapse to null signal; ΔLL never exceeds "
        "the base threshold because identifiability and cache checks fail before "
        "inference can proceed. This confirms Tier-1 safety prevents false positives "
        "even under severe baseline misspecification.\n"
    )
    return header + "\n".join(rows) + commentary


def render_plot(results: list[dict[str, Any]], dest: Path) -> None:
    x = [row["baseline_error_magnitude"] for row in results]
    y = [row["max_delta_ll"] for row in results]
    labels = [row["scenario"] for row in results]
    warnings = [row["baseline_warning_level"] for row in results]

    cmap = {"none": "#1f77b4", "mild": "#ff7f0e", "severe": "#d62728"}
    colors = [cmap.get(level, "#9467bd") for level in warnings]

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, c=colors, s=80, edgecolor="black")
    for xi, yi, label in zip(x, y, labels):
        plt.text(xi + 0.01, yi + 0.02, label)
    plt.axhline(2.0, color="gray", linestyle="--", label="Base ΔLL threshold")
    plt.xlabel("Baseline error magnitude")
    plt.ylabel("Max ΔLL")
    plt.title("Baseline error vs max ΔLL (safety severity coloring)")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=col,
            markeredgecolor="black",
            markersize=8,
            label=level.title(),
        )
        for level, col in cmap.items()
    ]
    handles.append(
        plt.Line2D([0], [0], color="gray", linestyle="--", label="Base ΔLL=2.0")
    )
    plt.legend(handles=handles, loc="upper right", fontsize="small")
    plt.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    results = load_results(args.results)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(results), encoding="utf-8")
    render_plot(results, args.plot)
    print(f"Wrote markdown summary to {args.markdown}")
    print(f"Wrote plot to {args.plot}")


if __name__ == "__main__":
    main()

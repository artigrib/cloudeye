"""Bar-chart render of a `scripts.audit.fit_prob` JSON result: one bar per platform,
height = `fits_count` (out of `trials`), sorted descending, annotated with each
platform's p5 corridor width - the "how many actually fit, and how tight was the
narrowest 5% of attempts" view side by side. Kept separate from `fit_prob.py` itself
so the CLI module that does the Monte-Carlo work has no matplotlib/plotting
dependency in its own import path - only this one-off render script needs it.

Usage:
    python -m scripts.audit.render_fit_prob_bar <fit_prob.json> <out.png>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def render(data: dict, out_path: Path) -> None:
    trials = data["trials"]
    platforms = sorted(data["platforms"].values(), key=lambda p: p["fits_count"], reverse=True)

    labels = [p["display_name"] for p in platforms]
    fits = [p["fits_count"] for p in platforms]
    colors = ["#2e7d32" if f >= trials * 0.5 else "#c62828" for f in fits]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, fits, color=colors)
    ax.set_ylim(0, trials + max(8, trials * 0.12))
    ax.set_ylabel(f"fits / {trials} trials")
    ax.set_title(
        f"Fit probability by platform (seed={data['seed']}, trials={trials}) - goal: {data.get('goal_target_id', '?')}"
    )
    ax.axhline(trials, color="#999999", linewidth=0.8, linestyle="--")

    for bar, platform in zip(bars, platforms):
        height = bar.get_height()
        p5 = platform["corridor_width_p5_m"]
        p5_label = f"{p5:.2f}m" if p5 is not None else "n/a"  # None when fits_count == 0 - no successful path to measure
        fallback = platform.get("fallback_trials", 0)
        fallback_label = f"\nfallback {fallback}/{trials}" if fallback else ""
        ax.annotate(
            f"{platform['fits_count']}/{trials}\np5={p5_label}{fallback_label}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("out_path", type=Path)
    args = parser.parse_args(argv)

    data = json.loads(args.json_path.read_text())
    render(data, args.out_path)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()

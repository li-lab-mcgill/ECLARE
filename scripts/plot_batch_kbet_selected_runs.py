"""
Compare batches_kbet_acceptance_rate for hand-picked MLflow nested eval runs.

Mirrors the CLIP / KD-CLIP / ECLARE comparison style in plot_figures.py (fig4),
but reads last-epoch values directly from mlruns/ instead of OUTPATH CSVs.

Usage (from ECLARE repo root):
  conda activate eclare_env
  python scripts/plot_batch_kbet_selected_runs.py
  python scripts/plot_batch_kbet_selected_runs.py --save /path/to/fig.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

ECLARE_ROOT = Path(__file__).resolve().parents[1]
MLRUNS = ECLARE_ROOT / "mlruns"
EXPERIMENT_ID = "715899953771574789"
METRIC = "batches_kbet_acceptance_rate"
METRIC_TITLE = "batches kBET acceptance rate"

# Selected nested eval runs in clip_mdd_04163347 (one representative per method / teacher).
SELECTED_RUNS = [
    {
        "run_id": "0f31f9b2fc634578916288bde5976f4d",
        "method": "eclare",
        "run_name": "MDD-0",
        "parent_run_name": "ECLARE_08085637",
        "source": np.nan,
        "target": "MDD",
    },
    {
        "run_id": "dc49223177c84dafad764e82a65aec92",
        "method": "kd-clip",
        "run_name": "PFC_V1_Wang-to-MDD-0",
        "parent_run_name": "KD_CLIP_07185908",
        "source": "PFC_V1_Wang",
        "target": "MDD",
    },
    {
        "run_id": "b85cfe62acd545e88876693b0bc642a6",
        "method": "kd-clip",
        "run_name": "PFC_Zhu-to-MDD-0",
        "parent_run_name": "KD_CLIP_07185908",
        "source": "PFC_Zhu",
        "target": "MDD",
    },
    {
        "run_id": "d5215281d0c74b9a8ee2b81ccce7ee37",
        "method": "clip",
        "run_name": "PFC_V1_Wang-to-MDD-0",
        "parent_run_name": "CLIP_04163347",
        "source": "PFC_V1_Wang",
        "target": "MDD",
    },
    {
        "run_id": "7a6789eab7a643efb091943dc2758c95",
        "method": "clip",
        "run_name": "PFC_Zhu-to-MDD-0",
        "parent_run_name": "CLIP_04163347",
        "source": "PFC_Zhu",
        "target": "MDD",
    },
]

METHOD_ORDER = ["eclare", "kd-clip", "clip"]
METHOD_LABELS = {"eclare": "ECLARE", "kd-clip": "KD-CLIP", "clip": "CLIP"}
METHOD_LABEL_ORDER = [METHOD_LABELS[m] for m in METHOD_ORDER]
METHOD_COLORS = {"eclare": "#1f77b4", "kd-clip": "#ff7f0e", "clip": "#2ca02c"}
SOURCE_MARKERS = {"PFC_V1_Wang": "s", "PFC_Zhu": "^"}


def read_last_metric(run_dir: Path, metric: str) -> tuple[float, int]:
    path = run_dir / "metrics" / metric
    if not path.exists():
        raise FileNotFoundError(f"Missing metric file: {path}")
    line = [ln for ln in path.read_text().splitlines() if ln.strip()][-1]
    parts = line.split()
    return float(parts[1]), int(parts[2])


def locate_run_dir(run_id: str) -> Path:
    direct = MLRUNS / EXPERIMENT_ID / run_id
    if (direct / "meta.yaml").exists():
        return direct
    matches = [p for p in MLRUNS.glob(f"*/{run_id}") if (p / "meta.yaml").exists()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Run not found under {MLRUNS}: {run_id}")
    raise RuntimeError(f"Ambiguous run locations for {run_id}: {matches}")


def load_selected_runs() -> pd.DataFrame:
    rows = []
    for spec in SELECTED_RUNS:
        run_dir = locate_run_dir(spec["run_id"])
        with open(run_dir / "meta.yaml") as f:
            meta = yaml.safe_load(f)
        value, step = read_last_metric(run_dir, METRIC)
        parent_tag = run_dir / "tags" / "mlflow.parentRunId"
        parent_id = parent_tag.read_text().splitlines()[-1].strip() if parent_tag.exists() else ""
        rows.append(
            {
                **spec,
                "experiment_id": meta.get("experiment_id", EXPERIMENT_ID),
                "status": meta.get("status"),
                "metric_step": step,
                METRIC: value,
                "run_path": f"clip_mdd_04163347/{spec['parent_run_name']}/{spec['run_name']}",
                "parent_run_id": parent_id,
            }
        )
    return pd.DataFrame(rows)


def metrics_df_for_method(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Shape compatible with plot_figures.combined_plot: MultiIndex (source, target)."""
    sub = df[df["method"] == method].copy()
    out = sub.set_index(["source", "target"])[[METRIC]]
    return out


def build_figure_title(df: pd.DataFrame) -> str:
    """MLflow nested run names above the metric title."""
    run_lines = []
    for method in METHOD_ORDER:
        sub = df[df["method"] == method].sort_values("run_name")
        for _, row in sub.iterrows():
            run_lines.append(f"{row['parent_run_name']}/{row['run_name']}")
    return "\n".join(run_lines + [METRIC_TITLE])


def plot_batch_kbet_mdd(df: pd.DataFrame, figsize=(8, 5)) -> plt.Figure:
    """Boxplot over methods + scatter by MDD source teacher (fig4-style)."""
    plot_df = df.copy()
    plot_df["method_label"] = plot_df["method"].map(METHOD_LABELS)
    plot_df["method_label"] = pd.Categorical(
        plot_df["method_label"], categories=METHOD_LABEL_ORDER, ordered=True
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.boxplot(
        x="method_label",
        y=METRIC,
        data=plot_df,
        order=METHOD_LABEL_ORDER,
        ax=ax,
        color="lightgray",
        showfliers=False,
        boxprops=dict(alpha=0.4),
        whiskerprops=dict(alpha=0.4),
        capprops=dict(alpha=0.4),
        medianprops=dict(alpha=0.7),
    )
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlabel("Method")
    ax.set_ylabel(None)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    position_mapping = {label: i for i, label in enumerate(METHOD_LABEL_ORDER)}
    for _, row in plot_df.iterrows():
        source = row["source"]
        marker = "o" if pd.isna(source) else SOURCE_MARKERS.get(source, "D")
        x = position_mapping[row["method_label"]] + np.random.uniform(-0.12, 0.12)
        ax.scatter(
            x,
            row[METRIC],
            color=METHOD_COLORS[row["method"]],
            marker=marker,
            edgecolor="black",
            s=90,
            zorder=10,
            alpha=0.9,
        )

    handles = [
        plt.Line2D([0], [0], marker="o", color="black", linestyle="None", label="all sources (ensemble)"),
        plt.Line2D([0], [0], marker=SOURCE_MARKERS["PFC_V1_Wang"], color="black", linestyle="None", label="PFC_V1_Wang"),
        plt.Line2D([0], [0], marker=SOURCE_MARKERS["PFC_Zhu"], color="black", linestyle="None", label="PFC_Zhu"),
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.set_title(build_figure_title(plot_df), fontsize=10)
    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=Path, default=None, help="Optional output path (.svg, .png, etc.)")
    parser.add_argument("--table", type=Path, default=None, help="Optional TSV summary path")
    args = parser.parse_args()

    df = load_selected_runs()
    print(df[
        ["run_id", "method", "parent_run_name", "run_name", "source", "target", "metric_step", METRIC, "run_path"]
    ].to_string(index=False))

    if args.table:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.table, sep="\t", index=False)
        print(f"\nWrote table: {args.table}")

    # Optional: reuse plot_figures.combined_plot exactly (heavy import).
    # target_dfs = [metrics_df_for_method(df, m) for m in METHOD_ORDER]
    # from plot_figures import combined_plot
    # fig = combined_plot(target_dfs, [METRIC], METHOD_ORDER, target_source_combinations=True, figsize=(8, 4))

    fig = plot_batch_kbet_mdd(df)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, bbox_inches="tight", dpi=300)
        print(f"Wrote figure: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

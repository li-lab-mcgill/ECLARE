#!/usr/bin/env python3
"""
Recreate the VIP eGRN network figure (DLX5 + POU2F1 hubs) directly from
science.adf0834_data_s3.xlsx, without needing the full SCENIC+ object.

The figure is a TF -> target-gene network for a single lineage:
    - TF nodes are large coloured labels (DLX5 green, POU2F1 red)
    - target genes are small grey dots with labels
    - edges are coloured by their source TF
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from adjustText import adjust_text


def build_network(excel_path, lineage="VIP", sheet_name="SCENIC_plus_results",
                  weight_col="TF2G_importance_x_abs_rho"):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    sub = df[df["lineage"] == lineage].dropna(subset=["TF", "Gene"]).copy()

    # one edge per TF->Gene pair, weight = max importance across regions
    edges = (
        sub.groupby(["TF", "Gene"])[weight_col]
        .max()
        .reset_index()
    )

    tfs = sorted(sub["TF"].unique().tolist())
    genes = sorted(sub["Gene"].unique().tolist())

    G = nx.Graph()
    G.add_nodes_from(tfs, kind="TF")
    G.add_nodes_from([g for g in genes if g not in tfs], kind="gene")
    for _, r in edges.iterrows():
        G.add_edge(r["TF"], r["Gene"], weight=float(r[weight_col]))
    return G, tfs


def plot_network(G, tfs, lineage, out_path, seed=2,
                 tf_colors=("#176117", "#a01618"), compactness=0.55):
    color_map = {tf: tf_colors[i % len(tf_colors)] for i, tf in enumerate(tfs)}

    # Force-directed layout: high-degree TF hubs repel, shared targets fall
    # in the middle, unique targets fan outward -- like the published figure.
    pos = nx.spring_layout(G, k=0.55, iterations=400, seed=seed,
                           weight=None)

    # Pull each target gene toward the centroid of the TF hub(s) it connects to
    # so outermost nodes sit closer to their hub and the figure is more compact.
    for g in G.nodes:
        if g in color_map:
            continue
        hubs = [n for n in G.neighbors(g) if n in color_map]
        if not hubs:
            continue
        anchor = np.mean([pos[h] for h in hubs], axis=0)
        # pull single-hub leaf nodes in harder than shared targets
        c = compactness + (0.15 if G.degree(g) == 1 else 0.0)
        pos[g] = anchor + (1.0 - c) * (pos[g] - anchor)

    # The layout is an intrinsically diagonal band (the DLX5<->POU2F1 axis
    # dominates the spread), which leaves the off-diagonal corners empty.
    # ZCA-whiten to decorrelate x/y and equalise variance -> roughly isotropic
    # cloud that fills all four corners, keeping orientation close to original.
    nodes = list(G.nodes)
    P = np.array([pos[n] for n in nodes])
    mean = P.mean(axis=0)
    cov = np.cov((P - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    W = evecs @ np.diag(1.0 / np.sqrt(evals + 1e-9)) @ evecs.T  # ZCA
    P = (P - mean) @ W

    # stretch to fill the box (2nd-98th pct -> [-1, 1])
    lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
    span = np.where((hi - lo) > 1e-9, hi - lo, 1.0)
    P = 2.0 * (P - lo) / span - 1.0

    # Rein in extreme outliers (e.g. the lone VIP node): pull anything beyond
    # the 88th-pct radius halfway back toward that radius.
    P -= P.mean(axis=0)
    r = np.linalg.norm(P, axis=1)
    rmax = np.percentile(r, 88)
    pull = r > rmax
    P[pull] *= ((rmax + 0.5 * (r[pull] - rmax)) / r[pull])[:, None]
    for n, p in zip(nodes, P):
        pos[n] = p

    # Keep a clearance zone around each hub so no gene dot sits under the big
    # TF label. The label is wide and short, so use an ellipse (wider in x)
    # and push any gene node inside it out to the ellipse boundary.
    cx, cy = 0.45, 0.20
    for g in [n for n in G.nodes if n not in color_map]:
        for tf in tfs:
            d = pos[g] - pos[tf]
            ed = np.hypot(d[0] / cx, d[1] / cy)
            if 1e-9 < ed < 1.0:
                pos[g] = pos[tf] + d / ed

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_axis_off()

    # edges coloured by source TF, alpha/width scaled by importance
    weights = np.array([d["weight"] for *_, d in G.edges(data=True)])
    wnorm = (weights - weights.min()) / (np.ptp(weights) + 1e-9)
    for (u, v, d), w in zip(G.edges(data=True), wnorm):
        tf = u if u in color_map else v
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=color_map[tf], alpha=0.25 + 0.55 * w,
                linewidth=0.6 + 1.4 * w, zorder=1)

    # gene nodes: small grey dots
    gene_nodes = [n for n in G.nodes if n not in color_map]
    nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes, node_size=30,
                           node_color="#666666", ax=ax)

    # TF nodes: large coloured labels (drawn first so gene labels avoid them)
    hub_texts = [ax.text(*pos[tf], tf, fontsize=22, fontweight="bold",
                         color=color_map[tf], ha="center", va="center",
                         zorder=4) for tf in tfs]

    # repel gene labels off each other, off the nodes, and off the hub labels,
    # with a thin connector drawn back to the dot the label belongs to.
    texts = [ax.text(*pos[g], g, fontsize=13, ha="center", va="center",
                     zorder=3) for g in gene_nodes]
    adjust_text(texts, ax=ax, objects=hub_texts,
                x=[pos[g][0] for g in gene_nodes],
                y=[pos[g][1] for g in gene_nodes],
                expand=(1.25, 1.5), force_text=(0.4, 0.6),
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5,
                                shrinkA=2, shrinkB=2))

    ax.set_title(lineage, fontsize=26, fontweight="bold",
                 color="#7030a0", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--excel", default=os.path.join(
        os.environ.get("DATAPATH", "."), "science.adf0834_data_s3.xlsx"))
    p.add_argument("--lineage", default="VIP")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--out", default=os.path.join(
        os.environ.get("OUTPATH", "."), "vip_egrn_network.png"))
    args = p.parse_args()

    G, tfs = build_network(args.excel, lineage=args.lineage)
    print(f"{args.lineage}: {len(tfs)} TFs ({tfs}), "
          f"{G.number_of_nodes() - len(tfs)} target genes, "
          f"{G.number_of_edges()} edges")
    plot_network(G, tfs, args.lineage, args.out, seed=args.seed)

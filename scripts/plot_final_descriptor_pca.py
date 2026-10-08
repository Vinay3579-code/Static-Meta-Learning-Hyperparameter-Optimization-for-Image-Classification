#!/usr/bin/env python3
"""
Plot the final task-descriptor PCA.

Recommended use:
    python scripts/plot_final_descriptor_pca.py \
        --descriptor-csv PATH_TO_FINAL_DESCRIPTOR_CSV

Expected metadata columns (case-insensitive):
    dataset, regime
and numeric descriptor columns for the 64 static features.

The script:
- excludes obvious identifiers/metadata columns,
- optionally uses the first 64 numeric descriptor columns,
- fits PCA on the meta-training rows only when a split column exists,
- projects all supplied rows,
- saves publication PDF + PNG.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def find_col(df, candidates):
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptor-csv", required=True)
    ap.add_argument("--out-dir", default="figures/final_v2")
    args = ap.parse_args()

    df = pd.read_csv(args.descriptor_csv)
    dataset_col = find_col(df, ["dataset", "dataset_name"])
    regime_col = find_col(df, ["regime", "episodic_regime"])
    split_col = find_col(df, ["split", "meta_split", "partition"])

    exclude = {
        c for c in df.columns
        if c.lower() in {
            "task_id", "task", "dataset", "dataset_name", "regime",
            "episodic_regime", "split", "meta_split", "partition",
            "descriptor_episodes", "descriptor_mean_std",
            "probe_steps", "probe_validation_episodes",
            "probe_elapsed_seconds"
        }
    }
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric if c not in exclude]

    # Prefer the first 64 descriptor dimensions for the Zero-SML view.
    if len(features) < 64:
        raise ValueError(f"Only {len(features)} numeric candidate features found; expected >= 64.")
    features = features[:64]

    X = df[features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    pca = PCA(n_components=2, random_state=42)

    if split_col is not None:
        mask = df[split_col].astype(str).str.lower().isin(["train", "meta-train", "metatrain"])
        if mask.sum() >= 20:
            scaler.fit(X.loc[mask])
            Z_train = scaler.transform(X.loc[mask])
            pca.fit(Z_train)
            Z = pca.transform(scaler.transform(X))
        else:
            Z = pca.fit_transform(scaler.fit_transform(X))
    else:
        Z = pca.fit_transform(scaler.fit_transform(X))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    if dataset_col is None:
        ax.scatter(Z[:, 0], Z[:, 1], s=16, alpha=0.65)
    else:
        for dataset_name, idx in df.groupby(dataset_col).groups.items():
            ax.scatter(Z[idx, 0], Z[idx, 1], s=18, alpha=0.72, label=str(dataset_name))
        ax.legend(ncol=2)

    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% variance)")
    ax.set_title("PCA of the final 64-dimensional static task descriptor")
    ax.grid(alpha=0.25)

    stem = out / "Fig1A_Final_TaskDescriptorPCA"
    plt.tight_layout()
    plt.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close()

    print("Saved:", stem.with_suffix(".pdf"))
    print("Explained variance ratio:", pca.explained_variance_ratio_)

if __name__ == "__main__":
    main()

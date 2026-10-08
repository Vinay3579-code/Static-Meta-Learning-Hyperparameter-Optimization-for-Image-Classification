#!/usr/bin/env python3
"""
Generate publication figures for the finalized SML study.

Run from the project root:
    python scripts/generate_final_paper_figures_v2.py

The script uses the frozen final numerical results reported in the
LODO-v3 manuscript. It deliberately does NOT recreate the obsolete
40-anchor / continuous-regression / learning-rate-sensitivity figures.

Optional descriptor PCA figure:
    python scripts/plot_final_descriptor_pca.py --descriptor-csv PATH
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("figures/final_v2")
OUT.mkdir(parents=True, exist_ok=True)

methods = [
    "Zero-SML", "Probe-SML", "Probe-NPP", "Random Search",
    "Bayesian Optimization", "Hyperband", "BOHB",
    "Validation-selected reference"
]
accuracy = np.array([62.964, 62.946, 62.359, 62.647, 62.161, 61.265, 61.195, 63.212])
sd5 = np.array([0.019, 0.045, 0.187, 0.068, 0.094, 0.189, 0.333, np.nan])

paired = pd.DataFrame({
    "Comparator": [
        "Probe-SML", "Probe ExtraTrees", "Random Search",
        "Bayesian Optimization", "Hyperband", "BOHB",
        "Validation-selected reference"
    ],
    "Difference": [0.018, 0.000, 0.317, 0.803, 1.699, 1.769, -0.247],
    "Low": [-0.004, -0.047, 0.159, 0.584, 1.376, 1.438, -0.351],
    "High": [0.043, 0.047, 0.479, 1.026, 2.048, 2.129, -0.155],
})

datasets = ["Omniglot", "CIFAR-100", "miniImageNet", "DTD", "Flowers102"]
dataset_values = {
    "Random Search": [99.947, 44.635, 41.314, 37.830, 89.509],
    "Bayesian Optimization": [99.597, 44.174, 40.718, 37.423, 88.892],
    "Hyperband": [99.344, 43.907, 39.970, 35.828, 87.277],
    "BOHB": [98.842, 43.757, 39.906, 35.862, 87.608],
    "Probe-NPP": [99.765, 44.495, 40.690, 37.244, 89.601],
    "Zero-SML": [99.991, 44.852, 41.450, 38.540, 89.989],
    "Validation-selected reference": [99.998, 44.932, 41.913, 38.977, 90.237],
}

regimes = ["5-way 1-shot", "5-way 5-shot", "10-way 1-shot", "10-way 5-shot"]
regime_values = {
    "Zero-SML": [59.243, 74.187, 51.303, 67.040],
    "Probe-SML": [59.383, 74.155, 51.331, 67.040],
    "Probe ExtraTrees": [59.292, 74.193, 51.332, 67.040],
    "Validation-selected reference": [59.984, 74.289, 51.540, 67.034],
}

lodo = pd.DataFrame({
    "Dataset": ["Omniglot", "CIFAR-100", "miniImageNet", "DTD", "Flowers102"],
    "Difference": [-0.052, 0.172, -0.603, -0.496, 0.299],
    "Low": [-0.104, 0.020, -1.154, -1.086, 0.017],
    "High": [-0.003, 0.333, -0.076, 0.086, 0.568],
})

cost = pd.DataFrame({
    "Method": ["Zero-SML", "Probe-SML", "Probe-NPP", "Random Search",
               "Bayesian Optimization", "Hyperband", "BOHB"],
    "Cost_FBE": [0, 0, 0, 40, 20, 16, 16],
    "Accuracy": [62.964, 62.946, 62.359, 62.647, 62.161, 61.265, 61.195],
})

agreement = pd.DataFrame({
    "Method": ["Zero-SML", "Probe-SML", "Probe ExtraTrees", "Validation-selected reference"],
    "Hard_agreement": [52.50, 55.00, 58.33, 100.00],
    "Equivalent_rate": [68.33, 71.67, 71.67, 100.00],
})

def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(OUT / f"{name}.png", dpi=400, bbox_inches="tight")
    plt.close()

# The plotting blocks below are intentionally kept explicit and simple so
# that every figure can be independently audited and edited.

# 1. Held-out accuracy
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(methods))
ax.errorbar(x[:-1], accuracy[:-1], yerr=sd5[:-1],
            fmt="o", capsize=4, linewidth=1.2, markersize=5,
            label="Mean +/- 5-run SD")
ax.scatter(x[-1], accuracy[-1], marker="D", s=42,
           label="Validation-selected reference")
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=28, ha="right")
ax.set_ylabel("Held-out test accuracy (%)")
ax.set_title("Final held-out performance across recommendation/search methods")
ax.grid(axis="y", alpha=0.25)
ax.legend()
savefig("Fig2_Final_HeldOutAccuracy")

# 2. Paired bootstrap forest plot
fig, ax = plt.subplots(figsize=(8.5, 5.3))
y = np.arange(len(paired))[::-1]
for yi, (_, row) in zip(y, paired.iterrows()):
    ax.plot([row["Low"], row["High"]], [yi, yi], linewidth=2)
    ax.plot(row["Difference"], yi, "o", markersize=6)
ax.axvline(0, linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(paired["Comparator"])
ax.set_xlabel("Zero-SML minus comparator accuracy difference (percentage points)")
ax.set_title("Paired 20,000-resample task-bootstrap comparisons")
ax.grid(axis="x", alpha=0.25)
savefig("Fig3_Final_PairedBootstrap")

# 3. Dataset-wise
fig, ax = plt.subplots(figsize=(10, 5.8))
plot_methods = ["Zero-SML", "Probe-NPP", "Random Search",
                "Bayesian Optimization", "Hyperband", "BOHB",
                "Validation-selected reference"]
offsets = np.linspace(-0.24, 0.24, len(plot_methods))
for off, method in zip(offsets, plot_methods):
    ax.plot(np.arange(len(datasets)) + off, dataset_values[method],
            marker="o", linewidth=1.2, markersize=4, label=method)
ax.set_xticks(np.arange(len(datasets)))
ax.set_xticklabels(datasets)
ax.set_ylabel("Mean test accuracy (%)")
ax.set_title("Dataset-wise final test accuracy")
ax.grid(axis="y", alpha=0.25)
ax.legend(ncol=2)
savefig("Fig4_Final_DatasetWiseAccuracy")

# 4. Regime-wise
fig, ax = plt.subplots(figsize=(9, 5.5))
for method, vals in regime_values.items():
    ax.plot(regimes, vals, marker="o", linewidth=1.2, markersize=4, label=method)
ax.set_ylabel("Mean test accuracy (%)")
ax.set_title("Performance across the four episodic regimes")
ax.grid(axis="y", alpha=0.25)
ax.legend(ncol=2)
savefig("Fig5_Final_RegimeWiseAccuracy")

# 5. Strict LODO: Zero-SML vs Random Search
fig, ax = plt.subplots(figsize=(8.8, 5.2))
y = np.arange(len(lodo))[::-1]
for yi, (_, row) in zip(y, lodo.iterrows()):
    ax.plot([row["Low"], row["High"]], [yi, yi], linewidth=2)
    ax.plot(row["Difference"], yi, "o", markersize=6)
ax.axvline(0, linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(lodo["Dataset"])
ax.set_xlabel("LODO Zero-SML minus Random Search (percentage points)")
ax.set_title("Strict leave-one-dataset-out transfer")
ax.grid(axis="x", alpha=0.25)
savefig("Fig6_Final_LODO_vs_RandomSearch")

# 6. Search cost vs accuracy
fig, ax = plt.subplots(figsize=(8.4, 5.4))
for _, row in cost.iterrows():
    ax.scatter(row["Cost_FBE"], row["Accuracy"], s=45)
    ax.annotate(row["Method"], (row["Cost_FBE"], row["Accuracy"]),
                xytext=(7, 3), textcoords="offset points", fontsize=9)
ax.set_xlabel("Target-task configuration-selection cost (FBE)")
ax.set_ylabel("Held-out test accuracy (%)")
ax.set_title("Accuracy versus target-task configuration-selection cost")
ax.grid(axis="both", alpha=0.25)
savefig("Fig7_Final_CostAccuracy")

# 7. Recommendation quality
fig, ax = plt.subplots(figsize=(8.3, 5.3))
x = np.arange(len(agreement))
width = 0.36
ax.bar(x - width/2, agreement["Hard_agreement"], width,
       label="Hard reference-label agreement")
ax.bar(x + width/2, agreement["Equivalent_rate"], width,
       label="Validation-equivalent recommendation rate")
ax.set_xticks(x)
ax.set_xticklabels(agreement["Method"], rotation=25, ha="right")
ax.set_ylim(0, 108)
ax.set_ylabel("Rate across held-out tasks (%)")
ax.set_title("Recommendation agreement and validation-equivalence")
ax.legend()
ax.grid(axis="y", alpha=0.25)
savefig("Fig8_Final_RecommendationQuality")

print(f"Saved figures to {OUT.resolve()}")

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(
    "results/final_analysis_v2"
)

OUT = ROOT / "paper_tables"
OUT.mkdir(
    parents=True,
    exist_ok=True,
)


DISPLAY_NAMES = {
    "zero_sml_ensemble":
        "Zero-SML (ours)",

    "probe_sml_ensemble":
        "Probe-SML (ours)",

    "global_majority":
        "Global default",

    "regime_majority":
        "Regime default",

    "dataset_majority":
        "Dataset default*",

    "dataset_regime_majority":
        "Dataset + regime default*",

    "probe_weighted_extra_trees":
        "Probe ExtraTrees",

    "validation_selected_oracle":
        "Validation-selected oracle",
}


ORDER = [
    "zero_sml_ensemble",
    "probe_sml_ensemble",
    "probe_weighted_extra_trees",
    "global_majority",
    "regime_majority",
    "dataset_majority",
    "dataset_regime_majority",
    "validation_selected_oracle",
]


def main():

    summary = pd.read_csv(
        ROOT / "method_summary.csv"
    )

    core = (
        summary[
            summary["method"].isin(
                ORDER
            )
        ]
        .copy()
    )

    order_map = {
        method: i
        for i, method
        in enumerate(ORDER)
    }

    core["_order"] = (
        core["method"]
        .map(order_map)
    )

    core = (
        core.sort_values(
            "_order"
        )
        .drop(
            columns="_order"
        )
    )

    core["Method"] = (
        core["method"]
        .map(DISPLAY_NAMES)
    )

    core["Test accuracy (%)"] = (
        core[
            "mean_test_accuracy_percent"
        ].map(
            lambda x:
                f"{x:.3f}"
        )
    )

    core["95% CI accuracy"] = [
        (
            f"[{lo:.3f}, "
            f"{hi:.3f}]"
        )
        for lo, hi
        in zip(
            core[
                "mean_test_accuracy_ci95_low"
            ],
            core[
                "mean_test_accuracy_ci95_high"
            ],
        )
    ]

    core["Gap to oracle (pp)"] = (
        core[
            "mean_signed_gap_pp"
        ].map(
            lambda x:
                f"{x:.3f}"
        )
    )

    core["Median gap (pp)"] = (
        core[
            "median_signed_gap_pp"
        ].map(
            lambda x:
                f"{x:.3f}"
        )
    )

    core["P95 gap (pp)"] = (
        core[
            "p95_signed_gap_pp"
        ].map(
            lambda x:
                f"{x:.3f}"
        )
    )

    core["Max gap (pp)"] = (
        core[
            "max_signed_gap_pp"
        ].map(
            lambda x:
                f"{x:.3f}"
        )
    )

    core["Hard agreement (%)"] = (
        core[
            "hard_oracle_agreement_percent"
        ].map(
            lambda x:
                f"{x:.2f}"
        )
    )

    core["Equivalent (%)"] = (
        core[
            "validation_oracle_equivalent_percent"
        ].map(
            lambda x:
                f"{x:.2f}"
        )
    )

    final = core[
        [
            "Method",
            "Test accuracy (%)",
            "95% CI accuracy",
            "Gap to oracle (pp)",
            "Median gap (pp)",
            "P95 gap (pp)",
            "Max gap (pp)",
            "Hard agreement (%)",
            "Equivalent (%)",
        ]
    ]

    final.to_csv(
        OUT / "table_final_test.csv",
        index=False,
    )

    # ----------------------------
    # Pairwise table
    # ----------------------------

    paired = pd.read_csv(
        ROOT
        / "paired_bootstrap_comparisons.csv"
    )

    zero = paired[
        paired["method_a"]
        == "zero_sml_ensemble"
    ].copy()

    zero["Comparator"] = (
        zero["method_b"]
        .map(DISPLAY_NAMES)
        .fillna(
            zero["method_b"]
        )
    )

    zero["Zero minus comparator (pp)"] = (
        zero[
            "mean_accuracy_difference_a_minus_b_pp"
        ].map(
            lambda x:
                f"{x:.3f}"
        )
    )

    zero["95% CI"] = [
        f"[{lo:.3f}, {hi:.3f}]"
        for lo, hi
        in zip(
            zero["ci95_low_pp"],
            zero["ci95_high_pp"],
        )
    ]

    zero["W/T/L"] = [
        f"{w}/{t}/{l}"
        for w, t, l
        in zip(
            zero["a_wins"],
            zero["ties"],
            zero["a_losses"],
        )
    ]

    zero[
        [
            "Comparator",
            "Zero minus comparator (pp)",
            "95% CI",
            "W/T/L",
        ]
    ].to_csv(
        OUT
        / "table_zero_pairwise.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("FINAL HELD-OUT TEST TABLE")
    print("=" * 100)

    print(
        final.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("ZERO-SML PAIRED COMPARISONS")
    print("=" * 100)

    print(
        zero[
            [
                "Comparator",
                "Zero minus comparator (pp)",
                "95% CI",
                "W/T/L",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "* Dataset-aware defaults "
        "use privileged dataset identity."
    )

    print()
    print(
        "PAPER TABLE GENERATION: PASS"
    )


if __name__ == "__main__":
    main()

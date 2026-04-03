import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ensure_output_dir() -> Path:
    """
    Save all outputs to the parent folder's Data directory.

    Example:
    current script: project/code/analyze_game_risk_dataset.py
    output folder : project/Data/
    """
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir.parent / "Data"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_analysis_result(df: pd.DataFrame, csv_path: str) -> Dict[str, Any]:
    feature_cols = [
        "enemy_reward",
        "death_penalty",
        "win_probability",
        "time_pressure",
        "recent_death",
        "player_type",
        "decision_score"
    ]

    basic_info = {
        "csv_path": str(csv_path),
        "num_samples": int(len(df)),
        "columns": list(df.columns)
    }

    decision_count = df["decision"].value_counts().sort_index()
    decision_ratio = df["decision"].value_counts(normalize=True).sort_index()

    decision_distribution = {
        str(int(k)): int(v)
        for k, v in decision_count.items()
    }

    decision_ratio_dict = {
        str(int(k)): round(float(v), 4)
        for k, v in decision_ratio.items()
    }

    archetype_distribution = {
        str(k): round(float(v), 4)
        for k, v in df["player_archetype"].value_counts(normalize=True).items()
    }

    feature_relation_df = (
        df.groupby("decision")[feature_cols]
        .mean()
        .round(4)
    )

    feature_mean_by_decision = {}
    for decision_label in feature_relation_df.index:
        feature_mean_by_decision[str(int(decision_label))] = {
            col: float(feature_relation_df.loc[decision_label, col])
            for col in feature_cols
        }

    feature_range_summary = {}
    for col in feature_cols:
        feature_range_summary[col] = {
            "min": float(round(df[col].min(), 4)),
            "max": float(round(df[col].max(), 4)),
            "mean": float(round(df[col].mean(), 4)),
            "std": float(round(df[col].std(), 4))
        }

    preview_rows = df.head(5).to_dict(orient="records")

    result = {
        "basic_info": basic_info,
        "decision_distribution": decision_distribution,
        "decision_ratio": decision_ratio_dict,
        "player_archetype_distribution": archetype_distribution,
        "feature_mean_by_decision": feature_mean_by_decision,
        "feature_range_summary": feature_range_summary,
        "preview_rows": preview_rows
    }

    return result


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_text_summary(df: pd.DataFrame, analysis_result: Dict[str, Any], output_path: Path) -> None:
    lines: List[str] = []

    lines.append("=" * 70)
    lines.append("Dataset Analysis Summary")
    lines.append("=" * 70)
    lines.append(f"Number of samples: {analysis_result['basic_info']['num_samples']}")
    lines.append(f"CSV path: {analysis_result['basic_info']['csv_path']}")
    lines.append("")

    lines.append("Columns:")
    for col in analysis_result["basic_info"]["columns"]:
        lines.append(f"- {col}")
    lines.append("")

    lines.append("Decision distribution:")
    for k, v in analysis_result["decision_distribution"].items():
        ratio = analysis_result["decision_ratio"][k]
        lines.append(f"  decision={k}: count={v}, ratio={ratio}")
    lines.append("")

    lines.append("Player archetype distribution:")
    for k, v in analysis_result["player_archetype_distribution"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("Feature range summary:")
    for col, stats in analysis_result["feature_range_summary"].items():
        lines.append(
            f"  {col}: min={stats['min']}, max={stats['max']}, "
            f"mean={stats['mean']}, std={stats['std']}"
        )
    lines.append("")

    lines.append("Feature mean by decision:")
    for decision_label, stats in analysis_result["feature_mean_by_decision"].items():
        lines.append(f"  decision={decision_label}")
        for col, value in stats.items():
            lines.append(f"    {col}: {value}")
    lines.append("")

    lines.append("First 5 rows preview:")
    lines.append(df.head(5).to_string(index=False))
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_preview_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.head(20).to_csv(output_path, index=False, encoding="utf-8-sig")


def plot_decision_distribution(df: pd.DataFrame, output_path: Path) -> None:
    counts = df["decision"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    plt.bar([str(i) for i in counts.index], counts.values)
    plt.title("Decision Distribution")
    plt.xlabel("Decision (0=Safe, 1=Risk)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_player_archetype_distribution(df: pd.DataFrame, output_path: Path) -> None:
    counts = df["player_archetype"].value_counts()

    plt.figure(figsize=(7, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Player Archetype Distribution")
    plt.xlabel("Player Archetype")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_feature_histograms(df: pd.DataFrame, output_path: Path) -> None:
    feature_cols = [
        "enemy_reward",
        "death_penalty",
        "win_probability",
        "time_pressure",
        "recent_death",
        "player_type",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, feature_cols):
        ax.hist(df[col], bins=20)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_boxplots_by_decision(df: pd.DataFrame, output_path: Path) -> None:
    feature_cols = [
        "enemy_reward",
        "death_penalty",
        "win_probability",
        "time_pressure",
        "player_type",
        "decision_score"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, col in zip(axes, feature_cols):
        data_0 = df[df["decision"] == 0][col]
        data_1 = df[df["decision"] == 1][col]

        ax.boxplot([data_0, data_1], labels=["0", "1"])
        ax.set_title(f"{col} by decision")
        ax.set_xlabel("Decision")
        ax.set_ylabel(col)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    cols = [
        "enemy_reward",
        "death_penalty",
        "win_probability",
        "time_pressure",
        "recent_death",
        "player_type",
        "decision_score",
        "decision"
    ]

    corr = df[cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j, i,
                f"{corr.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    ax.set_title("Correlation Heatmap")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_feature_mean_by_decision_csv(df: pd.DataFrame, output_path: Path) -> None:
    feature_cols = [
        "enemy_reward",
        "death_penalty",
        "win_probability",
        "time_pressure",
        "recent_death",
        "player_type",
        "decision_score"
    ]

    result = (
        df.groupby("decision")[feature_cols]
        .mean()
        .round(4)
        .reset_index()
    )
    result.to_csv(output_path, index=False, encoding="utf-8-sig")


def analyze_dataset(csv_path: str) -> None:
    output_dir = ensure_output_dir()
    df = load_dataset(csv_path)
    analysis_result = build_analysis_result(df, csv_path)

    # Save structured outputs
    save_json(analysis_result, output_dir / "analysis_result.json")
    save_text_summary(df, analysis_result, output_dir / "analysis_summary.txt")
    save_preview_csv(df, output_dir / "dataset_preview_top20.csv")
    save_feature_mean_by_decision_csv(df, output_dir / "feature_mean_by_decision.csv")

    # Save figures
    plot_decision_distribution(df, output_dir / "fig_decision_distribution.png")
    plot_player_archetype_distribution(df, output_dir / "fig_player_archetype_distribution.png")
    plot_feature_histograms(df, output_dir / "fig_feature_histograms.png")
    plot_boxplots_by_decision(df, output_dir / "fig_boxplots_by_decision.png")
    plot_correlation_heatmap(df, output_dir / "fig_correlation_heatmap.png")

    print("=" * 70)
    print("Dataset analysis completed successfully.")
    print("=" * 70)
    print(f"Input CSV: {csv_path}")
    print(f"Output directory: {output_dir}")
    print("")
    print("Generated files:")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            print(f"- {p.name}")


if __name__ == "__main__":
    
    csv_path = r"C:\Users\ys_huang\Desktop\ys_docu\AIHW\ASIM-1\Data\game_risk_decision_dataset.csv"
    analyze_dataset(csv_path)
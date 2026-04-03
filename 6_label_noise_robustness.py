import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


RANDOM_STATE = 414551032

# Colors
LR_COLOR = "#1F4E79"   # deep blue
RF_COLOR = "#2A9D8F"   # teal
SVM_COLOR = "#E9C46A"  # gold
GB_COLOR = "#8E6C8A"   # muted purple

# Paths
DATA_DIR = Path(r"C:\Users\ys_huang\Desktop\ys_docu\AIHW\ASIM-1\Data")
INPUT_CSV_PATH = DATA_DIR / "game_risk_decision_dataset.csv"
OUTPUT_DIR = DATA_DIR / "6"

NOISE_LEVELS = [0, 5, 10, 15]


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def get_feature_cols() -> List[str]:
    return [
        "enemy_reward",
        "death_penalty",
        "win_probability",
        "time_pressure",
        "recent_death",
        "player_type"
    ]


def get_model_color(model_name: str) -> str:
    color_map = {
        "logistic_regression": LR_COLOR,
        "random_forest": RF_COLOR,
        "svm_rbf": SVM_COLOR,
        "gradient_boosting": GB_COLOR
    }
    return color_map[model_name]


def get_display_name(model_name: str) -> str:
    display_map = {
        "logistic_regression": "LR",
        "random_forest": "RF",
        "svm_rbf": "SVM",
        "gradient_boosting": "GB"
    }
    return display_map[model_name]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_models() -> Dict[str, object]:
    logistic_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    random_forest_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ))
    ])

    svm_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    gradient_boosting_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        ))
    ])

    return {
        "logistic_regression": logistic_model,
        "random_forest": random_forest_model,
        "svm_rbf": svm_model,
        "gradient_boosting": gradient_boosting_model
    }


def create_noisy_dataset(
    df: pd.DataFrame,
    noise_percent: int,
    seed: int
) -> pd.DataFrame:
    """
    Flip the labels of a given percentage of samples.
    Features remain unchanged.
    """
    noisy_df = df.copy()
    noisy_df["original_decision"] = noisy_df["decision"]

    if noise_percent == 0:
        noisy_df["is_noisy_label"] = 0
        return noisy_df

    n_samples = len(noisy_df)
    n_flip = int(round(n_samples * noise_percent / 100.0))

    rng = np.random.default_rng(seed)
    flip_indices = rng.choice(n_samples, size=n_flip, replace=False)

    noisy_df["is_noisy_label"] = 0
    noisy_df.loc[flip_indices, "decision"] = 1 - noisy_df.loc[flip_indices, "decision"]
    noisy_df.loc[flip_indices, "is_noisy_label"] = 1

    return noisy_df


def save_noisy_datasets(base_df: pd.DataFrame) -> Dict[int, Path]:
    """
    Save noisy datasets to the same folder as the original dataset.
    """
    saved_paths = {}

    for noise_percent in NOISE_LEVELS:
        if noise_percent == 0:
            continue

        noisy_df = create_noisy_dataset(
            df=base_df,
            noise_percent=noise_percent,
            seed=RANDOM_STATE + noise_percent
        )

        output_path = DATA_DIR / f"game_risk_decision_dataset_noise_{noise_percent}.csv"
        noisy_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        saved_paths[noise_percent] = output_path

    return saved_paths


def prepare_X_y(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].copy()
    y = df["decision"].copy()
    return X, y


def compute_fold_metrics(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold
) -> List[Dict[str, float]]:
    fold_results: List[Dict[str, float]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        fold_results.append({
            "fold": fold_idx,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auroc": float(roc_auc_score(y_test, y_score))
        })

    return fold_results


def summarize_fold_metrics(fold_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    metric_names = ["accuracy", "precision", "recall", "f1", "auroc"]
    summary = {}

    for metric in metric_names:
        values = [row[metric] for row in fold_results]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }

    return summary


def get_cross_validated_scores(model, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold):
    y_pred_all = cross_val_predict(model, X, y, cv=cv, method="predict")

    if hasattr(model, "predict_proba"):
        y_score_all = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    else:
        y_score_all = cross_val_predict(model, X, y, cv=cv, method="decision_function")

    return y_pred_all, y_score_all


def evaluate_model(
    model_name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold
) -> Dict:
    fold_results = compute_fold_metrics(model, X, y, cv)
    fold_summary = summarize_fold_metrics(fold_results)

    y_pred_all, y_score_all = get_cross_validated_scores(model, X, y, cv)
    cm = confusion_matrix(y, y_pred_all)

    overall_metrics = {
        "accuracy": float(accuracy_score(y, y_pred_all)),
        "precision": float(precision_score(y, y_pred_all, zero_division=0)),
        "recall": float(recall_score(y, y_pred_all, zero_division=0)),
        "f1": float(f1_score(y, y_pred_all, zero_division=0)),
        "auroc": float(roc_auc_score(y, y_score_all))
    }

    return {
        "model_name": model_name,
        "fold_results": fold_results,
        "fold_summary": fold_summary,
        "overall_metrics": overall_metrics,
        "confusion_matrix": cm.tolist()
    }


def run_noise_experiments(
    dataset_map: Dict[int, pd.DataFrame]
) -> Dict:
    feature_cols = get_feature_cols()
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    all_results = {
        "input_csv": str(INPUT_CSV_PATH),
        "random_state": RANDOM_STATE,
        "cv_strategy": f"StratifiedKFold(n_splits=5, shuffle=True, random_state={RANDOM_STATE})",
        "feature_cols": feature_cols,
        "noise_experiments": {}
    }

    for noise_percent, df in dataset_map.items():
        X, y = prepare_X_y(df, feature_cols)

        exp_result = {
            "noise_percent": noise_percent,
            "num_samples": int(len(df)),
            "num_noisy_labels": int(df["is_noisy_label"].sum()) if "is_noisy_label" in df.columns else 0,
            "models": {}
        }

        for model_name, model in models.items():
            result = evaluate_model(
                model_name=model_name,
                model=model,
                X=X,
                y=y,
                cv=cv
            )
            exp_result["models"][model_name] = result

        all_results["noise_experiments"][str(noise_percent)] = exp_result

    return all_results


def build_summary_dataframe(results: Dict) -> pd.DataFrame:
    rows = []

    for noise_key, exp_result in results["noise_experiments"].items():
        noise_percent = int(noise_key)

        for model_name, model_result in exp_result["models"].items():
            rows.append({
                "noise_percent": noise_percent,
                "model_name": model_name,
                "num_samples": exp_result["num_samples"],
                "num_noisy_labels": exp_result["num_noisy_labels"],
                "accuracy_mean": model_result["fold_summary"]["accuracy"]["mean"],
                "accuracy_std": model_result["fold_summary"]["accuracy"]["std"],
                "precision_mean": model_result["fold_summary"]["precision"]["mean"],
                "precision_std": model_result["fold_summary"]["precision"]["std"],
                "recall_mean": model_result["fold_summary"]["recall"]["mean"],
                "recall_std": model_result["fold_summary"]["recall"]["std"],
                "f1_mean": model_result["fold_summary"]["f1"]["mean"],
                "f1_std": model_result["fold_summary"]["f1"]["std"],
                "auroc_mean": model_result["fold_summary"]["auroc"]["mean"],
                "auroc_std": model_result["fold_summary"]["auroc"]["std"],
                "overall_accuracy": model_result["overall_metrics"]["accuracy"],
                "overall_precision": model_result["overall_metrics"]["precision"],
                "overall_recall": model_result["overall_metrics"]["recall"],
                "overall_f1": model_result["overall_metrics"]["f1"],
                "overall_auroc": model_result["overall_metrics"]["auroc"]
            })

    return pd.DataFrame(rows)


def build_delta_dataframe(summary_df: pd.DataFrame) -> pd.DataFrame:
    baseline_df = summary_df[summary_df["noise_percent"] == 0].copy()
    baseline_lookup = {}

    for _, row in baseline_df.iterrows():
        baseline_lookup[row["model_name"]] = {
            "accuracy_mean": row["accuracy_mean"],
            "precision_mean": row["precision_mean"],
            "recall_mean": row["recall_mean"],
            "f1_mean": row["f1_mean"],
            "auroc_mean": row["auroc_mean"]
        }

    rows = []

    for _, row in summary_df.iterrows():
        base = baseline_lookup[row["model_name"]]
        rows.append({
            "noise_percent": row["noise_percent"],
            "model_name": row["model_name"],
            "delta_accuracy": base["accuracy_mean"] - row["accuracy_mean"],
            "delta_precision": base["precision_mean"] - row["precision_mean"],
            "delta_recall": base["recall_mean"] - row["recall_mean"],
            "delta_f1": base["f1_mean"] - row["f1_mean"],
            "delta_auroc": base["auroc_mean"] - row["auroc_mean"]
        })

    return pd.DataFrame(rows)


def save_confusion_matrices(results: Dict, output_dir: Path) -> None:
    for noise_key, exp_result in results["noise_experiments"].items():
        for model_name, model_result in exp_result["models"].items():
            cm = model_result["confusion_matrix"]
            df_cm = pd.DataFrame(
                cm,
                index=["true_0", "true_1"],
                columns=["pred_0", "pred_1"]
            )
            output_path = output_dir / f"noise_{noise_key}_{model_name}_confusion_matrix.csv"
            df_cm.to_csv(output_path, encoding="utf-8-sig")


def save_json(data: Dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_run_summary_txt(
    results: Dict,
    summary_df: pd.DataFrame,
    output_path: Path
) -> None:
    lines = []
    lines.append("=" * 90)
    lines.append("RQ10 Label Noise Robustness Summary")
    lines.append("=" * 90)
    lines.append(f"Input CSV: {results['input_csv']}")
    lines.append(f"Random State: {results['random_state']}")
    lines.append(f"CV Strategy: {results['cv_strategy']}")
    lines.append("Noise levels: 0%, 5%, 10%, 15%")
    lines.append("")

    for noise_percent in sorted(summary_df["noise_percent"].unique()):
        lines.append(f"[Noise {noise_percent}%]")
        sub_df = summary_df[summary_df["noise_percent"] == noise_percent]

        num_noisy = int(sub_df["num_noisy_labels"].iloc[0])
        lines.append(f"num_noisy_labels: {num_noisy}")

        for _, row in sub_df.iterrows():
            lines.append(
                f"  {row['model_name']}: "
                f"acc={row['accuracy_mean']:.4f}, "
                f"prec={row['precision_mean']:.4f}, "
                f"rec={row['recall_mean']:.4f}, "
                f"f1={row['f1_mean']:.4f}, "
                f"auc={row['auroc_mean']:.4f}"
            )
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_metric_vs_noise(
    summary_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    output_path: Path
) -> None:
    model_order = ["logistic_regression", "random_forest", "svm_rbf", "gradient_boosting"]

    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    for model_name in model_order:
        sub_df = summary_df[summary_df["model_name"] == model_name].copy()
        sub_df = sub_df.sort_values("noise_percent")

        ax.plot(
            sub_df["noise_percent"],
            sub_df[metric_col],
            marker="o",
            linewidth=2,
            markersize=6,
            color=get_model_color(model_name),
            label=get_display_name(model_name)
        )

        for _, row in sub_df.iterrows():
            ax.text(
                row["noise_percent"],
                row[metric_col] + 0.005,
                f"{row[metric_col]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    ax.set_xlabel("Label Noise (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(NOISE_LEVELS)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_delta_metric_vs_noise(
    delta_df: pd.DataFrame,
    delta_col: str,
    title: str,
    ylabel: str,
    output_path: Path
) -> None:
    model_order = ["logistic_regression", "random_forest", "svm_rbf", "gradient_boosting"]

    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    for model_name in model_order:
        sub_df = delta_df[delta_df["model_name"] == model_name].copy()
        sub_df = sub_df.sort_values("noise_percent")

        ax.plot(
            sub_df["noise_percent"],
            sub_df[delta_col],
            marker="o",
            linewidth=2,
            markersize=6,
            color=get_model_color(model_name),
            label=get_display_name(model_name)
        )

        for _, row in sub_df.iterrows():
            ax.text(
                row["noise_percent"],
                row[delta_col] + 0.002,
                f"{row[delta_col]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    ax.set_xlabel("Label Noise (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(NOISE_LEVELS)

    ymax = max(delta_df[delta_col].max() * 1.25, 0.03)
    ax.set_ylim(0, ymax)

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    output_dir = ensure_output_dir()

    base_df = load_dataset(INPUT_CSV_PATH)

    # Save noisy datasets beside the original dataset
    saved_noisy_paths = save_noisy_datasets(base_df)

    # Build dataset map for experiments
    dataset_map = {}

    base_df_for_exp = base_df.copy()
    base_df_for_exp["original_decision"] = base_df_for_exp["decision"]
    base_df_for_exp["is_noisy_label"] = 0
    dataset_map[0] = base_df_for_exp

    for noise_percent, path in saved_noisy_paths.items():
        dataset_map[noise_percent] = load_dataset(path)

    # Run experiments
    results = run_noise_experiments(dataset_map)

    summary_df = build_summary_dataframe(results)
    delta_df = build_delta_dataframe(summary_df)

    summary_df.to_csv(output_dir / "6_label_noise_summary.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(output_dir / "6_label_noise_delta_summary.csv", index=False, encoding="utf-8-sig")
    save_confusion_matrices(results, output_dir)

    # Save JSON
    json_ready = {
        "input_csv": str(INPUT_CSV_PATH),
        "random_state": RANDOM_STATE,
        "noise_levels": NOISE_LEVELS,
        "saved_noisy_paths": {str(k): str(v) for k, v in saved_noisy_paths.items()},
        "results": results
    }
    save_json(json_ready, output_dir / "6_label_noise_results.json")

    save_run_summary_txt(
        results=results,
        summary_df=summary_df,
        output_path=output_dir / "6_run_summary.txt"
    )

    # Plots
    plot_metric_vs_noise(
        summary_df=summary_df,
        metric_col="f1_mean",
        title="Label Noise Robustness: F1 vs Noise Level",
        ylabel="F1 Score",
        output_path=output_dir / "fig_noise_f1_vs_noise.png"
    )

    plot_metric_vs_noise(
        summary_df=summary_df,
        metric_col="auroc_mean",
        title="Label Noise Robustness: AUROC vs Noise Level",
        ylabel="AUROC",
        output_path=output_dir / "fig_noise_auroc_vs_noise.png"
    )

    plot_delta_metric_vs_noise(
        delta_df=delta_df,
        delta_col="delta_f1",
        title="Label Noise Robustness: ΔF1 from Baseline",
        ylabel="ΔF1",
        output_path=output_dir / "fig_noise_delta_f1.png"
    )

    plot_delta_metric_vs_noise(
        delta_df=delta_df,
        delta_col="delta_auroc",
        title="Label Noise Robustness: ΔAUROC from Baseline",
        ylabel="ΔAUROC",
        output_path=output_dir / "fig_noise_delta_auroc.png"
    )

    print("=" * 90)
    print("6_label_noise_robustness.py finished successfully")
    print("=" * 90)
    print(f"Input CSV: {INPUT_CSV_PATH}")
    print(f"Output directory: {output_dir}")
    print("")
    print("Saved noisy datasets:")
    for noise_percent, path in saved_noisy_paths.items():
        print(f"- {noise_percent}% -> {path}")

    print("")
    print("Generated files in Data\\6:")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            print(f"- {p.name}")

    print("")
    print("Quick summary (F1 / AUROC):")
    for noise_percent in sorted(summary_df["noise_percent"].unique()):
        print(f"[Noise {noise_percent}%]")
        sub_df = summary_df[summary_df["noise_percent"] == noise_percent]
        for _, row in sub_df.iterrows():
            print(
                f"  {row['model_name']}: "
                f"F1={row['f1_mean']:.4f}, "
                f"AUROC={row['auroc_mean']:.4f}"
            )
        print("")


if __name__ == "__main__":
    main()
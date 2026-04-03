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

INPUT_CSV_PATH = r"C:\Users\ys_huang\Desktop\ys_docu\AIHW\ASIM-1\Data\game_risk_decision_dataset.csv"
OUTPUT_DIR = Path(r"C:\Users\ys_huang\Desktop\ys_docu\AIHW\ASIM-1\Data\4")


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_all_feature_cols() -> List[str]:
    return [
        "enemy_reward",
        "death_penalty",
        "win_probability",
        "time_pressure",
        "recent_death",
        "player_type"
    ]


def get_ablation_settings() -> List[Dict]:
    return [
        {
            "experiment_name": "baseline",
            "removed_feature": None
        },
        {
            "experiment_name": "remove_win_probability",
            "removed_feature": "win_probability"
        },
        {
            "experiment_name": "remove_player_type",
            "removed_feature": "player_type"
        },
        {
            "experiment_name": "remove_recent_death",
            "removed_feature": "recent_death"
        }
    ]


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


def get_model_color(model_name: str) -> str:
    color_map = {
        "logistic_regression": LR_COLOR,
        "random_forest": RF_COLOR,
        "svm_rbf": SVM_COLOR,
        "gradient_boosting": GB_COLOR
    }
    return color_map[model_name]


def prepare_X_y(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
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


def run_ablation_experiments(df: pd.DataFrame) -> Dict:
    all_feature_cols = get_all_feature_cols()
    ablation_settings = get_ablation_settings()
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    all_results = {
        "input_csv": INPUT_CSV_PATH,
        "num_samples": int(len(df)),
        "label_col": "decision",
        "random_state": RANDOM_STATE,
        "cv_strategy": f"StratifiedKFold(n_splits=5, shuffle=True, random_state={RANDOM_STATE})",
        "experiments": {}
    }

    for setting in ablation_settings:
        experiment_name = setting["experiment_name"]
        removed_feature = setting["removed_feature"]

        if removed_feature is None:
            feature_cols = all_feature_cols.copy()
        else:
            feature_cols = [col for col in all_feature_cols if col != removed_feature]

        X, y = prepare_X_y(df, feature_cols)

        experiment_result = {
            "removed_feature": removed_feature,
            "used_features": feature_cols,
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
            experiment_result["models"][model_name] = result

        all_results["experiments"][experiment_name] = experiment_result

    return all_results


def save_json(data: Dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def build_summary_dataframe(results: Dict) -> pd.DataFrame:
    rows = []

    for experiment_name, experiment_data in results["experiments"].items():
        removed_feature = experiment_data["removed_feature"]

        for model_name, model_result in experiment_data["models"].items():
            row = {
                "experiment_name": experiment_name,
                "removed_feature": removed_feature if removed_feature is not None else "None",
                "model_name": model_name,
                "used_features": ", ".join(experiment_data["used_features"]),
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
            }
            rows.append(row)

    return pd.DataFrame(rows)


def build_delta_dataframe(summary_df: pd.DataFrame) -> pd.DataFrame:
    baseline_df = summary_df[summary_df["experiment_name"] == "baseline"].copy()
    baseline_lookup = {}

    for _, row in baseline_df.iterrows():
        baseline_lookup[row["model_name"]] = {
            "f1_mean": row["f1_mean"],
            "auroc_mean": row["auroc_mean"],
            "accuracy_mean": row["accuracy_mean"],
            "precision_mean": row["precision_mean"],
            "recall_mean": row["recall_mean"]
        }

    rows = []

    for _, row in summary_df.iterrows():
        model_name = row["model_name"]
        base = baseline_lookup[model_name]

        rows.append({
            "experiment_name": row["experiment_name"],
            "removed_feature": row["removed_feature"],
            "model_name": model_name,
            "delta_accuracy": base["accuracy_mean"] - row["accuracy_mean"],
            "delta_precision": base["precision_mean"] - row["precision_mean"],
            "delta_recall": base["recall_mean"] - row["recall_mean"],
            "delta_f1": base["f1_mean"] - row["f1_mean"],
            "delta_auroc": base["auroc_mean"] - row["auroc_mean"]
        })

    return pd.DataFrame(rows)


def save_confusion_matrices(results: Dict, output_dir: Path) -> None:
    for experiment_name, experiment_data in results["experiments"].items():
        for model_name, model_result in experiment_data["models"].items():
            cm = model_result["confusion_matrix"]
            df_cm = pd.DataFrame(
                cm,
                index=["true_0", "true_1"],
                columns=["pred_0", "pred_1"]
            )
            out_path = output_dir / f"{experiment_name}_{model_name}_confusion_matrix.csv"
            df_cm.to_csv(out_path, encoding="utf-8-sig")


def save_run_summary_txt(results: Dict, summary_df: pd.DataFrame, output_path: Path) -> None:
    lines = []
    lines.append("=" * 90)
    lines.append("Ablation Study Summary")
    lines.append("=" * 90)
    lines.append(f"Input CSV: {results['input_csv']}")
    lines.append(f"Random State: {results['random_state']}")
    lines.append(f"CV Strategy: {results['cv_strategy']}")
    lines.append("")

    for experiment_name, experiment_data in results["experiments"].items():
        lines.append(f"[{experiment_name}]")
        lines.append(f"Removed feature: {experiment_data['removed_feature']}")
        lines.append(f"Used features: {', '.join(experiment_data['used_features'])}")

        sub_df = summary_df[summary_df["experiment_name"] == experiment_name]
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


def plot_metric_by_experiment(
    summary_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    output_path: Path
) -> None:
    experiment_order = [
        "baseline",
        "remove_win_probability",
        "remove_player_type",
        "remove_recent_death"
    ]
    model_order = [
        "logistic_regression",
        "random_forest",
        "svm_rbf",
        "gradient_boosting"
    ]
    display_name = {
        "logistic_regression": "LR",
        "random_forest": "RF",
        "svm_rbf": "SVM",
        "gradient_boosting": "GB"
    }

    x = np.arange(len(experiment_order))
    width = 0.12

    fig, ax = plt.subplots(figsize=(11, 6))
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for model_name, offset in zip(model_order, offsets):
        model_df = summary_df[summary_df["model_name"] == model_name].copy()
        model_df = model_df.set_index("experiment_name").loc[experiment_order].reset_index()

        values = model_df[metric_col].tolist()

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=get_model_color(model_name),
            edgecolor="black",
            linewidth=0.6,
            label=display_name[model_name]
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    xtick_labels = [
        "Baseline",
        "Remove\nwin_probability",
        "Remove\nplayer_type",
        "Remove\nrecent_death"
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 1.08)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_delta_metric(
    delta_df: pd.DataFrame,
    delta_col: str,
    title: str,
    ylabel: str,
    output_path: Path
) -> None:
    experiment_order = [
        "remove_win_probability",
        "remove_player_type",
        "remove_recent_death"
    ]
    model_order = [
        "logistic_regression",
        "random_forest",
        "svm_rbf",
        "gradient_boosting"
    ]
    display_name = {
        "logistic_regression": "LR",
        "random_forest": "RF",
        "svm_rbf": "SVM",
        "gradient_boosting": "GB"
    }

    x = np.arange(len(experiment_order))
    width = 0.12

    fig, ax = plt.subplots(figsize=(11, 6))
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for model_name, offset in zip(model_order, offsets):
        model_df = delta_df[delta_df["model_name"] == model_name].copy()
        model_df = model_df[model_df["experiment_name"].isin(experiment_order)]
        model_df = model_df.set_index("experiment_name").loc[experiment_order].reset_index()

        values = model_df[delta_col].tolist()

        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=get_model_color(model_name),
            edgecolor="black",
            linewidth=0.6,
            label=display_name[model_name]
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.003,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    xtick_labels = [
        "Remove\nwin_probability",
        "Remove\nplayer_type",
        "Remove\nrecent_death"
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)

    ymax = max(delta_df[delta_col].max() * 1.25, 0.05)
    ax.set_ylim(0, ymax)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    output_dir = ensure_output_dir()
    df = load_dataset(INPUT_CSV_PATH)

    results = run_ablation_experiments(df)

    summary_df = build_summary_dataframe(results)
    delta_df = build_delta_dataframe(summary_df)

    save_json(results, output_dir / "4_ablation_study_results.json")
    summary_df.to_csv(output_dir / "4_ablation_summary.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(output_dir / "4_ablation_delta_summary.csv", index=False, encoding="utf-8-sig")

    save_confusion_matrices(results, output_dir)

    save_run_summary_txt(
        results=results,
        summary_df=summary_df,
        output_path=output_dir / "4_run_summary.txt"
    )

    plot_metric_by_experiment(
        summary_df=summary_df,
        metric_col="f1_mean",
        title="Ablation Study: F1 Comparison",
        ylabel="F1 Score",
        output_path=output_dir / "fig_ablation_f1_comparison.png"
    )

    plot_metric_by_experiment(
        summary_df=summary_df,
        metric_col="auroc_mean",
        title="Ablation Study: AUROC Comparison",
        ylabel="AUROC",
        output_path=output_dir / "fig_ablation_auroc_comparison.png"
    )

    plot_delta_metric(
        delta_df=delta_df,
        delta_col="delta_f1",
        title="Ablation Study: F1 Drop from Baseline",
        ylabel="ΔF1",
        output_path=output_dir / "fig_ablation_delta_f1.png"
    )

    plot_delta_metric(
        delta_df=delta_df,
        delta_col="delta_auroc",
        title="Ablation Study: AUROC Drop from Baseline",
        ylabel="ΔAUROC",
        output_path=output_dir / "fig_ablation_delta_auroc.png"
    )

    print("=" * 90)
    print("4_ablation_study.py finished successfully")
    print("=" * 90)
    print(f"Input CSV: {INPUT_CSV_PATH}")
    print(f"Output directory: {output_dir}")
    print(f"Random State: {RANDOM_STATE}")
    print("")
    print("Generated files:")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            print(f"- {p.name}")

    print("")
    print("Quick summary (F1 mean / AUROC mean):")
    for experiment_name in summary_df["experiment_name"].unique():
        print(f"[{experiment_name}]")
        sub_df = summary_df[summary_df["experiment_name"] == experiment_name]
        for _, row in sub_df.iterrows():
            print(
                f"  {row['model_name']}: "
                f"F1={row['f1_mean']:.4f}, "
                f"AUROC={row['auroc_mean']:.4f}"
            )
        print("")


if __name__ == "__main__":
    main()
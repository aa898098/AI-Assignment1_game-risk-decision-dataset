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
    confusion_matrix,
    roc_curve
)


RANDOM_STATE = 414551032

# Colors
LR_COLOR = "#1F4E79"          # deep blue
RF_COLOR = "#2A9D8F"          # teal
SVM_COLOR = "#E9C46A"         # gold
GB_COLOR = "#8E6C8A"          # muted purple

# Output
INPUT_CSV_PATH = r"C:\Users\ys_huang\Desktop\ys_docu\AIHW\ASIM-1\Data\game_risk_decision_dataset.csv"
OUTPUT_DIR = Path(r"C:\Users\ys_huang\Desktop\ys_docu\AIHW\ASIM-1\Data\3")


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def prepare_features_and_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feature_cols = [
        "enemy_reward",
        "death_penalty",
        "win_probability",
        "time_pressure",
        "recent_death",
        "player_type"
    ]
    label_col = "decision"

    X = df[feature_cols].copy()
    y = df[label_col].copy()

    return X, y, feature_cols


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


def extract_model_explainability(
    model_name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str]
) -> Dict[str, float] | None:
    model.fit(X, y)
    clf = model.named_steps["classifier"]

    if model_name == "logistic_regression":
        coef = clf.coef_[0]
        return {
            feature: float(value)
            for feature, value in zip(feature_cols, coef)
        }

    if model_name == "random_forest":
        importances = clf.feature_importances_
        return {
            feature: float(value)
            for feature, value in zip(feature_cols, importances)
        }

    if model_name == "gradient_boosting":
        importances = clf.feature_importances_
        return {
            feature: float(value)
            for feature, value in zip(feature_cols, importances)
        }

    return None


def evaluate_model(
    model_name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    feature_cols: List[str]
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

    fpr, tpr, thresholds = roc_curve(y, y_score_all)

    feature_importance = extract_model_explainability(
        model_name=model_name,
        model=model,
        X=X,
        y=y,
        feature_cols=feature_cols
    )

    return {
        "model_name": model_name,
        "fold_results": fold_results,
        "fold_summary": fold_summary,
        "overall_metrics": overall_metrics,
        "confusion_matrix": cm.tolist(),
        "feature_importance": feature_importance,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist()
        }
    }


def save_json(data: Dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_metrics_csv(all_results: Dict[str, Dict], output_path: Path) -> None:
    rows = []

    for model_name, result in all_results.items():
        rows.append({
            "model_name": model_name,
            "accuracy_mean": result["fold_summary"]["accuracy"]["mean"],
            "accuracy_std": result["fold_summary"]["accuracy"]["std"],
            "precision_mean": result["fold_summary"]["precision"]["mean"],
            "precision_std": result["fold_summary"]["precision"]["std"],
            "recall_mean": result["fold_summary"]["recall"]["mean"],
            "recall_std": result["fold_summary"]["recall"]["std"],
            "f1_mean": result["fold_summary"]["f1"]["mean"],
            "f1_std": result["fold_summary"]["f1"]["std"],
            "auroc_mean": result["fold_summary"]["auroc"]["mean"],
            "auroc_std": result["fold_summary"]["auroc"]["std"],
            "overall_accuracy": result["overall_metrics"]["accuracy"],
            "overall_precision": result["overall_metrics"]["precision"],
            "overall_recall": result["overall_metrics"]["recall"],
            "overall_f1": result["overall_metrics"]["f1"],
            "overall_auroc": result["overall_metrics"]["auroc"]
        })

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def save_fold_results_csv(all_results: Dict[str, Dict], output_path: Path) -> None:
    rows = []

    for model_name, result in all_results.items():
        for fold_result in result["fold_results"]:
            row = {"model_name": model_name}
            row.update(fold_result)
            rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def save_confusion_matrix_csv(cm: List[List[int]], output_path: Path) -> None:
    df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    df.to_csv(output_path, encoding="utf-8-sig")


def save_feature_importance_csv(
    feature_importance: Dict[str, float],
    output_path: Path,
    value_col_name: str
) -> None:
    df = pd.DataFrame({
        "feature": list(feature_importance.keys()),
        value_col_name: list(feature_importance.values())
    })
    df = df.sort_values(by=value_col_name, ascending=False, key=lambda s: s.abs())
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_run_summary_txt(
    csv_path: str,
    feature_cols: List[str],
    all_results: Dict[str, Dict],
    output_path: Path
) -> None:
    lines = []
    lines.append("=" * 80)
    lines.append("Baseline Model Training Summary")
    lines.append("=" * 80)
    lines.append(f"Input CSV: {csv_path}")
    lines.append(f"Random State: {RANDOM_STATE}")
    lines.append("")
    lines.append("Used features:")
    for col in feature_cols:
        lines.append(f"- {col}")
    lines.append("")
    lines.append("Models:")
    lines.append("- Logistic Regression")
    lines.append("- Random Forest")
    lines.append("- SVM (RBF)")
    lines.append("- Gradient Boosting")
    lines.append("")
    lines.append("Cross-validation:")
    lines.append("- Stratified 5-Fold")
    lines.append("")

    for model_name, result in all_results.items():
        lines.append(f"[{model_name}]")
        for metric, stats in result["fold_summary"].items():
            lines.append(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        lines.append("  overall_metrics:")
        for metric, value in result["overall_metrics"].items():
            lines.append(f"    {metric}: {value:.4f}")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_confusion_matrix(cm: List[List[int]], title: str, output_path: Path, cmap: str = "Blues") -> None:
    cm_array = np.array(cm)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_array, cmap=cmap)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred_0", "pred_1"])
    ax.set_yticklabels(["true_0", "true_1"])
    ax.set_title(title, fontsize=13)

    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(j, i, str(cm_array[i, j]), ha="center", va="center")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_model_comparison(all_results: Dict[str, Dict], output_path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_display_names = ["Accuracy", "Precision", "Recall", "F1"]
    model_order = ["logistic_regression", "random_forest", "svm_rbf", "gradient_boosting"]
    model_display = {
        "logistic_regression": "LR",
        "random_forest": "RF",
        "svm_rbf": "SVM",
        "gradient_boosting": "GB"
    }

    values_by_model = {
        model_name: [all_results[model_name]["fold_summary"][m]["mean"] for m in metrics]
        for model_name in model_order
    }

    x = np.arange(len(metrics))
    width = 0.12

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for model_name, offset in zip(model_order, offsets):
        values = values_by_model[model_name]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=get_model_color(model_name),
            edgecolor="black",
            linewidth=0.6,
            label=model_display[model_name]
        )
        for i, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{values[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_display_names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Baseline Model Comparison (5-Fold Mean)", fontsize=13)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_auroc_comparison(all_results: Dict[str, Dict], output_path: Path) -> None:
    model_order = ["logistic_regression", "random_forest", "svm_rbf", "gradient_boosting"]
    display_names = ["Logistic Regression", "Random Forest", "SVM (RBF)", "Gradient Boosting"]
    values = [all_results[m]["fold_summary"]["auroc"]["mean"] for m in model_order]
    colors = [get_model_color(m) for m in model_order]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(display_names, values, width=0.55, color=colors, edgecolor="black", linewidth=0.6)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC Comparison", fontsize=13)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_roc_curves(all_results: Dict[str, Dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    model_order = ["logistic_regression", "random_forest", "svm_rbf", "gradient_boosting"]
    display_names = {
        "logistic_regression": "LR",
        "random_forest": "RF",
        "svm_rbf": "SVM",
        "gradient_boosting": "GB"
    }

    for model_name in model_order:
        fpr = all_results[model_name]["roc_curve"]["fpr"]
        tpr = all_results[model_name]["roc_curve"]["tpr"]
        auc_value = all_results[model_name]["overall_metrics"]["auroc"]

        ax.plot(
            fpr,
            tpr,
            label=f"{display_names[model_name]} (AUC={auc_value:.3f})",
            color=get_model_color(model_name),
            linewidth=2
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_feature_bar(
    feature_importance: Dict[str, float],
    output_path: Path,
    title: str,
    ylabel: str,
    color: str
) -> None:
    items = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    features = [x[0] for x in items]
    values = [x[1] for x in items]

    plt.figure(figsize=(7, 4.5))
    plt.bar(features, values, width=0.55, color=color, edgecolor="black", linewidth=0.6)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    output_dir = ensure_output_dir()

    df = load_dataset(INPUT_CSV_PATH)
    X, y, feature_cols = prepare_features_and_label(df)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = build_models()

    all_results: Dict[str, Dict] = {}

    for model_name, model in models.items():
        all_results[model_name] = evaluate_model(
            model_name=model_name,
            model=model,
            X=X,
            y=y,
            cv=cv,
            feature_cols=feature_cols
        )

    summary_json = {
        "input_csv": INPUT_CSV_PATH,
        "num_samples": int(len(df)),
        "feature_cols": feature_cols,
        "label_col": "decision",
        "random_state": RANDOM_STATE,
        "cv_strategy": f"StratifiedKFold(n_splits=5, shuffle=True, random_state={RANDOM_STATE})",
        "results": all_results
    }

    save_json(summary_json, output_dir / "baseline_metrics_summary.json")
    save_metrics_csv(all_results, output_dir / "baseline_metrics_summary.csv")
    save_fold_results_csv(all_results, output_dir / "baseline_fold_results.csv")

    for model_name, result in all_results.items():
        save_confusion_matrix_csv(
            result["confusion_matrix"],
            output_dir / f"{model_name}_confusion_matrix.csv"
        )

    if all_results["logistic_regression"]["feature_importance"] is not None:
        save_feature_importance_csv(
            all_results["logistic_regression"]["feature_importance"],
            output_dir / "logistic_regression_coefficients.csv",
            value_col_name="coefficient"
        )

    if all_results["random_forest"]["feature_importance"] is not None:
        save_feature_importance_csv(
            all_results["random_forest"]["feature_importance"],
            output_dir / "random_forest_feature_importance.csv",
            value_col_name="importance"
        )

    if all_results["gradient_boosting"]["feature_importance"] is not None:
        save_feature_importance_csv(
            all_results["gradient_boosting"]["feature_importance"],
            output_dir / "gradient_boosting_feature_importance.csv",
            value_col_name="importance"
        )

    save_run_summary_txt(
        csv_path=INPUT_CSV_PATH,
        feature_cols=feature_cols,
        all_results=all_results,
        output_path=output_dir / "3_run_summary.txt"
    )

    cmap_map = {
        "logistic_regression": "Blues",
        "random_forest": "BuGn",
        "svm_rbf": "YlOrBr",
        "gradient_boosting": "Purples"
    }

    title_map = {
        "logistic_regression": "Logistic Regression Confusion Matrix",
        "random_forest": "Random Forest Confusion Matrix",
        "svm_rbf": "SVM (RBF) Confusion Matrix",
        "gradient_boosting": "Gradient Boosting Confusion Matrix"
    }

    for model_name, result in all_results.items():
        plot_confusion_matrix(
            result["confusion_matrix"],
            title=title_map[model_name],
            output_path=output_dir / f"fig_{model_name}_confusion_matrix.png",
            cmap=cmap_map[model_name]
        )

    plot_model_comparison(
        all_results=all_results,
        output_path=output_dir / "fig_baseline_model_comparison.png"
    )

    plot_auroc_comparison(
        all_results=all_results,
        output_path=output_dir / "fig_auroc_comparison.png"
    )

    plot_roc_curves(
        all_results=all_results,
        output_path=output_dir / "fig_roc_curves.png"
    )

    if all_results["logistic_regression"]["feature_importance"] is not None:
        plot_feature_bar(
            all_results["logistic_regression"]["feature_importance"],
            output_path=output_dir / "fig_logistic_regression_coefficients.png",
            title="Logistic Regression Coefficients",
            ylabel="Coefficient",
            color=LR_COLOR
        )

    if all_results["random_forest"]["feature_importance"] is not None:
        plot_feature_bar(
            all_results["random_forest"]["feature_importance"],
            output_path=output_dir / "fig_random_forest_feature_importance.png",
            title="Random Forest Feature Importance",
            ylabel="Importance",
            color=RF_COLOR
        )

    if all_results["gradient_boosting"]["feature_importance"] is not None:
        plot_feature_bar(
            all_results["gradient_boosting"]["feature_importance"],
            output_path=output_dir / "fig_gradient_boosting_feature_importance.png",
            title="Gradient Boosting Feature Importance",
            ylabel="Importance",
            color=GB_COLOR
        )

    print("=" * 80)
    print("3_train_baseline_models.py finished successfully")
    print("=" * 80)
    print(f"Input CSV: {INPUT_CSV_PATH}")
    print(f"Output directory: {output_dir}")
    print(f"Random State: {RANDOM_STATE}")
    print("")
    print("Generated files:")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            print(f"- {p.name}")

    print("")
    print("Quick summary:")
    for model_name, result in all_results.items():
        print(f"[{model_name}]")
        for metric, stats in result["fold_summary"].items():
            print(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        print("")


if __name__ == "__main__":
    main()
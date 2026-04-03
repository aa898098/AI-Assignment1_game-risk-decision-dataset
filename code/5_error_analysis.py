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
from sklearn.metrics import confusion_matrix


RANDOM_STATE = 414551032

# Colors
LR_COLOR = "#1F4E79"   # deep blue
RF_COLOR = "#2A9D8F"   # teal
SVM_COLOR = "#E9C46A"  # gold
GB_COLOR = "#8E6C8A"   # muted purple

INPUT_CSV_PATH = r"C:\Users\ys_huang\Desktop\ys_docu\AIHW\ASIM-1\Data\game_risk_decision_dataset.csv"
OUTPUT_DIR = Path(r"C:\Users\ys_huang\Desktop\ys_docu\AIHW\ASIM-1\Data\5")


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_feature_cols() -> List[str]:
    return [
        "enemy_reward",
        "death_penalty",
        "win_probability",
        "time_pressure",
        "recent_death",
        "player_type"
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


def get_display_name(model_name: str) -> str:
    display_map = {
        "logistic_regression": "LR",
        "random_forest": "RF",
        "svm_rbf": "SVM",
        "gradient_boosting": "GB"
    }
    return display_map[model_name]


def prepare_X_y(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].copy()
    y = df["decision"].copy()
    return X, y


def get_cv_predictions(model, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold):
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")

    if hasattr(model, "predict_proba"):
        y_score = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    else:
        y_score = cross_val_predict(model, X, y, cv=cv, method="decision_function")

    return y_pred, y_score


def assign_prediction_group(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


def build_prediction_dataframe(
    df: pd.DataFrame,
    model_name: str,
    y_pred: np.ndarray,
    y_score: np.ndarray
) -> pd.DataFrame:
    out_df = df.copy()
    out_df["model_name"] = model_name
    out_df["y_true"] = out_df["decision"]
    out_df["y_pred"] = y_pred
    out_df["y_score"] = y_score
    out_df["is_correct"] = (out_df["y_true"] == out_df["y_pred"]).astype(int)
    out_df["prediction_group"] = [
        assign_prediction_group(int(t), int(p))
        for t, p in zip(out_df["y_true"], out_df["y_pred"])
    ]
    out_df["error_type"] = np.where(out_df["is_correct"] == 1, "correct", "error")
    out_df["abs_decision_score"] = out_df["decision_score"].abs()
    return out_df


def summarize_correct_vs_error(pred_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    summary_cols = feature_cols + ["decision_score", "abs_decision_score", "y_score"]

    grouped = (
        pred_df.groupby("error_type")[summary_cols]
        .mean()
        .round(4)
        .reset_index()
    )
    return grouped


def summarize_prediction_groups(pred_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    summary_cols = feature_cols + ["decision_score", "abs_decision_score", "y_score"]

    grouped = (
        pred_df.groupby("prediction_group")[summary_cols]
        .mean()
        .round(4)
        .reset_index()
    )

    group_order = ["TP", "TN", "FP", "FN"]
    grouped["prediction_group"] = pd.Categorical(
        grouped["prediction_group"],
        categories=group_order,
        ordered=True
    )
    grouped = grouped.sort_values("prediction_group").reset_index(drop=True)
    return grouped


def summarize_error_counts(pred_df: pd.DataFrame) -> pd.DataFrame:
    counts = pred_df["prediction_group"].value_counts().reindex(["TP", "TN", "FP", "FN"]).fillna(0).astype(int)
    ratios = (counts / len(pred_df)).round(4)

    out_df = pd.DataFrame({
        "prediction_group": counts.index,
        "count": counts.values,
        "ratio": ratios.values
    })
    return out_df


def summarize_boundary_analysis(pred_df: pd.DataFrame, boundary_threshold: float = 10.0) -> pd.DataFrame:
    pred_df = pred_df.copy()
    pred_df["boundary_region"] = np.where(
        pred_df["abs_decision_score"] <= boundary_threshold,
        "near_boundary",
        "far_from_boundary"
    )

    result = (
        pred_df.groupby("boundary_region")
        .agg(
            sample_count=("sample_id", "count"),
            error_rate=("is_correct", lambda x: 1.0 - float(np.mean(x))),
            mean_abs_decision_score=("abs_decision_score", "mean"),
            mean_win_probability=("win_probability", "mean"),
            mean_player_type=("player_type", "mean"),
            mean_time_pressure=("time_pressure", "mean")
        )
        .reset_index()
        .round(4)
    )
    return result


def summarize_error_only(pred_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    error_df = pred_df[pred_df["is_correct"] == 0].copy()
    if len(error_df) == 0:
        return pd.DataFrame()

    summary_cols = feature_cols + ["decision_score", "abs_decision_score", "y_score"]
    result = error_df[summary_cols].describe().round(4)
    return result


def analyze_one_model(
    df: pd.DataFrame,
    model_name: str,
    model,
    feature_cols: List[str],
    cv: StratifiedKFold
) -> Dict:
    X, y = prepare_X_y(df, feature_cols)
    y_pred, y_score = get_cv_predictions(model, X, y, cv)

    pred_df = build_prediction_dataframe(
        df=df,
        model_name=model_name,
        y_pred=y_pred,
        y_score=y_score
    )

    cm = confusion_matrix(pred_df["y_true"], pred_df["y_pred"])

    result = {
        "model_name": model_name,
        "num_samples": int(len(pred_df)),
        "num_errors": int((pred_df["is_correct"] == 0).sum()),
        "error_rate": float(round((pred_df["is_correct"] == 0).mean(), 4)),
        "confusion_matrix": cm.tolist(),
        "error_counts": summarize_error_counts(pred_df),
        "correct_vs_error_summary": summarize_correct_vs_error(pred_df, feature_cols),
        "prediction_group_summary": summarize_prediction_groups(pred_df, feature_cols),
        "boundary_analysis": summarize_boundary_analysis(pred_df, boundary_threshold=10.0),
        "error_only_describe": summarize_error_only(pred_df, feature_cols),
        "prediction_dataframe": pred_df
    }
    return result


def save_json(data: Dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_analysis_tables(results: Dict[str, Dict], output_dir: Path) -> None:
    for model_name, result in results.items():
        result["error_counts"].to_csv(
            output_dir / f"{model_name}_error_counts.csv",
            index=False,
            encoding="utf-8-sig"
        )

        result["correct_vs_error_summary"].to_csv(
            output_dir / f"{model_name}_correct_vs_error_summary.csv",
            index=False,
            encoding="utf-8-sig"
        )

        result["prediction_group_summary"].to_csv(
            output_dir / f"{model_name}_prediction_group_summary.csv",
            index=False,
            encoding="utf-8-sig"
        )

        result["boundary_analysis"].to_csv(
            output_dir / f"{model_name}_boundary_analysis.csv",
            index=False,
            encoding="utf-8-sig"
        )

        if not result["error_only_describe"].empty:
            result["error_only_describe"].to_csv(
                output_dir / f"{model_name}_error_only_describe.csv",
                encoding="utf-8-sig"
            )

        result["prediction_dataframe"].to_csv(
            output_dir / f"{model_name}_prediction_details.csv",
            index=False,
            encoding="utf-8-sig"
        )


def build_overall_summary(results: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for model_name, result in results.items():
        row = {
            "model_name": model_name,
            "num_samples": result["num_samples"],
            "num_errors": result["num_errors"],
            "error_rate": result["error_rate"]
        }

        error_counts_df = result["error_counts"]
        for _, r in error_counts_df.iterrows():
            row[f"{r['prediction_group']}_count"] = int(r["count"])
            row[f"{r['prediction_group']}_ratio"] = float(r["ratio"])

        boundary_df = result["boundary_analysis"]
        for _, r in boundary_df.iterrows():
            label = r["boundary_region"]
            row[f"{label}_sample_count"] = int(r["sample_count"])
            row[f"{label}_error_rate"] = float(r["error_rate"])

        rows.append(row)

    return pd.DataFrame(rows)


def convert_results_for_json(results: Dict[str, Dict]) -> Dict:
    json_ready = {}
    for model_name, result in results.items():
        json_ready[model_name] = {
            "model_name": result["model_name"],
            "num_samples": result["num_samples"],
            "num_errors": result["num_errors"],
            "error_rate": result["error_rate"],
            "confusion_matrix": result["confusion_matrix"],
            "error_counts": result["error_counts"].to_dict(orient="records"),
            "correct_vs_error_summary": result["correct_vs_error_summary"].to_dict(orient="records"),
            "prediction_group_summary": result["prediction_group_summary"].to_dict(orient="records"),
            "boundary_analysis": result["boundary_analysis"].to_dict(orient="records")
        }
    return json_ready


def plot_error_rate_by_model(results: Dict[str, Dict], output_path: Path) -> None:
    model_order = ["logistic_regression", "random_forest", "svm_rbf", "gradient_boosting"]
    display_names = ["Logistic Regression", "Random Forest", "SVM (RBF)", "Gradient Boosting"]
    values = [results[m]["error_rate"] for m in model_order]
    colors = [get_model_color(m) for m in model_order]

    fig, ax = plt.subplots(figsize=(8.8, 5))
    bars = ax.bar(display_names, values, width=0.55, color=colors, edgecolor="black", linewidth=0.6)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.set_ylabel("Error Rate")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title("Overall Error Rate by Model", fontsize=13)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_boundary_error_rate(results: Dict[str, Dict], output_path: Path) -> None:
    model_order = ["logistic_regression", "random_forest", "svm_rbf", "gradient_boosting"]
    x = np.arange(len(model_order))
    width = 0.18

    near_vals = []
    far_vals = []

    for model_name in model_order:
        df = results[model_name]["boundary_analysis"].copy().set_index("boundary_region")
        near_vals.append(float(df.loc["near_boundary", "error_rate"]))
        far_vals.append(float(df.loc["far_from_boundary", "error_rate"]))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    near_bars = ax.bar(
        x - width / 2,
        near_vals,
        width=width,
        color="#E76F51",
        edgecolor="black",
        linewidth=0.6,
        label="Near boundary"
    )
    far_bars = ax.bar(
        x + width / 2,
        far_vals,
        width=width,
        color="#457B9D",
        edgecolor="black",
        linewidth=0.6,
        label="Far from boundary"
    )

    for bars, vals in [(near_bars, near_vals), (far_bars, far_vals)]:
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.005,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["LR", "RF", "SVM", "GB"])
    ax.set_ylabel("Error Rate")
    ax.set_title("Error Rate: Near vs Far from Decision Boundary", fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_prediction_group_distribution(results: Dict[str, Dict], output_path: Path) -> None:
    model_order = ["logistic_regression", "random_forest", "svm_rbf", "gradient_boosting"]
    group_order = ["TP", "TN", "FP", "FN"]
    group_colors = {
        "TP": "#2A9D8F",
        "TN": "#457B9D",
        "FP": "#E9C46A",
        "FN": "#E76F51"
    }

    x = np.arange(len(model_order))
    bottom = np.zeros(len(model_order))

    fig, ax = plt.subplots(figsize=(10, 5.8))

    for group in group_order:
        values = []
        for model_name in model_order:
            df = results[model_name]["error_counts"].copy().set_index("prediction_group")
            values.append(float(df.loc[group, "ratio"]))

        ax.bar(
            x,
            values,
            bottom=bottom,
            color=group_colors[group],
            edgecolor="black",
            linewidth=0.5,
            label=group
        )
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(["LR", "RF", "SVM", "GB"])
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 1.0)
    ax.set_title("Prediction Group Distribution by Model", fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_correct_vs_error_feature(
    results: Dict[str, Dict],
    feature_name: str,
    output_path: Path
) -> None:
    model_order = ["logistic_regression", "random_forest", "svm_rbf", "gradient_boosting"]
    x = np.arange(len(model_order))
    width = 0.18

    correct_vals = []
    error_vals = []

    for model_name in model_order:
        df = results[model_name]["correct_vs_error_summary"].copy().set_index("error_type")
        correct_vals.append(float(df.loc["correct", feature_name]))
        error_vals.append(float(df.loc["error", feature_name]))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    bars1 = ax.bar(
        x - width / 2,
        correct_vals,
        width=width,
        color="#2A9D8F",
        edgecolor="black",
        linewidth=0.6,
        label="Correct"
    )

    bars2 = ax.bar(
        x + width / 2,
        error_vals,
        width=width,
        color="#E76F51",
        edgecolor="black",
        linewidth=0.6,
        label="Error"
    )

    for bars, vals in [(bars1, correct_vals), (bars2, error_vals)]:
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.01 if feature_name != "decision_score" else 1.0),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["LR", "RF", "SVM", "GB"])
    ax.set_ylabel(feature_name)
    ax.set_title(f"{feature_name}: Correct vs Error", fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_run_summary_txt(results: Dict[str, Dict], output_path: Path) -> None:
    lines = []
    lines.append("=" * 90)
    lines.append("RQ7 Error Analysis Summary")
    lines.append("=" * 90)
    lines.append(f"Input CSV: {INPUT_CSV_PATH}")
    lines.append(f"Random State: {RANDOM_STATE}")
    lines.append("Boundary threshold: abs(decision_score) <= 10")
    lines.append("")

    for model_name, result in results.items():
        lines.append(f"[{model_name}]")
        lines.append(f"num_samples: {result['num_samples']}")
        lines.append(f"num_errors: {result['num_errors']}")
        lines.append(f"error_rate: {result['error_rate']:.4f}")
        lines.append("error counts:")
        for _, row in result["error_counts"].iterrows():
            lines.append(
                f"  {row['prediction_group']}: "
                f"count={int(row['count'])}, ratio={float(row['ratio']):.4f}"
            )
        lines.append("boundary analysis:")
        for _, row in result["boundary_analysis"].iterrows():
            lines.append(
                f"  {row['boundary_region']}: "
                f"sample_count={int(row['sample_count'])}, "
                f"error_rate={float(row['error_rate']):.4f}"
            )
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    output_dir = ensure_output_dir()
    df = load_dataset(INPUT_CSV_PATH)
    feature_cols = get_feature_cols()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = build_models()

    results: Dict[str, Dict] = {}

    for model_name, model in models.items():
        results[model_name] = analyze_one_model(
            df=df,
            model_name=model_name,
            model=model,
            feature_cols=feature_cols,
            cv=cv
        )

    overall_summary_df = build_overall_summary(results)
    overall_summary_df.to_csv(
        output_dir / "5_error_analysis_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    save_analysis_tables(results, output_dir)

    json_ready = {
        "input_csv": INPUT_CSV_PATH,
        "random_state": RANDOM_STATE,
        "feature_cols": feature_cols,
        "models": convert_results_for_json(results)
    }
    save_json(json_ready, output_dir / "5_error_analysis_results.json")

    save_run_summary_txt(results, output_dir / "5_run_summary.txt")

    plot_error_rate_by_model(
        results=results,
        output_path=output_dir / "fig_error_rate_by_model.png"
    )

    plot_boundary_error_rate(
        results=results,
        output_path=output_dir / "fig_boundary_error_rate.png"
    )

    plot_prediction_group_distribution(
        results=results,
        output_path=output_dir / "fig_prediction_group_distribution.png"
    )

    plot_correct_vs_error_feature(
        results=results,
        feature_name="win_probability",
        output_path=output_dir / "fig_correct_vs_error_win_probability.png"
    )

    plot_correct_vs_error_feature(
        results=results,
        feature_name="abs_decision_score",
        output_path=output_dir / "fig_correct_vs_error_abs_decision_score.png"
    )

    print("=" * 90)
    print("5_error_analysis.py finished successfully")
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
    print("Quick summary:")
    for model_name, result in results.items():
        print(
            f"[{model_name}] "
            f"errors={result['num_errors']}, "
            f"error_rate={result['error_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
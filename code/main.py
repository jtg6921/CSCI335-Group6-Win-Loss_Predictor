"""
NFL home win prediction — leakage-safe expanded feature set.

Task: binary classification — predict whether the home team wins (label: Home_Win).

Features use only information available before kickoff:
  - Schedule: week, rest days per team, prior win rate and games played.
  - Team rolling stats: season-to-date and last-5 means of points for/against,
    yards for/against, turnovers for/against, and win rate. Rolling windows
    are computed per team on a chronological timeline, then shifted by 1 so
    the current game is never part of its own feature values.
  - Matchup differences: home minus away for the last-5 rolling features.

The wide Home_* / Away_* stat columns in ML_Ready_NFL_2024.csv are full-season
aggregates merged onto every game row and are intentionally NOT used.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "ML_Ready_NFL_2024.csv"
LOGISTIC_RESULTS_MD = PROJECT_ROOT / "LogisticRegression_results.md"
FIGS_DIR = PROJECT_ROOT / "figs"
LR_METRICS_FIG = FIGS_DIR / "lr_metrics_bar.png"
LR_CONFUSION_FIG = FIGS_DIR / "lr_confusion_matrix_test.png"

# Playoff rows use string labels; map to synthetic week numbers after regular season (18)
_PLAYOFF_WEEK = {
    "WildCard": 19.0,
    "Division": 20.0,
    "ConfChamp": 21.0,
    "SuperBowl": 22.0,
}

# Time-based split by NFL week
TRAIN_MAX_WEEK = 12
VAL_MAX_WEEK = 14

# Columns we compute rolling aggregates for (per team, chronologically).
ROLL_BASE_COLUMNS = [
    "pts_for",
    "pts_against",
    "yds_for",
    "yds_against",
    "to_for",
    "to_against",
    "won",
]

# Last-5 rolling columns we also build matchup-difference features for.
DIFF_BASE_COLUMNS = [
    "pts_for",
    "pts_against",
    "yds_for",
    "yds_against",
    "to_for",
    "won",
]

NEUTRAL_WIN_PCT = 0.5
DEFAULT_REST_DAYS = 7.0


def _normalize_week(raw: object) -> float:
    s = str(raw).strip()
    if s in _PLAYOFF_WEEK:
        return _PLAYOFF_WEEK[s]
    try:
        return float(int(s))
    except ValueError:
        return float("nan")


def load_games(path: Path = DATA_PATH) -> pd.DataFrame:
    usecols = [
        "Week", "Date", "Time", "Home", "Away", "Home_Win",
        "Home_Score", "Away_Score", "YdsW", "YdsL", "TOW", "TOL",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df["week"] = df["Week"].map(_normalize_week)
    df = df.dropna(subset=["week"]).copy()

    raw_dt = df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip()
    df["kickoff"] = pd.to_datetime(raw_dt, format="%Y-%m-%d %I:%M%p", errors="coerce")
    if df["kickoff"].isna().any():
        df["kickoff"] = pd.to_datetime(raw_dt, errors="coerce")
    if df["kickoff"].isna().any():
        bad = df.loc[df["kickoff"].isna(), ["Date", "Time"]]
        raise ValueError(f"Could not parse some kickoff times:\n{bad}")

    df = df.sort_values("kickoff", kind="mergesort").reset_index(drop=True)

    # Winner/Loser-coded yards and turnovers -> home/away orientation.
    home_won = df["Home_Win"].to_numpy() == 1
    df["home_yds"] = np.where(home_won, df["YdsW"], df["YdsL"])
    df["away_yds"] = np.where(home_won, df["YdsL"], df["YdsW"])
    df["home_to"] = np.where(home_won, df["TOW"], df["TOL"])
    df["away_to"] = np.where(home_won, df["TOL"], df["TOW"])
    return df


def build_team_timeline(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game with pre-kickoff rolling stats (shifted to exclude current)."""
    home = pd.DataFrame({
        "team": games["Home"].values,
        "kickoff": games["kickoff"].values,
        "pts_for": games["Home_Score"].values,
        "pts_against": games["Away_Score"].values,
        "yds_for": games["home_yds"].values,
        "yds_against": games["away_yds"].values,
        "to_for": games["home_to"].values,
        "to_against": games["away_to"].values,
        "won": games["Home_Win"].values,
    })
    away = pd.DataFrame({
        "team": games["Away"].values,
        "kickoff": games["kickoff"].values,
        "pts_for": games["Away_Score"].values,
        "pts_against": games["Home_Score"].values,
        "yds_for": games["away_yds"].values,
        "yds_against": games["home_yds"].values,
        "to_for": games["away_to"].values,
        "to_against": games["home_to"].values,
        "won": 1 - games["Home_Win"].values,
    })
    t = pd.concat([home, away], ignore_index=True)
    t = t.sort_values(["team", "kickoff"], kind="mergesort").reset_index(drop=True)

    g = t.groupby("team", sort=False)
    for col in ROLL_BASE_COLUMNS:
        shifted = g[col].shift()
        t[f"{col}_std"] = shifted.groupby(t["team"]).transform(
            lambda s: s.expanding().mean()
        )
        t[f"{col}_r5"] = shifted.groupby(t["team"]).transform(
            lambda s: s.rolling(5, min_periods=1).mean()
        )
    return t


def build_feature_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Merge schedule features + rolling team stats onto each game row."""
    timeline = build_team_timeline(games)
    roll_feat_cols = [f"{c}_std" for c in ROLL_BASE_COLUMNS] + [
        f"{c}_r5" for c in ROLL_BASE_COLUMNS
    ]
    keep = ["team", "kickoff"] + roll_feat_cols

    home_feats = timeline[keep].rename(
        columns={"team": "Home", **{c: f"home_{c}" for c in roll_feat_cols}}
    )
    away_feats = timeline[keep].rename(
        columns={"team": "Away", **{c: f"away_{c}" for c in roll_feat_cols}}
    )

    df = games.merge(home_feats, on=["Home", "kickoff"], how="left")
    df = df.merge(away_feats, on=["Away", "kickoff"], how="left")

    # Schedule features: prior win pct, games played, rest days (chronological pass).
    wins: dict[str, list[int]] = {}
    last_played: dict[str, pd.Timestamp] = {}
    h_pcts, a_pcts, h_games, a_games, h_rest, a_rest = [], [], [], [], [], []

    for _, row in df.iterrows():
        home, away, t = row["Home"], row["Away"], row["kickoff"]
        hw, aw = wins.get(home, []), wins.get(away, [])
        h_pcts.append(float(np.mean(hw)) if hw else NEUTRAL_WIN_PCT)
        a_pcts.append(float(np.mean(aw)) if aw else NEUTRAL_WIN_PCT)
        h_games.append(float(len(hw)))
        a_games.append(float(len(aw)))
        h_rest.append(
            (t - last_played[home]).total_seconds() / 86400.0
            if home in last_played
            else DEFAULT_REST_DAYS
        )
        a_rest.append(
            (t - last_played[away]).total_seconds() / 86400.0
            if away in last_played
            else DEFAULT_REST_DAYS
        )
        home_won = bool(row["Home_Win"])
        wins.setdefault(home, []).append(1 if home_won else 0)
        wins.setdefault(away, []).append(0 if home_won else 1)
        last_played[home] = t
        last_played[away] = t

    df["home_win_pct_prior"] = h_pcts
    df["away_win_pct_prior"] = a_pcts
    df["home_games_prior"] = h_games
    df["away_games_prior"] = a_games
    df["home_rest_days"] = h_rest
    df["away_rest_days"] = a_rest

    # Matchup difference features (home minus away) on last-5 rolling stats.
    for base in DIFF_BASE_COLUMNS:
        df[f"diff_{base}_r5"] = df[f"home_{base}_r5"] - df[f"away_{base}_r5"]

    return df


def feature_columns() -> list[str]:
    cols = [
        "week",
        "home_rest_days", "away_rest_days",
        "home_win_pct_prior", "away_win_pct_prior",
        "home_games_prior", "away_games_prior",
    ]
    for base in ROLL_BASE_COLUMNS:
        cols += [f"home_{base}_std", f"away_{base}_std",
                 f"home_{base}_r5", f"away_{base}_r5"]
    for base in DIFF_BASE_COLUMNS:
        cols.append(f"diff_{base}_r5")
    return cols


def week_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = df["week"].to_numpy()
    train = w <= TRAIN_MAX_WEEK
    val = (w > TRAIN_MAX_WEEK) & (w <= VAL_MAX_WEEK)
    test = w > VAL_MAX_WEEK
    return train, val, test


def compute_split_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    m: dict[str, float] = {
        "log_loss": float(log_loss(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }
    try:
        m["roc_auc"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        m["roc_auc"] = float("nan")
    return m


def evaluate_split(name: str, y_true: np.ndarray, proba: np.ndarray) -> None:
    met = compute_split_metrics(y_true, proba)
    print(
        f"  {name}: log_loss={met['log_loss']:.4f}  "
        f"brier={met['brier']:.4f}  roc_auc={met['roc_auc']:.4f}"
    )


def plot_lr_metrics_bar(
    metrics: dict[str, dict[str, float]],
    path: Path,
) -> None:
    """Grouped bar chart: train/val/test bars per metric (Log loss, Brier, ROC-AUC)."""
    metric_keys = ["log_loss", "brier", "roc_auc"]
    metric_labels = ["Log loss", "Brier", "ROC-AUC"]
    split_order = ["train", "val", "test"]
    split_labels = ["Train", "Validation", "Test"]

    x = np.arange(len(metric_keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, split in enumerate(split_order):
        vals = [metrics[split][k] for k in metric_keys]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=split_labels[i])
        for b, v in zip(bars, vals):
            ax.annotate(
                f"{v:.3f}",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Logistic Regression — metrics by split")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_lr_confusion_matrix(
    y_true: np.ndarray,
    proba: np.ndarray,
    path: Path,
    threshold: float = 0.5,
) -> None:
    """Confusion matrix on the test set at the given probability threshold."""
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tick_labels = ["Away win (0)", "Home win (1)"]

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"LR confusion matrix (test, threshold={threshold:.2f})")

    threshold_color = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > threshold_color else "black",
                fontsize=14, fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_logistic_regression_results_md(
    path: Path,
    *,
    feature_names: list[str],
    row_counts: dict[str, int],
    metrics: dict[str, dict[str, float]],
    metrics_fig: Path | None = None,
    confusion_fig: Path | None = None,
) -> None:
    """Write train / validation / test metrics for Logistic Regression."""
    iso_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Logistic Regression — NFL home win (2024)",
        "",
        f"*Generated: {iso_now}*",
        "",
        "## How to train and evaluate",
        "",
        "From the project root (with dependencies installed):",
        "",
        "```bash",
        "python3 -m venv .venv",
        "source .venv/bin/activate   # Windows: .venv\\Scripts\\activate",
        "pip install -r requirements/requirements.txt",
        "python code/main.py",
        "```",
        "",
        "Logistic Regression runs first; this file is overwritten each run. To only run Logistic Regression (faster):",
        "",
        "```bash",
        "python code/main.py --lr-only",
        "```",
        "",
        "## Data and split",
        "",
        f"- **CSV:** `{DATA_PATH.relative_to(PROJECT_ROOT)}`",
        f"- **Train:** regular season weeks **1–{TRAIN_MAX_WEEK}** ({row_counts['train']} games)",
        f"- **Validation:** weeks **{TRAIN_MAX_WEEK + 1}–{VAL_MAX_WEEK}** ({row_counts['val']} games)",
        f"- **Test:** week **>{VAL_MAX_WEEK}** (late season + playoffs; {row_counts['test']} games)",
        "",
        "Metrics use predicted **P(home team wins)** vs label `Home_Win`.",
        "",
        "## Model",
        "",
        "- **Pipeline:** `SimpleImputer(median)` → `StandardScaler` → `LogisticRegression`",
        "- **Hyperparameters:** `max_iter=5000`, `C=0.5`, `class_weight='balanced'`, `random_state=42`",
        "",
        "## Metrics",
        "",
        "| Split | Games | Log loss | Brier | ROC-AUC |",
        "|-------|------:|-----------:|------:|--------:|",
    ]
    for split_key, label in [("train", "Train"), ("val", "Validation"), ("test", "Test")]:
        m = metrics[split_key]
        n = row_counts[split_key]
        lines.append(
            f"| {label} | {n} | {m['log_loss']:.4f} | {m['brier']:.4f} | {m['roc_auc']:.4f} |"
        )
    lines += [
        "",
        "- **Log loss** — lower is better; penalizes overconfident wrong probabilities.",
        "- **Brier** — lower is better; mean squared error of probabilities vs 0/1 outcome.",
        "- **ROC-AUC** — higher is better; ranking quality (not calibration).",
        "",
    ]

    if metrics_fig is not None or confusion_fig is not None:
        lines.append("## Visualizations")
        lines.append("")
        if metrics_fig is not None:
            rel = metrics_fig.relative_to(PROJECT_ROOT)
            lines.append(f"![Metrics by split]({rel})")
            lines.append("")
        if confusion_fig is not None:
            rel = confusion_fig.relative_to(PROJECT_ROOT)
            lines.append(f"![Confusion matrix (test, threshold=0.5)]({rel})")
            lines.append("")

    lines += [
        "## Features used",
        "",
        f"**Count:** {len(feature_names)}",
        "",
    ]
    for name in feature_names:
        lines.append(f"- `{name}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def run_model(
    name: str,
    clf,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
) -> None:
    model = make_pipeline(clf)
    model.fit(X_train, y_train)
    print(f"\nMetrics for {name} (probability = P(home win)):")
    evaluate_split("train", y_train, model.predict_proba(X_train)[:, 1])
    evaluate_split("val  ", y_val, model.predict_proba(X_val)[:, 1])
    evaluate_split("test ", y_test, model.predict_proba(X_test)[:, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="NFL home win models")
    parser.add_argument(
        "--lr-only",
        action="store_true",
        help="Train only Logistic Regression, write LogisticRegression_results.md, and exit.",
    )
    args = parser.parse_args()

    games = load_games()
    df = build_feature_frame(games)
    cols = feature_columns()

    train_m, val_m, test_m = week_masks(df)

    X = df[cols].to_numpy(dtype=float)
    y = df["Home_Win"].to_numpy(dtype=int)

    X_train, y_train = X[train_m], y[train_m]
    X_val, y_val = X[val_m], y[val_m]
    X_test, y_test = X[test_m], y[test_m]

    row_counts = {
        "train": int(X_train.shape[0]),
        "val": int(X_val.shape[0]),
        "test": int(X_test.shape[0]),
    }

    print("NFL Win/Loss — expanded pre-kickoff features")
    print(f"  Feature count: {len(cols)}")
    print(
        f"  Splits by week — train: week<={TRAIN_MAX_WEEK}, "
        f"val: {TRAIN_MAX_WEEK + 1}–{VAL_MAX_WEEK}, test: week>{VAL_MAX_WEEK}"
    )
    print(
        f"  Rows — train: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}"
    )

    lr_model = make_pipeline(
        LogisticRegression(
            max_iter=5000,
            C=0.5,
            class_weight="balanced",
            random_state=42,
        )
    )
    lr_model.fit(X_train, y_train)
    lr_metrics = {
        "train": compute_split_metrics(y_train, lr_model.predict_proba(X_train)[:, 1]),
        "val": compute_split_metrics(y_val, lr_model.predict_proba(X_val)[:, 1]),
        "test": compute_split_metrics(y_test, lr_model.predict_proba(X_test)[:, 1]),
    }
    test_proba = lr_model.predict_proba(X_test)[:, 1]
    plot_lr_metrics_bar(lr_metrics, LR_METRICS_FIG)
    plot_lr_confusion_matrix(y_test, test_proba, LR_CONFUSION_FIG)

    write_logistic_regression_results_md(
        LOGISTIC_RESULTS_MD,
        feature_names=cols,
        row_counts=row_counts,
        metrics=lr_metrics,
        metrics_fig=LR_METRICS_FIG,
        confusion_fig=LR_CONFUSION_FIG,
    )

    print("\nMetrics for Logistic Regression (probability = P(home win)):")
    evaluate_split("train", y_train, lr_model.predict_proba(X_train)[:, 1])
    evaluate_split("val  ", y_val, lr_model.predict_proba(X_val)[:, 1])
    evaluate_split("test ", y_test, lr_model.predict_proba(X_test)[:, 1])
    print(f"\nWrote Logistic Regression report to {LOGISTIC_RESULTS_MD.relative_to(PROJECT_ROOT)}")

    if args.lr_only:
        return

    run_model(
        "Random Forest",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        X_train, y_train, X_val, y_val, X_test, y_test,
    )

    run_model(
        "KNN",
        KNeighborsClassifier(n_neighbors=15, weights="distance"),
        X_train, y_train, X_val, y_val, X_test, y_test,
    )

    run_model(
        "MLP",
        MLPClassifier(
            max_iter=2000,
            random_state=42,
            hidden_layer_sizes=(64, 16),
            activation="relu",
            alpha=1e-3,
        ),
        X_train, y_train, X_val, y_val, X_test, y_test,
    )


if __name__ == "__main__":
    main()

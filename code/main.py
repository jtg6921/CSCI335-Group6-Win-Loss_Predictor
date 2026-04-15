"""
NFL home win prediction — steps 1–3: task definition, time-based validation, baseline model.

Task: binary classification — predict whether the home team wins (label: Home_Win).
Features use only information available before kickoff: prior win rate and rest days
per team, plus week number. Rows are processed in kickoff order so same-day games
never see each other’s outcomes.

The wide Home_* / Away_* stat columns in ML_Ready_NFL_2024.csv are full-season-style
aggregates and are not used here (they are not honest pre-game signals as merged).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "ML_Ready_NFL_2024.csv"

# Playoff rows use string labels; map to synthetic week numbers after regular season (18)
_PLAYOFF_WEEK = {
    "WildCard": 19.0,
    "Division": 20.0,
    "ConfChamp": 21.0,
    "SuperBowl": 22.0,
}

# Time-based split by NFL week (regular season structure)
TRAIN_MAX_WEEK = 12
VAL_MAX_WEEK = 14

FEATURE_COLUMNS = [
    "week",
    "home_win_pct_prior",
    "away_win_pct_prior",
    "home_games_prior",
    "away_games_prior",
    "home_rest_days",
    "away_rest_days",
]

NEUTRAL_WIN_PCT = 0.5
DEFAULT_REST_DAYS = 7.0  # stand-in when a team has no prior game in the sample


def _normalize_week(raw: object) -> float:
    s = str(raw).strip()
    if s in _PLAYOFF_WEEK:
        return _PLAYOFF_WEEK[s]
    try:
        return float(int(s))
    except ValueError:
        return float("nan")


def load_games(path: Path = DATA_PATH) -> pd.DataFrame:
    usecols = ["Week", "Date", "Time", "Home", "Away", "Home_Win"]
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
    return df


def add_pregame_features(games: pd.DataFrame) -> pd.DataFrame:
    """Append pre-game features using only games strictly before each kickoff."""
    wins: dict[str, list[int]] = {}
    last_played: dict[str, pd.Timestamp] = {}

    rows = []
    for _, row in games.iterrows():
        home = row["Home"]
        away = row["Away"]
        t = row["kickoff"]

        hw = wins.get(home, [])
        aw = wins.get(away, [])
        hg, ag = len(hw), len(aw)
        h_pct = float(np.mean(hw)) if hg else NEUTRAL_WIN_PCT
        a_pct = float(np.mean(aw)) if ag else NEUTRAL_WIN_PCT

        h_rest = (
            (t - last_played[home]).total_seconds() / 86400.0
            if home in last_played
            else DEFAULT_REST_DAYS
        )
        a_rest = (
            (t - last_played[away]).total_seconds() / 86400.0
            if away in last_played
            else DEFAULT_REST_DAYS
        )

        rows.append(
            {
                "week": float(row["week"]),
                "home_win_pct_prior": h_pct,
                "away_win_pct_prior": a_pct,
                "home_games_prior": float(hg),
                "away_games_prior": float(ag),
                "home_rest_days": h_rest,
                "away_rest_days": a_rest,
                "Home_Win": int(row["Home_Win"]),
                "kickoff": t,
            }
        )

        home_won = bool(row["Home_Win"])
        wins.setdefault(home, []).append(1 if home_won else 0)
        wins.setdefault(away, []).append(0 if home_won else 1)
        last_played[home] = t
        last_played[away] = t

    return pd.DataFrame(rows)


def week_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = df["week"].to_numpy()
    train = w <= TRAIN_MAX_WEEK
    val = (w > TRAIN_MAX_WEEK) & (w <= VAL_MAX_WEEK)
    test = w > VAL_MAX_WEEK
    return train, val, test


def evaluate_split(name: str, y_true: np.ndarray, proba: np.ndarray) -> None:
    ll = log_loss(y_true, proba)
    brier = brier_score_loss(y_true, proba)
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = float("nan")
    print(f"  {name}: log_loss={ll:.4f}  brier={brier:.4f}  roc_auc={auc:.4f}")


def main() -> None:
    games = load_games()
    df = add_pregame_features(games)
    train_m, val_m, test_m = week_masks(df)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["Home_Win"].to_numpy(dtype=int)

    X_train, y_train = X[train_m], y[train_m]
    X_val, y_val = X[val_m], y[val_m]
    X_test, y_test = X[test_m], y[test_m]

    print("NFL Win/Loss — baseline (logistic regression)")
    print(f"  Task: predict Home_Win; features: {FEATURE_COLUMNS}")
    print(
        f"  Splits by week — train: week<={TRAIN_MAX_WEEK}, "
        f"val: {TRAIN_MAX_WEEK + 1}–{VAL_MAX_WEEK}, test: week>{VAL_MAX_WEEK}"
    )
    print(
        f"  Rows — train: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}"
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    print("\nMetrics (probability = P(home win)):")
    evaluate_split("train", y_train, model.predict_proba(X_train)[:, 1])
    evaluate_split("val  ", y_val, model.predict_proba(X_val)[:, 1])
    evaluate_split("test ", y_test, model.predict_proba(X_test)[:, 1])


if __name__ == "__main__":
    main()

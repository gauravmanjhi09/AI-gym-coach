"""
services/tracking/stats.py

Calculation helpers for workout analytics: streaks, personal bests,
weekly volume, and goal progress. Pure functions that operate on the
same row shape returned by get_users_exercises() (dicts with keys:
exercise_name, reps, sets, time, created_at).
"""

from datetime import datetime, timedelta
import pandas as pd


def _to_dataframe(history_rows):
    """Convert raw DB rows into a clean DataFrame with a proper date column.

    history_rows may be a list of sqlite3.Row objects. pandas does NOT treat
    sqlite3.Row as dict-like when building a DataFrame directly (it falls
    back to positional/integer columns), so each row is converted to a
    plain dict first to guarantee named columns."""
    if not history_rows:
        return pd.DataFrame(columns=["exercise_name", "reps", "sets", "time", "created_at", "date"])

    df = pd.DataFrame([dict(row) for row in history_rows])
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at"].dt.date
    return df


def calculate_streak(history_rows):
    """
    Returns (current_streak, longest_streak) in consecutive days.
    A day counts if at least one workout was logged that day.
    """
    df = _to_dataframe(history_rows)
    if df.empty:
        return 0, 0

    workout_dates = sorted(set(df["date"]))

    # Longest streak
    longest = 1
    current_run = 1
    for i in range(1, len(workout_dates)):
        if (workout_dates[i] - workout_dates[i - 1]).days == 1:
            current_run += 1
            longest = max(longest, current_run)
        else:
            current_run = 1

    # Current streak (must include today or yesterday to still be "alive")
    today = datetime.now().date()
    current_streak = 0
    check_date = today

    date_set = set(workout_dates)

    if today not in date_set:
        check_date = today - timedelta(days=1)
        if check_date not in date_set:
            return 0, longest

    while check_date in date_set:
        current_streak += 1
        check_date -= timedelta(days=1)

    return current_streak, longest


def personal_bests(history_rows):
    """
    Returns a dict: { exercise_name: {"max_reps": int, "max_sets": int, "best_date": date} }
    """
    df = _to_dataframe(history_rows)
    if df.empty:
        return {}

    bests = {}
    for exercise, group in df.groupby("exercise_name"):
        best_row = group.loc[group["reps"].idxmax()]
        bests[exercise] = {
            "max_reps": int(group["reps"].max()),
            "max_sets": int(group["sets"].max()),
            "best_date": best_row["date"],
        }
    return bests


def weekly_volume(history_rows, weeks_back=8):
    """
    Returns a DataFrame with columns [week_start, total_reps, total_sets]
    for the last `weeks_back` weeks (fills in zero weeks so charts don't skip).
    """
    df = _to_dataframe(history_rows)

    today = datetime.now().date()
    week_starts = [today - timedelta(days=today.weekday() + 7 * i) for i in range(weeks_back)]
    week_starts = sorted(week_starts)

    if df.empty:
        return pd.DataFrame({
            "week_start": week_starts,
            "total_reps": [0] * len(week_starts),
            "total_sets": [0] * len(week_starts),
        })

    df["week_start"] = df["created_at"].dt.to_period("W").apply(lambda p: p.start_time.date())

    grouped = df.groupby("week_start").agg(
        total_reps=("reps", "sum"),
        total_sets=("sets", "sum"),
    ).reset_index()

    result = pd.DataFrame({"week_start": week_starts})
    result = result.merge(grouped, on="week_start", how="left").fillna(0)
    result["total_reps"] = result["total_reps"].astype(int)
    result["total_sets"] = result["total_sets"].astype(int)
    return result


def exercise_distribution(history_rows):
    """
    Returns a DataFrame [exercise_name, total_sets] for pie/donut chart.
    """
    df = _to_dataframe(history_rows)
    if df.empty:
        return pd.DataFrame(columns=["exercise_name", "total_sets"])

    return df.groupby("exercise_name")["sets"].sum().reset_index(name="total_sets")


def consistency_calendar(history_rows, days_back=84):
    """
    Returns a DataFrame [date, workout_count] for the last `days_back` days,
    used to build a GitHub-style contribution heatmap. Missing days are 0.
    """
    df = _to_dataframe(history_rows)

    today = datetime.now().date()
    all_days = [today - timedelta(days=i) for i in range(days_back)][::-1]

    if df.empty:
        return pd.DataFrame({"date": all_days, "workout_count": [0] * len(all_days)})

    daily = df.groupby("date").size().reset_index(name="workout_count")
    result = pd.DataFrame({"date": all_days})
    result = result.merge(daily, on="date", how="left").fillna(0)
    result["workout_count"] = result["workout_count"].astype(int)
    return result


def total_summary(history_rows):
    """Quick top-line numbers for summary cards."""
    df = _to_dataframe(history_rows)
    if df.empty:
        return {"total_workouts": 0, "total_reps": 0, "total_sets": 0, "total_time_min": 0}

    return {
        "total_workouts": len(df["date"].unique()),
        "total_reps": int(df["reps"].sum()),
        "total_sets": int(df["sets"].sum()),
        "total_time_min": round(df["time"].sum() / 60, 1) if "time" in df.columns else 0,
    }
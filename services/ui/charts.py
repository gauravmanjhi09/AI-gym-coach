"""
services/ui/charts.py

Plotly figure builders for the Analytics section. Each function takes
already-computed data (from services.tracking.stats) and returns a
plotly.graph_objects Figure, themed to match the app's dark UI.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Matches static/style.css "Performance Telemetry" theme tokens
ACCENT = "#00D4B4"          # Signal Teal
ACCENT_SECONDARY = "#FFB020"  # Phosphor Amber
BG_TRANSPARENT = "rgba(0,0,0,0)"
TEXT_COLOR = "#E0E0E0"
GRID_COLOR = "rgba(255,255,255,0.08)"

# Cycled across exercises in multi-series charts (distribution, per-exercise lines)
PALETTE = ["#00D4B4", "#FFB020", "#FF5A5F", "#5AC8FA", "#B980F0", "#8FD694"]


def _base_layout(fig, height=320):
    fig.update_layout(
        paper_bgcolor=BG_TRANSPARENT,
        plot_bgcolor=BG_TRANSPARENT,
        font=dict(color=TEXT_COLOR, family="AdobeClean, sans-serif"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    return fig


def progress_line_chart(history_df, exercise_filter=None):
    """
    Line chart of reps over time. history_df needs columns: date, exercise_name, reps.
    If exercise_filter is given, shows only that exercise; otherwise one line per exercise.
    """
    if history_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data yet — complete a workout to see progress",
                            showarrow=False, font=dict(color=TEXT_COLOR))
        return _base_layout(fig)

    df = history_df.copy()
    if exercise_filter and exercise_filter != "All":
        df = df[df["exercise_name"] == exercise_filter]

    daily = df.groupby(["date", "exercise_name"])["reps"].sum().reset_index()

    fig = px.line(
        daily, x="date", y="reps", color="exercise_name",
        markers=True, title="Reps Over Time",
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_xaxes(type="date", tickformat="%b %d")
    return _base_layout(fig)


def weekly_volume_chart(weekly_df):
    """Bar chart of total reps per week. weekly_df: [week_start, total_reps, total_sets]."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weekly_df["week_start"], y=weekly_df["total_reps"],
        marker_color=ACCENT, name="Total Reps",
    ))
    fig.update_layout(title="Weekly Volume (Reps)")
    return _base_layout(fig)


def exercise_distribution_chart(dist_df):
    """Donut chart of sets performed per exercise."""
    if dist_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data yet", showarrow=False, font=dict(color=TEXT_COLOR))
        return _base_layout(fig, height=280)

    fig = go.Figure(data=[go.Pie(
        labels=dist_df["exercise_name"], values=dist_df["total_sets"],
        hole=0.55,
        marker=dict(colors=PALETTE, line=dict(color="#0a0d14", width=2)),
        textposition="inside",
        textinfo="percent",
        insidetextfont=dict(color="#0a0d14", size=14, family="JetBrains Mono"),
    )])
    fig.update_layout(showlegend=True, legend=dict(font=dict(color=TEXT_COLOR, size=11)))
    fig.update_layout(title="Set Distribution by Exercise")
    return _base_layout(fig, height=320)


def consistency_heatmap(calendar_df):
    """
    GitHub-style contribution heatmap. calendar_df: [date, workout_count].
    Renders as a week-by-weekday grid.
    """
    if calendar_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data yet", showarrow=False, font=dict(color=TEXT_COLOR))
        return _base_layout(fig, height=180)

    df = calendar_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.weekday  # 0=Mon
    df["week"] = (df["date"] - df["date"].min()).dt.days // 7

    pivot = df.pivot_table(index="weekday", columns="week", values="workout_count", fill_value=0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        colorscale=[[0, "rgba(255,255,255,0.05)"], [1, ACCENT]],
        showscale=False,
        xgap=3, ygap=3,
    ))
    fig.update_layout(
        title="Consistency (last 12 weeks)",
        yaxis=dict(tickmode="array", tickvals=list(range(7)),
                   ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
        xaxis=dict(showticklabels=False),
    )
    return _base_layout(fig, height=220)
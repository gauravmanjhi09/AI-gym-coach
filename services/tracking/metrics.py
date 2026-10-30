import streamlit as st
import time
from services.config.workout_config import METRICS_FIELDS
from services.persistence.exercise_repository import add_exercise


def _play_event(event, exercise, metrics):
    """Voice pipeline ko event bhejo aur result session_state mein rakho."""
    pipeline = st.session_state.get("voice_pipeline")
    if not pipeline:
        return
    result = pipeline.process_event(event=event, exercise=exercise, metrics=metrics)
    if result:
        st.session_state.audio_to_play, st.session_state.coach_feedback = result


def sync_metrics_update(context):
    if not context or not hasattr(context, "state") or not context.state.playing:
        return

    processor = getattr(context, "video_processor", None)
    if processor is None:
        return

    exercise = st.session_state.get("exercise_type")
    if not exercise:
        return

    # Method exist na kare to crash ke bajay skip karo
    set_exercise = getattr(processor, "set_exercise", None)
    get_metrics = getattr(processor, "get_latest_metrics", None)
    if not callable(set_exercise) or not callable(get_metrics):
        return

    set_exercise(exercise)
    latest_metrics = get_metrics()
    if not latest_metrics:
        return

    reps = latest_metrics.get("reps") or 0
    st.session_state.reps = reps

    fields = METRICS_FIELDS.get(exercise)
    if not fields:
        return

    for key, default in fields.items():
        st.session_state[key] = latest_metrics.get(key, default)

    reps_per_set = st.session_state.get("reps_per_set") or 0
    target_sets = st.session_state.get("target_sets") or 0

    if reps_per_set > 0 and target_sets > 0:
        sets_completed = reps // reps_per_set
        current_set_reps = reps % reps_per_set
        workout_completed = sets_completed >= target_sets
    else:
        sets_completed = 0
        current_set_reps = 0
        workout_completed = False

    st.session_state.sets_completed = sets_completed
    st.session_state.current_set_reps = current_set_reps
    st.session_state.workout_completed = workout_completed

    last_saved_sets = st.session_state.get("last_saved_sets_completed", 0)

    if reps_per_set > 0 and target_sets > 0 and sets_completed > last_saved_sets:
        newly_completed = sets_completed - last_saved_sets
        now_ts = time.time()
        started_at = st.session_state.get("set_cycle_started_at", now_ts)
        time_taken = now_ts - started_at
        user_id = st.session_state.get("user_id", 0)

        add_exercise(user_id, exercise, newly_completed * reps_per_set, newly_completed, time_taken)
        _play_event("set_completed", exercise, latest_metrics)

        st.session_state.set_cycle_started_at = now_ts
        st.session_state.last_saved_sets_completed = sets_completed

    if workout_completed and not st.session_state.get("last_notified_workout_complete", False):
        st.session_state.last_notified_workout_complete = True
        _play_event("workout_completed", exercise, latest_metrics)

    if not latest_metrics.get("pose_detected", True):
        _play_event(
            "no_pose_detected",
            exercise,
            {"issue": "No pose detected! Please step into the camera frame."},
        )
    else:
        _play_event("ongoing_form_check", exercise, latest_metrics)
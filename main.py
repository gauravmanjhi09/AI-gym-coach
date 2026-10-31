import streamlit as st
import os
import time
import pandas as pd
from services.auth.login_wall import render_login_wall
from services.state.session_defaults import initial_session_defaults
from services.config.workout_config import EXERCISE_OPTIONS
from services.ui.style_loader import load_css, inject_local_font, inject_webrtc_styles
from services.persistence.exercise_repository import init_db
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from services.vision.exercise_video_processor import VideoProcessorClass
from services.tracking.metrics import sync_metrics_update
from services.persistence.exercise_repository import get_users_exercises
from groq import Groq
from services.coaching.llm import LLMCoach
from services.coaching.tts import TextToSpeech
from services.coaching.voice_pipeline import VoicePipeline, autoplay_audio

# --- Analytics additions ---
from services.tracking.stats import (
    calculate_streak,
    personal_bests,
    weekly_volume,
    exercise_distribution,
    consistency_calendar,
    total_summary,
)
from services.ui.charts import (
    progress_line_chart,
    weekly_volume_chart,
    exercise_distribution_chart,
    consistency_heatmap,
)


def _to_dataframe_for_ui(history_rows):
    """Same date-parsing logic as services/tracking/stats.py, kept in sync
    so the UI and the stats calculations never disagree about 'date'.

    history_rows may be a list of sqlite3.Row objects, which pandas does
    NOT treat as dict-like when building a DataFrame (it falls back to
    positional/integer columns). Converting each row to a plain dict first
    guarantees named columns (created_at, exercise_name, reps, sets, time)."""
    df = pd.DataFrame([dict(row) for row in history_rows])
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at"].dt.date
    return df


def main():
    st.set_page_config(
        page_icon="🏋️‍♀️",
        page_title="Real-time AI GYM Coach",
        initial_sidebar_state="expanded",
        layout="centered"
    )

    load_css(os.path.join(os.getcwd(), "static", "style.css"))
    inject_local_font(os.path.join(os.getcwd(), "static", "AdobeClean.otf"), "AdobeClean")

    init_db()

    if not render_login_wall():
        return

    initial_session_defaults()

    if "voice_pipeline" not in st.session_state:
        try:
            api_key = os.environ.get("GROQ_API_KEY", "")

            if not api_key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]

            groq_client = Groq(api_key=api_key)
            llm_coach = LLMCoach(groq_client)
            tts = TextToSpeech()
            st.session_state.voice_pipeline = VoicePipeline(llm_coach, tts)
        except Exception as e:
            st.session_state.voice_pipeline = None

    workout_started = st.session_state.get("workout_started", False)

    with st.sidebar:
        st.title("🏋🏻‍♂️ AI GYM Trainer")

        if st.session_state.username:
            st.caption(f"👤 Login as: {st.session_state.username}")

        st.divider()

        st.subheader("Workout Plan")

        if not workout_started:
            plan_exercise = st.selectbox("Exercise", options = EXERCISE_OPTIONS, key="plan_exercise")

            plan_sets = st.number_input("Sets", min_value=0, max_value=50, key="plan_sets",step=1)

            plan_reps = st.number_input("Reps Per Set", min_value=0, max_value=50, key="plan_reps",step=1)

            st.markdown("")

            start_session_button =st.button("Start Workout",width = "stretch",  key="start_session_button")

            if start_session_button:
                st.session_state.exercise_type = plan_exercise
                st.session_state.target_sets = int(plan_sets)
                st.session_state.reps_per_set = int(plan_reps)
                st.session_state.reps = 0
                st.session_state.workout_started = True
                st.session_state.set_cycle_started_at = time.time()
                st.session_state.last_saved_sets_completed = 0

                if st.session_state.voice_pipeline:
                    result = st.session_state.voice_pipeline.process_event(
                        event="workout_started",
                        exercise=plan_exercise,
                        metrics={}
                    )

                    if result:
                        st.session_state.audio_to_play, st.session_state.coach_feedback = result

                st.session_state.last_notified_sets_completed = 0
                st.session_state.last_notified_workout_complete = False
                st.rerun()
        else:
            exercise = st.session_state.get("exercise_type")
            sets = st.session_state.get("target_sets")
            reps = st.session_state.get("reps_per_set")

            st.info(f"** {exercise}** -- {sets} Sets /  {reps} Reps")

            end_session_button = st.button("End Workout", key="end_session_button",width = "stretch")

            if end_session_button:
                st.session_state.workout_started = False

                if st.session_state.voice_pipeline:
                    result = st.session_state.voice_pipeline.process_event(
                        event="workout_completed",
                        exercise=exercise,
                        metrics={}
                    )
                    if result:
                        st.session_state.audio_to_play, st.session_state.coach_feedback = result

                st.rerun()

        if workout_started:
            st.divider()
            exercise = st.session_state.get("exercise_type")
            total_reps = st.session_state.get("reps")
            current_set_reps = st.session_state.get("current_set_reps")
            reps_per_set = st.session_state.get("reps_per_set")
            sets_completed = st.session_state.get("sets_completed")
            target_sets = st.session_state.get("target_sets")


            st.subheader("Workout Progress")

            st.metric("Total Reps", f"{total_reps}")
            st.metric("Current Set Reps", f"{current_set_reps} / {reps_per_set}")
            st.metric("Sets Completed", f"{sets_completed} / {target_sets}")

            st.divider()

            if exercise == "Squats":
                st.subheader("Squat Metrics")
                st.metric("Knee Angle", f"{st.session_state.knee_angle}°")
                st.metric("Back Angle", f"{st.session_state.back_angle}°")
                st.metric("Depth Status", st.session_state.depth_status)

            elif exercise == "Push-ups":
                st.subheader("Push-up Metrics")
                st.metric("Elbow Angle", f"{st.session_state.elbow_angle}°")
                st.metric("Body Alignment", st.session_state.body_alignment)
                st.metric("Hip Position", st.session_state.hip_status)

            elif exercise == "Biceps Curls (Dumbbell)":
                st.subheader("Curl Metrics")
                st.metric("Elbow Angle", f"{st.session_state.elbow_angle}°")
                st.metric("Shoulder Stability", st.session_state.shoulder_status)
                st.metric("Swing Detection", st.session_state.swing_status)

            elif exercise == "Shoulder Press":
                st.subheader("Shoulder Press Metrics")
                st.metric("Elbow Angle", f"{st.session_state.elbow_angle}°")
                st.metric("Arm Extension", st.session_state.extension_status)
                st.metric("Back Arch", st.session_state.back_arch_status)

            elif exercise == "Lunges":
                st.subheader("Lunge Metrics")
                st.metric("Front Knee Angle", f"{st.session_state.front_knee_angle}°")
                st.metric("Torso Angle", f"{st.session_state.torso_angle}°")
                st.metric("Balance Status", st.session_state.balance_status)


    st.title("AI Real-time GYM Coach")
    st.markdown("#### Real-time pose detection with proactive AI voice coaching")

    if st.session_state.get("audio_to_play"):
        autoplay_audio(st.session_state.audio_to_play)

    if st.session_state.get("coach_feedback"):
        st.markdown("")
        st.success(f"🤖 **Coach:** {st.session_state.coach_feedback}")

    if not workout_started:
        st.markdown(
            """
            <div style="
                border: 10px dashed #444;
                border-radius: 0px;
                padding: 48px 32px;
                text-align: center;
                color: #888;
                margin-top: 32px;
                margin-bottom: 32px;
            ">
                <h2 style="color:#ccc; margin-bottom:8px;">👈 Set your workout plan</h2>
                <p style="font-size:1.05rem;">
                    Choose your exercise, sets and reps in the sidebar,<br>
                    then click <strong>Start Workout</strong> to activate the camera and AI coach.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        # Lower resolution + FPS so the (shared, weak) cloud CPU can keep up
        context = webrtc_streamer(
            key="exercise-analysis",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessorClass,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15, "max": 15},
                },
                "audio": False,
            },
            async_processing=True,
        )

        sync_metrics_update(context)

        if context.state.playing:
            # 1s instead of 0.25s: fewer full-script reruns on the cloud
            time.sleep(1.0)
            st.rerun()

        inject_webrtc_styles()


    st.divider()
    st.markdown("### 📊 Analytics")

    user_id = st.session_state.get("user_id", 0)

    if isinstance(user_id, int):
        history_rows = get_users_exercises(user_id)

        if not history_rows:
            st.info("No workout history yet — complete a workout to see your analytics.")
        else:
            history_df = _to_dataframe_for_ui(history_rows)


            summary = total_summary(history_rows)
            current_streak, longest_streak = calculate_streak(history_rows)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🔥 Current Streak", f"{current_streak} days")
            c2.metric("🏆 Longest Streak", f"{longest_streak} days")
            c3.metric("💪 Total Reps", summary["total_reps"])
            c4.metric("📅 Workouts Logged", summary["total_workouts"])

            st.markdown("")

            # ---- Exercise filter + progress line chart ----
            exercise_names = sorted(history_df["exercise_name"].unique().tolist())
            selected_exercise = st.selectbox(
                "Filter by exercise", options=["All"] + exercise_names, key="analytics_filter"
            )
            st.plotly_chart(
                progress_line_chart(history_df, selected_exercise),
                width="stretch",
                config={"displayModeBar": False},
            )

            # ---- Weekly volume + distribution side by side ----
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    weekly_volume_chart(weekly_volume(history_rows)),
                    width="stretch",
                    config={"displayModeBar": False},
                )
            with col_b:
                st.plotly_chart(
                    exercise_distribution_chart(exercise_distribution(history_rows)),
                    width="stretch",
                    config={"displayModeBar": False},
                )

            # ---- Consistency heatmap ----
            st.plotly_chart(
                consistency_heatmap(consistency_calendar(history_rows)),
                width="stretch",
                config={"displayModeBar": False},
            )

            # ---- Personal bests ----
            st.markdown("#### 🏅 Personal Bests")
            bests = personal_bests(history_rows)
            if bests:
                cols = st.columns(len(bests))
                for col, (exercise, data) in zip(cols, bests.items()):
                    with col:
                        st.markdown(
                            f"""
                            <div style="border:1px solid #333; border-radius:8px; padding:12px; text-align:center;">
                                <div style="font-size:0.85rem; color:#999;">{exercise}</div>
                                <div style="font-size:1.4rem; font-weight:700; color:#00C2A8;">{data['max_reps']} reps</div>
                                <div style="font-size:0.75rem; color:#777;">{data['best_date']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


            with st.expander("View raw workout log"):
                agg_df = history_df.groupby(["exercise_name", "date"]).agg(
                    Reps=("reps", "sum"), Sets=("sets", "sum"), Time=("time", "sum")
                ).reset_index()
                agg_df.index += 1
                st.table(agg_df)


if __name__ == "__main__":

    main()
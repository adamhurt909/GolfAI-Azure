import streamlit as st
import subprocess
import sys
import json
from pathlib import Path
import shutil
from datetime import datetime
import shutil
import pandas as pd

# -------------------------
# Page setup
# -------------------------

st.set_page_config(
    page_title="GolfAI Swing Analysis",
    page_icon="🏌️",
    layout="wide"
)

st.write(
    "Upload a golf swing video, detect the key swing frames, "
    "calculate movement metrics, and generate AI coaching feedback."
)

st.caption("GolfAI Container Version 1: 18-Aug-2026")

page = st.sidebar.radio(
    "Navigation",
    [
        "Analyse Swing",
        "Compare Swings",
        "Swing History"
    ]
)

# -------------------------
# Development Options
# -------------------------

def score_metric(
    value,
    excellent_min,
    good_min
):
    if value >= excellent_min:
        return 100, "🟢 Excellent"

    elif value >= good_min:
        return 70, "🟡 Good"

    else:
        return 40, "🔴 Needs Work"

def score_bar(score):

    if score >= 90:
        colour = "#16A34A"     # Green

    elif score >= 70:
        colour = "#EAB308"     # Amber

    else:
        colour = "#DC2626"     # Red

    st.markdown(
        f"""
        <div style="
            background-color:#E5E7EB;
            border-radius:12px;
            height:24px;
            width:100%;
            overflow:hidden;
        ">
            <div style="
                background-color:{colour};
                height:24px;
                width:{score}%;
                display:flex;
                align-items:center;
                justify-content:center;
                color:white;
                font-weight:bold;
                font-size:14px;
                border-radius:12px;
            ">
                {score}/100
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# Metric Targets
# -------------------------

METRIC_TARGETS = {

    "Head Movement":
        "🎯 Target: As close to 0 as possible",

    "Lead Arm Angle":
        "🎯 Target: 130°-170°. A wider lead arm helps create width and power.",

    "Trail Arm Angle":
        "🎯 Target: 70°-100°. A folded trail arm creates leverage at the top.",

    "Spine Angle Change":
        "🎯 Target: As close to 0° as possible. Maintaining posture improves consistency.",

    "Tempo Ratio":
        "🎯 Target: Approximately 3:1",

    "Swing Width":
        "🎯 Target: Maintain width in the backswing",

    "Hip Sway":
        "🎯 Target: Minimise lateral hip movement during the backswing",

    "Finish Stability":
        "🎯 Target: Hold a stable finish position after the swing"

}

# -------------------------
# Recommended Drill Videos
# -------------------------

DRILL_MAP = {

    "Head Stability": {
        "name": "Head Against Wall Drill",
        "description":
            "Helps reduce excessive head movement "
            "between Address and Impact.",
        "video_url":
            "https://www.youtube.com/watch?v=njKeUckRSrM"
    },

    "Lead Arm Width": {
        "name": "Lead Arm Extension Drill",
        "description":
            "Helps create width and extension "
            "during the backswing.",
        "video_url":
            "https://www.youtube.com/watch?v=FrVS4L47YmE"
    },

    "Trail Arm Position": {
        "name": "Towel Under Arm Drill",
        "description":
            "Helps improve trail-arm structure "
            "and connection.",
        "video_url":
            "https://www.youtube.com/watch?v=2FmhOSsOLuM"
    },

    "Posture": {
        "name": "Chair Posture Drill",
        "description":
            "Helps maintain spine angle and posture "
            "through Impact.",
        "video_url":
            "https://www.youtube.com/watch?v=xzvtN-0dRI4"
    },

    "Tempo": {
        "name": "3:1 Tempo Drill",
        "description":
            "Helps improve the timing relationship "
            "between the backswing and downswing.",
        "video_url":
            "https://www.youtube.com/watch?v=ECcFoD4C8j0"
    },

    "Swing Width": {
        "name": "Alignment Stick Width Drill",
        "description":
            "Helps create a wider backswing arc "
            "and better extension.",
        "video_url":
            "https://www.youtube.com/watch?v=umKkps7yHuU"
    },

    "Hip Sway": {
        "name": "Hip Barrier Drill",
        "description":
            "Helps reduce excessive lateral hip movement "
            "during the backswing.",
        "video_url":
            "https://www.youtube.com/watch?v=Q0zvcYZPOok"
    },

    "Finish Stability": {
        "name": "Hold The Finish Drill",
        "description":
            "Helps improve stability and control "
            "after the swing.",
        "video_url":
            "https://www.youtube.com/watch?v=O6HjVDf9OnA"
    }
}

def compare_metric(metric_name, value_1, value_2):

    if metric_name == "Head Movement":

        return (
            compare_1
            if value_1 < value_2
            else compare_2
        )

    elif metric_name == "Lead Arm Angle":

        target = 150

        diff_1 = abs(
            value_1 - target
        )

        diff_2 = abs(
            value_2 - target
        )

        return (
            compare_1
            if diff_1 < diff_2
            else compare_2
        )

    elif metric_name == "Trail Arm Angle":

        target = 85

        diff_1 = abs(
            value_1 - target
        )

        diff_2 = abs(
            value_2 - target
        )

        return (
            compare_1
            if diff_1 < diff_2
            else compare_2
        )

    elif metric_name == "Spine Angle Change":

        return (
            compare_1
            if abs(value_1)
            < abs(value_2)
            else compare_2
        )

    elif metric_name == "Tempo Ratio":

        target = 3.0

        diff_1 = abs(
            value_1 - target
        )

        diff_2 = abs(
            value_2 - target
        )

        return (
            compare_1
            if diff_1 < diff_2
            else compare_2
        )

    elif metric_name == "Swing Width":

        return (
            compare_1
            if value_1 > value_2
            else compare_2
        )

    elif metric_name == "Hip Sway":

        return (
            compare_1
            if value_1 < value_2
            else compare_2
        )

    elif metric_name == "Finish Stability":

        return (
            compare_1
            if value_1 < value_2
            else compare_2
        )

    return "Tie"

# -------------------------
# Project paths
# -------------------------

BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "videos"
DATA_DIR = BASE_DIR / "data"
ANALYSIS_DIR = BASE_DIR / "analysis"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

HISTORY_DIR = DATA_DIR / "history"
HISTORY_VIDEOS_DIR = HISTORY_DIR / "videos"

HISTORY_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HISTORY_VIDEOS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HISTORY_CSV = (
    HISTORY_DIR
    / "swing_history.csv"
)

if not HISTORY_CSV.exists():

    with open(
        HISTORY_CSV,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "date,"
            "filename,"
            "address,"
            "top,"
            "impact,"
            "finish,"
            "head_move,"
            "tempo_ratio,"
            "swing_width,"
            "hip_sway,"
            "finish_stability\n"
        )

report_folders = sorted(
    [
        folder.name
        for folder in REPORTS_DIR.iterdir()
        if folder.is_dir()
    ]
)

VIDEOS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

PYTHON_EXE = sys.executable

# -------------------------
# Upload video
# -------------------------

uploaded_file = st.file_uploader(
    "Upload swing video",
    type=["mp4", "mov", "avi"]
)

if page == "Analyse Swing":

    if uploaded_file is not None:

        video_path = VIDEOS_DIR / uploaded_file.name

        video_name = Path(
            uploaded_file.name
        ).stem

        report_folder = (
            REPORTS_DIR
            / video_name
        )

        report_folder.mkdir(
            exist_ok=True
        )

        with open(video_path, "wb") as f:
            f.write(
                uploaded_file.getbuffer()
            )
        
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )       

        history_filename = (
            f"{timestamp}_{uploaded_file.name}"
        )

        history_video_path = (
            HISTORY_VIDEOS_DIR
            / history_filename
        )

        shutil.copy2(
            video_path,
            history_video_path
        )

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        video_col, info_col = st.columns([1, 1])

        with video_col:

            st.video(str(video_path))

            analyse_button = st.button(
                "🏌️ Analyse Swing",
                use_container_width=True
            )

        with info_col:

            with st.expander(
                "📊 Metrics We Measure",
                expanded=True
            ):

                st.info(
                """
            ### 📊 GolfAI Metrics Guide

            ---

            **Head Movement**
            🎯 Target: As close to 0 as possible

            Measures head stability between Address and Impact.

            ---

            **Lead Arm Angle**
            🎯 Target: 130°-170°

            Measures width and extension at the top of the backswing.

            ---

            **Trail Arm Angle**
            🎯 Target: 70°-100°

            Measures trail arm leverage and compactness.

            ---

            **Tempo Ratio**
            🎯 Target: Approximately 3:1

            Measures backswing speed compared to downswing speed.

            ---

            **Swing Width**
            🎯 Target: Maintain width in the backswing.

            Measures the width of the swing arc.

            ---

            **Hip Sway**
            🎯 Target: Minimise lateral hip movement.

            Measures sideways hip movement during the backswing.

            ---

            **Spine Angle Change**
            🎯 Target: As close to 0° as possible.

            Measures posture retention through impact.

            ---

            **Finish Stability**
            🎯 Target: Hold a stable finish position.

            Measures how stable the finish remains after the swing.
            """
            )

        if analyse_button:

            # -------------------------
            # Run phase detector
            # -------------------------

            st.info("Detecting swing phases...")

            phase_result = subprocess.run(
               [
                    PYTHON_EXE,
                    str(ANALYSIS_DIR / "phase_detector_v2.py"),
                    str(video_path)
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            with st.expander("Phase detector output"):

                if phase_result.stdout:
                    st.text(phase_result.stdout)

    #                  if phase_result.stderr:
    #                       st.text(phase_result.stderr)

            # -------------------------
            # Output file paths
            # -------------------------

            phase_report_path = DATA_DIR / "phase_report.json"
            swing_report_path = DATA_DIR / "swing_report.json"
            coach_report_path = DATA_DIR / "coach_report.txt"

            address_image_path = DATA_DIR / "address.jpg"
            top_image_path = DATA_DIR / "top.jpg"
            impact_image_path = DATA_DIR / "impact.jpg"
            finish_image_path = DATA_DIR / "finish.jpg"

            # -------------------------
            # Phase Detection Results
            # -------------------------

            with st.expander(
                "Phase Detection Results",
                expanded=True
            ):

                if phase_report_path.exists():

                    with open(
                        phase_report_path,
                        "r"
                    ) as f:

                        phase_report = json.load(f)

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric(
                        "Address",
                        phase_report["address"]
                    )

                    col2.metric(
                        "Top",
                        phase_report["top"]
                    )

                    col3.metric(
                        "Impact",
                        phase_report["impact"]
                    )

                    col4.metric(
                        "Finish",
                        phase_report["finish"]
                    )

                st.divider()

#               Bigger, 2 then 2
                img1, img2 = st.columns(2)
                img3, img4 = st.columns(2)

#               Smaller, 4 in a row
#               img1, img2, img3, img4 = st.columns(4)

                if address_image_path.exists():
                    img1.image(
                        str(address_image_path),
                        caption="Address",
                        use_container_width=True
                    )

                if top_image_path.exists():
                    img2.image(
                        str(top_image_path),
                        caption="Top of Backswing",
                        use_container_width=True
                    )

                if impact_image_path.exists():
                    img3.image(
                        str(impact_image_path),
                        caption="Impact",
                        use_container_width=True
                    )

                if finish_image_path.exists():
                    img4.image(
                        str(finish_image_path),
                        caption="Finish",
                        use_container_width=True
                    )

            # -------------------------
            # Run pose detector
            # -------------------------

            st.info("Calculating swing metrics...")

            pose_result = subprocess.run(
                [
                    PYTHON_EXE,
                    str(ANALYSIS_DIR / "pose_detector.py"),
                    str(video_path)
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

#              with st.expander("Pose detector output"):
#                  if pose_result.stdout:
#                     st.text(pose_result.stdout)

#                 if pose_result.stderr:
#                      st.text(pose_result.stderr)

            with st.expander("Pose detector output"):

                if pose_result.stdout:

                    cleaned_output = pose_result.stdout

                    if "STRUCTURED REPORT" in cleaned_output:

                        cleaned_output = (
                            cleaned_output
                            .split("STRUCTURED REPORT")[0]
                        )

                    st.text(cleaned_output)


            # -------------------------
            # Swing Metrics
            # -------------------------

            if swing_report_path.exists():

                with open(
                    swing_report_path,
                    "r"
                ) as f:

                    swing_report = json.load(f)

                metrics = swing_report["metrics"]

                tempo_ratio = metrics[
                    "tempo_ratio"
                ]

                swing_width = metrics[
                    "swing_width"
                ]

                hip_sway = metrics[
                    "hip_sway"
                ]

                finish_stability = metrics[
                    "finish_stability"
                ]

                head_move = metrics[
                    "head_movement_address_to_impact"
                ]

                lead_arm = metrics[
                    "lead_arm_angle_at_top_degrees"
                ]

                trail_arm = metrics[
                    "trail_arm_angle_at_top_degrees"
                ]

                spine_change = abs(
                    metrics[
                        "spine_angle_change_degrees"
                    ]
                )

                # -------------------------
                # Head Movement Score
                # Lower is better
                # -------------------------

                if head_move <= 0.02:
                    head_score = 100
                    head_rating = "🟢 Excellent"

                elif head_move <= 0.05:
                    head_score = 70
                    head_rating = "🟡 Good"

                else:
                    head_score = 40
                    head_rating = "🔴 Needs Work"

                # -------------------------
                # Lead Arm Angle Score
                # Higher/wider is generally better
                # -------------------------

                if 130 <= lead_arm <= 170:
                    lead_arm_score = 100
                    lead_arm_rating = "🟢 Excellent"

                elif 120 <= lead_arm <= 180:
                    lead_arm_score = 70
                    lead_arm_rating = "🟡 Good"

                else:       
                    lead_arm_score = 40
                    lead_arm_rating = "🔴 Needs Work"

                # -------------------------
                # Trail Arm Angle Score
                # Ideal range is around 70°-100°
                # -------------------------

                if 70 <= trail_arm <= 100:
                    trail_arm_score = 100
                    trail_arm_rating = "🟢 Excellent"

                elif 60 <= trail_arm <= 110:
                    trail_arm_score = 70
                    trail_arm_rating = "🟡 Good"

                else:
                    trail_arm_score = 40
                    trail_arm_rating = "🔴 Needs Work"

                # -------------------------
                # Spine Angle Change Score
                # Lower is better
                # -------------------------

                if spine_change <= 5:
                    spine_score = 100
                    spine_rating = "🟢 Excellent"

                elif spine_change <= 10:
                    spine_score = 70
                    spine_rating = "🟡 Good"

                else:
                    spine_score = 40
                    spine_rating = "🔴 Needs Work"

                # -------------------------
                # Tempo Score
                # -------------------------

                if 2.7 <= tempo_ratio <= 3.3:

                    tempo_score = 100
                    tempo_rating = "🟢 Excellent"

                elif 2.4 <= tempo_ratio <= 3.6:

                    tempo_score = 70
                    tempo_rating = "🟡 Good"

                else:

                    tempo_score = 40
                    tempo_rating = "🔴 Needs Work"

                # -------------------------
                # Swing Width Score
                # -------------------------

                if swing_width >= 0.18:

                    width_score = 100
                    width_rating = "🟢 Excellent"

                elif swing_width >= 0.15:

                    width_score = 70
                    width_rating = "🟡 Good"

                else:

                    width_score = 40
                    width_rating = "🔴 Needs Work"

                # -------------------------
                # Hip Sway Score
                # Lower is better
                # -------------------------

                if hip_sway <= 0.02:

                    hip_sway_score = 100
                    hip_sway_rating = "🟢 Excellent"

                elif hip_sway <= 0.04:

                    hip_sway_score = 70
                    hip_sway_rating = "🟡 Good"

                else:

                    hip_sway_score = 40
                    hip_sway_rating = "🔴 Needs Work"

                # -------------------------
                # Finish Stability Score
                # Lower is better
                # -------------------------

                if finish_stability <= 0.01:

                    finish_stability_score = 100
                    finish_stability_rating = "🟢 Excellent"

                elif finish_stability <= 0.025:

                    finish_stability_score = 70
                    finish_stability_rating = "🟡 Good"

                else:

                    finish_stability_score = 40
                    finish_stability_rating = "🔴 Needs Work"

                # -------------------------
                # Overall Swing Score
                # -------------------------

                overall_score = round(
                    (
                        head_score
                        + lead_arm_score
                        + trail_arm_score
                        + spine_score
                        + tempo_score
                        + width_score
                        + hip_sway_score
                        + finish_stability_score
                    ) /8
                )

                if overall_score >= 90:
                    overall_band = "🟢 TOUR QUALITY"

                elif overall_score >= 70:
                    overall_band = "🟡 GOOD SWING"

                elif overall_score >= 50:
                    overall_band = "🟠 DEVELOPING"

                else:
                    overall_band = "🔴 MAJOR ISSUES"

                metric_scores = {
                    "Head Stability": head_score,
                    "Lead Arm Width": lead_arm_score,
                    "Trail Arm Position": trail_arm_score,
                    "Posture": spine_score,
                    "Tempo": tempo_score,
                    "Swing Width": width_score,
                    "Hip Sway": hip_sway_score,
                    "Finish Stability": finish_stability_score
                }

                best_metric = max(
                    metric_scores,
                    key=metric_scores.get
                )

                worst_metric = min(
                    metric_scores,
                    key=metric_scores.get
                )

                history_row = {

                    "date": datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    ),

                    "filename": uploaded_file.name,

                    "address": phase_report["address"],
                    "top": phase_report["top"],
                    "impact": phase_report["impact"],
                    "finish": phase_report["finish"],

                    "head_move":
                        metrics[
                            "head_movement_address_to_impact"
                        ],

                    "tempo_ratio":
                        tempo_ratio,

                    "swing_width":
                        swing_width,

                    "hip_sway":
                        hip_sway,

                    "finish_stability":
                        finish_stability
                }

                history_df = pd.read_csv(
                    HISTORY_CSV
                )

                history_df = pd.concat(
                    [
                        history_df,
                        pd.DataFrame([history_row])
                    ],
                    ignore_index=True
                )

                history_df.to_csv(
                    HISTORY_CSV,
                    index=False
                )

                st.subheader("🏌️ Swing Summary")

                summary_col1, summary_col2 = st.columns(2)

                with summary_col1:

                    st.metric(
                        overall_band,
                        f"{overall_score}/100"
                    )

                with summary_col2:

                    st.success(
                        f"✅ Strength: {best_metric} ({metric_scores[best_metric]}/100)"
                    )

                    st.warning(
                        f"⚠ Weakness: {worst_metric} ({metric_scores[worst_metric]}/100)"
                    )

                st.subheader("📊 Swing Analysis Report")

                left_col, right_col = st.columns(2)

                with left_col:

                    with st.container(border=True):

                        st.metric(
                            "Head Movement",
                            round(head_move, 4)
                        )

                        st.caption(
                            "🎯 Target: As close to 0 as possible"
                        )

                        score_bar(head_score)

                        st.markdown(
                            f"{head_rating} ({head_score}/100)"
                        )

                        st.markdown(
                            "<hr style='margin-top:10px; margin-bottom:10px;'>",
                            unsafe_allow_html=True
                        )

                        st.metric(
                            "Trail Arm Angle",
                            round(trail_arm, 2)
                        )

                        st.caption(
                            "🎯 Target: 70°-100°"
                        )

                        score_bar(trail_arm_score)

                        st.markdown(
                            f"{trail_arm_rating} ({trail_arm_score}/100)"
                        )

                        st.markdown(
                            "<hr style='margin-top:10px; margin-bottom:10px;'>",
                            unsafe_allow_html=True
                        )

                        st.metric(
                            "Tempo Ratio",
                            f"{tempo_ratio:.2f}:1"
                        )

                        st.caption(
                            "🎯 Target: Approximately 3:1"
                        )

                        score_bar(tempo_score)

                        st.markdown(
                            f"{tempo_rating} ({tempo_score}/100)"
                        )

                        st.markdown(
                            "<hr style='margin-top:10px; margin-bottom:10px;'>",
                            unsafe_allow_html=True
                        )

                        st.metric(
                            "Hip Sway",
                            round(hip_sway, 4)
                        )

                        st.caption(
                            "🎯 Target: Excellent ≤ 0.02 | Good ≤ 0.04"
                        )

                        score_bar(hip_sway_score)

                        st.markdown(
                            f"{hip_sway_rating} ({hip_sway_score}/100)"
                        )

                with right_col:

                    with st.container(border=True):

                        st.metric(
                            "Lead Arm Angle",
                            round(lead_arm, 2)
                        )

                        st.caption(
                            "🎯 Target: 130°-170°"
                        )

                        score_bar(lead_arm_score)

                        st.markdown(
                            f"{lead_arm_rating} ({lead_arm_score}/100)"
                        )

                        st.markdown(
                            "<hr style='margin-top:10px; margin-bottom:10px;'>",
                            unsafe_allow_html=True
                        )

                        st.metric(
                            "Spine Angle Change",
                            round(spine_change, 2)
                        )

                        st.caption(
                            "🎯 Target: As close to 0° as possible"
                        )

                        score_bar(spine_score)

                        st.markdown(
                            f"{spine_rating} ({spine_score}/100)"
                        )

                        st.markdown(
                            "<hr style='margin-top:10px; margin-bottom:10px;'>",
                            unsafe_allow_html=True
                        )

                        st.metric(
                            "Swing Width",
                            round(swing_width, 3)
                        )

                        st.caption(
                            "🎯 Target: ≥ 0.18"
                        )

                        score_bar(width_score)

                        st.markdown(
                            f"{width_rating} ({width_score}/100)"
                        )

                        st.markdown(
                            "<hr style='margin-top:10px; margin-bottom:10px;'>",
                            unsafe_allow_html=True
                        )

                        st.metric(
                            "Finish Stability",
                            round(finish_stability, 4)
                        )

                        st.caption(
                            "🎯 Target: Excellent ≤ 0.01 | Good ≤ 0.025"
                        )

                        score_bar(finish_stability_score)

                        st.markdown(
                            f"{finish_stability_rating} ({finish_stability_score}/100)"
                        )


            # -------------------------
            # Run AI coach
            # -------------------------

            st.info("Generating coaching feedback...")

            coach_result = subprocess.run(
                [
                    PYTHON_EXE,
                    str(ANALYSIS_DIR / "ai_coach.py")
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            with st.expander("AI coach output"):

                if coach_result.stdout:
                    st.text(coach_result.stdout)

#                if coach_result.stderr:
#                    st.text(coach_result.stderr)

            # -------------------------
            # Save Results Per Video
            # -------------------------

            files_to_copy = [

                "phase_report.json",
                "swing_report.json",
                "coach_report.txt",

                "address.jpg",
                "top.jpg",
                "impact.jpg",
                "finish.jpg"
            ]

            for filename in files_to_copy:

                source_file = (
                    DATA_DIR
                    / filename
                )

                if source_file.exists():

                    shutil.copy2(
                        source_file,
                        report_folder / filename
                    )

            st.success("Analysis complete")


            # -------------------------
            # Coaching Report
            # -------------------------

            if coach_report_path.exists():

                with open(
                    coach_report_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    coach_report = f.read()

                st.subheader("AI Coaching Report")

                st.markdown(
                    coach_report
                )

            else:

                st.warning(
                    "The AI coaching report could not be found."
                )


            # -------------------------
            # Recommended Drill
            # -------------------------

            st.divider()

            st.subheader("🎥 Recommended Drill")

            st.caption(
                f"Selected weakness: {worst_metric}"
            )

            drill_info = DRILL_MAP.get(
                worst_metric
            )

            if drill_info is not None:

                st.markdown(
                    f"### {drill_info['name']}"
                )

                st.write(
                    drill_info["description"]
                )

                drill_video_url = (
                    drill_info["video_url"]
                    .strip()
                )

                if drill_video_url:

                    st.caption(
                        f"Recommended because the lowest-rated "
                        f"metric is {worst_metric}."
                    )

                    st.video(
                        drill_video_url
                    )

                    st.link_button(
                        "▶️ Open drill on YouTube",
                        drill_video_url
                    )

                else:

                    st.info(
                        "A demonstration video has not yet "
                        "been added for this drill."
                    )

            else:

                st.warning(
                    f"No drill is currently mapped to: "
                    f"{worst_metric}"
                )

if page == "Compare Swings":

    st.header("⚖️ Compare Swings")

    if len(report_folders) >= 2:

        compare_1 = st.selectbox(
            "Swing 1",
            report_folders,
            key="compare1"
        )

        compare_2 = st.selectbox(
            "Swing 2",
            report_folders,
            key="compare2"
        )

        compare_button = st.button(
            "Compare Swings"
        )

    else:

        st.info(
            "Analyse at least two swings "
            "to enable comparison."
        )

if page == "Swing History":

    st.header("📈 Swing History")
    with st.expander(
        "🎯 GolfAI Swing Targets",
        expanded=False
    ):

        for target in METRIC_TARGETS.values():
            st.write(target)

    st.success("Swing History page loaded")

    st.write("History CSV location:")
    st.write(HISTORY_CSV)

    history_df = pd.read_csv(HISTORY_CSV)

    st.write("Rows found:")
    st.write(len(history_df))

    st.dataframe(history_df)

    st.metric(
        "Total Swings Analysed",
        len(history_df)
    )

    st.subheader("Head Movement Trend")

    st.caption(
        METRIC_TARGETS["Head Movement"]
    )

    st.line_chart(
        history_df["head_move"]
    )

    st.subheader("Tempo Trend")

    st.caption(
        METRIC_TARGETS["Tempo Ratio"]
    )

    st.line_chart(
        history_df["tempo_ratio"]
    )

    st.subheader("Swing Width Trend")

    st.caption(
        METRIC_TARGETS["Swing Width"]
    )

    st.line_chart(
        history_df["swing_width"]
    )

    st.subheader("Hip Sway Trend")

    st.caption(
        METRIC_TARGETS["Hip Sway"]
    )

    st.line_chart(
        history_df["hip_sway"]
    )

    st.subheader("Finish Stability Trend")

    st.caption(
        METRIC_TARGETS["Finish Stability"]
    )

    st.line_chart(
        history_df["finish_stability"]
    )

    if len(history_df) > 0:

        best_tempo_index = (
            history_df["tempo_ratio"]
            .sub(3.0)
            .abs()
            .idxmin()
        )

        best_tempo = history_df.loc[
            best_tempo_index,
            "tempo_ratio"
        ]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Best Tempo",
            f"{best_tempo:.2f}:1"
        )

        col2.metric(
            "Best Swing Width",
            round(
                history_df["swing_width"].max(),
                4
            )
        )

        col3.metric(
            "Best Hip Sway",
            round(
                history_df["hip_sway"].min(),
                4
            )
        )

        col4.metric(
            "Best Finish Stability",
            round(
                history_df["finish_stability"].min(),
                4
            )
        )

        st.metric(
            "Best Head Stability",
            round(
                history_df["head_move"].min(),
                4
            )
        )

    else:

        st.info(
            "Analyse at least one swing to populate Swing History."
        )

if (
    page == "Compare Swings"
    and len(report_folders) >= 2
    and "compare_button" in locals()
    and compare_button
):

    report_1 = (
        REPORTS_DIR
        / compare_1
        / "swing_report.json"
    )

    report_2 = (
        REPORTS_DIR
        / compare_2
        / "swing_report.json"
    )

    if (
        report_1.exists()
        and report_2.exists()
    ):

        with open(report_1, "r") as f:
            swing_1 = json.load(f)

        with open(report_2, "r") as f:
            swing_2 = json.load(f)

        metrics_1 = swing_1["metrics"]
        metrics_2 = swing_2["metrics"]

        def calculate_swing_score(metrics):

            score = 0

            # Head Movement
            if metrics["head_movement_address_to_impact"] <= 0.02:
                score += 100
            elif metrics["head_movement_address_to_impact"] <= 0.05:
                score += 70
            else:
                score += 40

            # Lead Arm
            if metrics["lead_arm_angle_at_top_degrees"] >= 150:
                score += 100
            elif metrics["lead_arm_angle_at_top_degrees"] >= 130:
                score += 70
            else:
                score += 40

            # Trail Arm
            trail = metrics["trail_arm_angle_at_top_degrees"]

            if 70 <= trail <= 100:
                score += 100
            elif 60 <= trail <= 110:
                score += 70
            else:
                score += 40

            # Spine
            spine = abs(
                metrics["spine_angle_change_degrees"]
            )

            if spine <= 5:
                score += 100
            elif spine <= 10:
                score += 70
            else:
                score += 40

            # Tempo

            tempo = metrics["tempo_ratio"]

            if 2.7 <= tempo <= 3.3:
                score += 100

            elif 2.4 <= tempo <= 3.6:
                score += 70

            else:
                score += 40


            # Swing Width

            width = metrics["swing_width"]

            if width >= 0.18:
                score += 100

            elif width >= 0.15:
                score += 70

            else:
                score += 40


            # Hip Sway

            hip_sway_value = metrics[
                "hip_sway"
            ]

            if hip_sway_value <= 0.02:
                score += 100

            elif hip_sway_value <= 0.04:
                score += 70

            else:
                score += 40


            # Finish Stability

            finish_stability_value = metrics[
                "finish_stability"
            ]

            if finish_stability_value <= 0.01:
                score += 100

            elif finish_stability_value <= 0.025:
                score += 70

            else:
                score += 40


            return round(score / 8)

        score_1 = calculate_swing_score(
            metrics_1
        )

        score_2 = calculate_swing_score(
            metrics_2
        )

        st.divider()

        st.header("Swing Comparison")

        c1, c2 = st.columns(2)

        c1.subheader(compare_1)
        c2.subheader(compare_2)

        st.subheader("🏌️ Swing Score Comparison")

        score_col1, score_col2 = st.columns(2)

        score_col1.metric(
            compare_1,
            f"{score_1}/100"
        )

        score_col2.metric(
            compare_2,
            f"{score_2}/100"
        )

        st.divider()

        address_1 = (
            REPORTS_DIR
            / compare_1
            / "address.jpg"
        )

        address_2 = (
            REPORTS_DIR
            / compare_2
            / "address.jpg"
        )

        top_1 = (
            REPORTS_DIR
            / compare_1
            / "top.jpg"
        )

        top_2 = (
            REPORTS_DIR
            / compare_2
            / "top.jpg"
        )

        impact_1 = (
            REPORTS_DIR
            / compare_1
            / "impact.jpg"
        )

        impact_2 = (
            REPORTS_DIR
            / compare_2
            / "impact.jpg"
        )

        finish_1 = (
            REPORTS_DIR
            / compare_1
            / "finish.jpg"
        )

        finish_2 = (
            REPORTS_DIR
            / compare_2
            / "finish.jpg"
        )

        st.subheader("Address Position")

        left, right = st.columns(2)

        if address_1.exists():
            left.image(
                str(address_1),
                caption=compare_1,
                use_container_width=True
            )

        if address_2.exists():
            right.image(
                str(address_2),
                caption=compare_2,
                use_container_width=True
            )

        st.subheader("Top of Backswing")

        left, right = st.columns(2)

        if top_1.exists():
            left.image(
                str(top_1),
                caption=compare_1,
                use_container_width=True
            )

        if top_2.exists():
            right.image(
                str(top_2),
                caption=compare_2,
                use_container_width=True
            )
        st.subheader("Impact")

        left, right = st.columns(2)

        if impact_1.exists():
            left.image(
                str(impact_1),
                caption=compare_1,
                use_container_width=True
            )

        if impact_2.exists():
            right.image(
                str(impact_2),
                caption=compare_2,
                use_container_width=True
            )

        st.subheader("Finish")

        left, right = st.columns(2)

        if finish_1.exists():
            left.image(
                str(finish_1),
                caption=compare_1,
                use_container_width=True
            )

        if finish_2.exists():
            right.image(
                str(finish_2),
                caption=compare_2,
                use_container_width=True
            )

        st.divider()

        st.subheader(
            "📊 Metrics Comparison"
        )

        comparison_metrics = [

            (
                "Head Movement",
                "head_movement_address_to_impact"
            ),

            (
                "Lead Arm Angle",
                "lead_arm_angle_at_top_degrees"
            ),

            (
                "Trail Arm Angle",
                "trail_arm_angle_at_top_degrees"
            ),

            (
                "Spine Angle Change",
                "spine_angle_change_degrees"
            ),

            (
                "Tempo Ratio",
                "tempo_ratio"
            ),

            (
                "Swing Width",
                "swing_width"
            ),

            (
                "Hip Sway",
                "hip_sway"
            ),

            (
                "Finish Stability",
                "finish_stability"
            )
        ]

        for label, key in comparison_metrics:

            st.markdown(f"### {label}")

            left, right = st.columns(2)

            left.metric(
                compare_1,
                metrics_1[key]
            )

            right.metric(
                compare_2,
                metrics_2[key]
            )

            winner = compare_metric(
                label,
                metrics_1[key],
                metrics_2[key]
            )

            st.success(
                f"🏆 Better Swing: {winner}"
            )

            st.caption(
                METRIC_TARGETS[label]
            )
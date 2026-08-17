import streamlit as st
import subprocess
import sys
import json
from pathlib import Path
import shutil

# -------------------------
# Page setup
# -------------------------

st.set_page_config(
    page_title="GolfAI Swing Analysis",
    page_icon="🏌️",
    layout="wide"
)

st.title("🏌️ GolfAI Swing Analysis")

st.write(
    "Upload a golf swing video, detect the key swing frames, "
    "calculate movement metrics, and generate AI coaching feedback."
)

# -------------------------
# Project paths
# -------------------------

BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "videos"
DATA_DIR = BASE_DIR / "data"
ANALYSIS_DIR = BASE_DIR / "analysis"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

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

st.divider()

st.subheader("Compare Saved Swings")

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

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    st.video(str(video_path))

    analyse_button = st.button(
        "Analyse swing"
    )

    if analyse_button:

        # -------------------------
        # Run phase detector
        # -------------------------

        st.info("Detecting swing phases...")

        phase_result = subprocess.run(
            [
                PYTHON_EXE,
                str(ANALYSIS_DIR / "phase_detector.py"),
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

            if phase_result.stderr:
                st.text(phase_result.stderr)

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

            img1, img2 = st.columns(2)
            img3, img4 = st.columns(2)

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

        with st.expander("Pose detector output"):
            if pose_result.stdout:
                st.text(pose_result.stdout)

            if pose_result.stderr:
                st.text(pose_result.stderr)

        # -------------------------
        # Swing Metrics
        # -------------------------

        if swing_report_path.exists():

            with open(
                swing_report_path,
                "r"
            ) as f:

                swing_report = json.load(f)

            st.subheader("Swing Metrics")

            metrics = swing_report["metrics"]

            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)
            m5, m6 = st.columns(2) 
            m7, _ = st.columns(2)  

            m1.metric(
                "Head Movement",
                metrics[
                    "head_movement_address_to_impact"
                ]
            )

            m2.metric(
                "Lead Arm Angle",
                metrics[
                    "lead_arm_angle_at_top_degrees"
                ]
            )

            m3.metric(
                "Trail Arm Angle",
                metrics[
                    "trail_arm_angle_at_top_degrees"
                ]
            )

            m4.metric(
                "Spine Angle Change",
                metrics[
                    "spine_angle_change_degrees"
                ]
            )

            m5.metric(
                "Shoulder Turn",
                metrics[
                    "shoulder_turn_change_degrees"
                ]
            )

            m6.metric(
                "Hip Turn",
                metrics[
                    "hip_turn_change_degrees"
                ]           
            )

            m7.metric(
                "X-Factor",
                metrics[
                    "x_factor_degrees"
                ]
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

            if coach_result.stderr:
                st.text(coach_result.stderr)

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
                f"<div style='color:white'>{coach_report}</div>",
                unsafe_allow_html=True
            )

if (
    len(report_folders) >= 2
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

        st.divider()

        st.header("Swing Comparison")

        c1, c2 = st.columns(2)

        c1.subheader(compare_1)
        c2.subheader(compare_2)

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
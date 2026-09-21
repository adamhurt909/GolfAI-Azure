import csv
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Configuration
# -------------------------

REFERENCE_FPS = 30.0

# Values calculated from the eight manually labelled swings.
AVERAGE_ADDRESS_TO_TOP_FRAMES = 32.1
AVERAGE_TOP_TO_IMPACT_FRAMES = 9.5
AVERAGE_IMPACT_TO_FINISH_FRAMES = 14.25

# Calibration derived from the eight manually labelled test videos.
# Positive values move the detected phase later in the video.

ADDRESS_CALIBRATION_FRAMES = 4
TOP_CALIBRATION_FRAMES = 2
IMPACT_CALIBRATION_FRAMES = 0
FINISH_CALIBRATION_FRAMES = 0

SHOW_DEBUG_WINDOWS = "--show" in sys.argv


# -------------------------
# Video path
# -------------------------

video_arguments = [
    argument
    for argument in sys.argv[1:]
    if argument != "--show"
]

if video_arguments:
    video_path = Path(video_arguments[0])
else:
    video_path = Path("videos") / "adam_1.mp4"


if not video_path.exists():
    print(f"Video not found: {video_path}")
    sys.exit(1)


# -------------------------
# Output paths
# -------------------------

project_dir = Path(__file__).resolve().parent.parent

output_dir = project_dir / "data"
output_dir.mkdir(
    parents=True,
    exist_ok=True
)

phase_report_path = output_dir / "phase_report.json"

address_image_path = output_dir / "address.jpg"
top_image_path = output_dir / "top.jpg"
impact_image_path = output_dir / "impact.jpg"
finish_image_path = output_dir / "finish.jpg"

manual_phases_path = (
    project_dir
    / "data"
    / "manual_phases.csv"
)


# -------------------------
# Helper functions
# -------------------------

def fill_missing(values):
    values = np.array(
        values,
        dtype=float
    )

    valid_indices = np.where(
        ~np.isnan(values)
    )[0]

    if len(valid_indices) == 0:
        raise ValueError(
            "MediaPipe did not detect any valid wrist positions."
        )

    first_valid = valid_indices[0]

    values[:first_valid] = values[first_valid]

    for index in range(
        first_valid + 1,
        len(values)
    ):
        if np.isnan(values[index]):
            values[index] = values[index - 1]

    return values


def smooth(values, window_size=9):
    values = np.array(
        values,
        dtype=float
    )

    if window_size < 3:
        return values

    padding = window_size // 2

    padded_values = np.pad(
        values,
        (padding, padding),
        mode="edge"
    )

    kernel = (
        np.ones(window_size)
        / window_size
    )

    return np.convolve(
        padded_values,
        kernel,
        mode="valid"
    )


def first_sustained_above(
    values,
    threshold,
    start_frame,
    end_frame,
    minimum_frames=6
):
    start_frame = max(
        0,
        start_frame
    )

    end_frame = min(
        end_frame,
        len(values) - minimum_frames
    )

    for frame_number in range(
        start_frame,
        end_frame
    ):
        movement_window = values[
            frame_number:
            frame_number + minimum_frames
        ]

        if np.all(
            movement_window > threshold
        ):
            return frame_number

    return None


def load_manual_phases(
    csv_path,
    current_video_name
):
    if not csv_path.exists():
        return None

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        for row in reader:
            if (
                row["video"].lower()
                == current_video_name.lower()
            ):
                return {
                    "address": int(
                        row["address"]
                    ),
                    "top": int(
                        row["top"]
                    ),
                    "impact": int(
                        row["impact"]
                    ),
                    "finish": int(
                        row["finish"]
                    )
                }

    return None


def save_labelled_image(
    frame,
    output_path,
    label,
    frame_number
):
    labelled_frame = frame.copy()

    cv2.putText(
        labelled_frame,
        f"{label} - Frame {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imwrite(
        str(output_path),
        labelled_frame
    )


# -------------------------
# Open video
# -------------------------

capture = cv2.VideoCapture(
    str(video_path)
)

if not capture.isOpened():
    print(
        f"Could not open video: {video_path}"
    )
    sys.exit(1)


video_fps = capture.get(
    cv2.CAP_PROP_FPS
)

if video_fps <= 0:
    video_fps = REFERENCE_FPS


# -------------------------
# Extract wrist positions
# -------------------------

mp_pose = mp.solutions.pose

frames = []
hand_x = []
hand_y = []

with mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while capture.isOpened():
        success, frame = capture.read()

        if not success:
            break

        frames.append(
            frame.copy()
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = pose.process(
            rgb_frame
        )

        if results.pose_landmarks:
            landmarks = (
                results
                .pose_landmarks
                .landmark
            )

            left_wrist = landmarks[
                mp_pose.PoseLandmark.LEFT_WRIST
            ]

            right_wrist = landmarks[
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]

            average_x = (
                left_wrist.x
                + right_wrist.x
            ) / 2

            average_y = (
                left_wrist.y
                + right_wrist.y
            ) / 2

            hand_x.append(
                average_x
            )

            hand_y.append(
                average_y
            )

        else:
            hand_x.append(
                np.nan
            )

            hand_y.append(
                np.nan
            )


capture.release()

total_frames = len(frames)

if total_frames == 0:
    print(
        "No frames were loaded from the video."
    )
    sys.exit(1)


# -------------------------
# Prepare wrist data
# -------------------------

hand_x = fill_missing(
    hand_x
)

hand_y = fill_missing(
    hand_y
)

hand_x_smooth = smooth(
    hand_x,
    window_size=9
)

hand_y_smooth = smooth(
    hand_y,
    window_size=9
)

difference_x = np.diff(
    hand_x_smooth,
    prepend=hand_x_smooth[0]
)

difference_y = np.diff(
    hand_y_smooth,
    prepend=hand_y_smooth[0]
)

hand_speed = np.sqrt(
    difference_x ** 2
    + difference_y ** 2
)

hand_speed_smooth = smooth(
    hand_speed,
    window_size=9
)


# -------------------------
# Detect Address / Takeaway
# -------------------------

baseline_frame_count = min(
    total_frames,
    max(
        20,
        int(video_fps * 1.5)
    )
)

baseline_speed = hand_speed_smooth[
    :baseline_frame_count
]

baseline_mean = np.mean(
    baseline_speed
)

baseline_standard_deviation = np.std(
    baseline_speed
)

movement_percentile = np.percentile(
    hand_speed_smooth,
    65
)

takeaway_threshold = max(
    movement_percentile,
    baseline_mean
    + (
        baseline_standard_deviation
        * 2.0
    )
)

takeaway_search_end = min(
    total_frames,
    int(total_frames * 0.65)
)

takeaway_frame = first_sustained_above(
    hand_speed_smooth,
    takeaway_threshold,
    start_frame=0,
    end_frame=takeaway_search_end,
    minimum_frames=6
)

if takeaway_frame is None:
    takeaway_frame = min(
        total_frames - 1,
        int(video_fps * 1.5)
    )

address_offset = max(
    1,
    int(video_fps * 0.5)
)

address_frame = max(
    0,
    takeaway_frame - address_offset
)


# -------------------------
# Detect Top of Backswing
# -------------------------
# The manually labelled swings show that Address to Top
# is normally around 32 frames at 30 FPS.
#
# We therefore search within a restricted window around
# the expected Top position rather than across a large
# portion of the video.

fps_scale = (
    video_fps
    / REFERENCE_FPS
)

expected_address_to_top_frames = max(
    1,
    round(
        AVERAGE_ADDRESS_TO_TOP_FRAMES
        * fps_scale
    )
)

expected_top_frame = min(
    total_frames - 1,
    address_frame
    + expected_address_to_top_frames
)

# Allow a modest margin around the expected position.
# At 30 FPS this is approximately expected Top - 6
# through expected Top + 6.

top_search_margin = max(
    3,
    round(6 * fps_scale)
)

top_search_start = max(
    address_frame + 1,
    expected_top_frame
    - top_search_margin
)

top_search_end = min(
    total_frames,
    expected_top_frame
    + top_search_margin
    + 1
)

search_speed = hand_speed_smooth[
    top_search_start:
    top_search_end
]

search_hand_y = hand_y_smooth[
    top_search_start:
    top_search_end
]

# At the Top:
# 1. The hands are normally high, meaning a smaller Y value.
# 2. Wrist speed normally reduces around the transition.
#
# Normalise both properties and combine them.

speed_range = np.ptp(
    search_speed
)

if speed_range > 0:
    normalised_speed = (
        search_speed
        - np.min(search_speed)
    ) / speed_range
else:
    normalised_speed = np.zeros_like(
        search_speed
    )

hand_y_range = np.ptp(
    search_hand_y
)

if hand_y_range > 0:
    normalised_hand_height = (
        np.max(search_hand_y)
        - search_hand_y
    ) / hand_y_range
else:
    normalised_hand_height = np.zeros_like(
        search_hand_y
    )

# Prefer a high hand position and low movement speed.
top_score = (
    normalised_hand_height * 0.65
    - normalised_speed * 0.35
)

if len(top_score) > 0:
    top_frame = (
        top_search_start
        + int(np.argmax(top_score))
    )
else:
    top_frame = expected_top_frame

top_frame = max(
    address_frame + 1,
    min(
        total_frames - 1,
        top_frame
    )
)

print("")
print("TOP OF BACKSWING DETECTION")
print("--------------------------")
print(
    f"Expected Address to Top: "
    f"{expected_address_to_top_frames} frames"
)
print(
    f"Expected Top frame: "
    f"{expected_top_frame}"
)
print(
    f"Top search window: "
    f"{top_search_start} to {top_search_end - 1}"
)
print(
    f"Detected Top frame: "
    f"{top_frame}"
)
print("")

# -------------------------
# Detect Impact
# -------------------------
# In image coordinates, a larger Y value means that the
# hands are lower on the screen.
#
# Search after Top for the lowest hand position during
# the downswing. The window is bounded to avoid selecting
# a later follow-through or relaxed position.

fps_scale = (
    video_fps
    / REFERENCE_FPS
)

backswing_duration = max(
    1,
    top_frame - address_frame
)

impact_search_start = min(
    total_frames - 1,
    top_frame + max(
        2,
        round(2 * fps_scale)
    )
)

# Allow the downswing window to vary with the detected
# backswing duration instead of assuming exactly 10 frames.

impact_search_length = max(
    round(18 * fps_scale),
    round(backswing_duration * 0.65)
)

impact_search_end = min(
    total_frames,
    impact_search_start
    + impact_search_length
)

impact_hand_y = hand_y_smooth[
    impact_search_start:
    impact_search_end
]

if len(impact_hand_y) > 0:

    # Maximum Y is the lowest hand position on screen.
    impact_frame = (
        impact_search_start
        + int(np.argmax(impact_hand_y))
    )

else:

    impact_frame = min(
        total_frames - 1,
        top_frame
        + round(
            AVERAGE_TOP_TO_IMPACT_FRAMES
            * fps_scale
        )
    )

print("")
print("IMPACT DETECTION")
print("----------------")
print(
    f"Impact search window: "
    f"{impact_search_start} to "
    f"{impact_search_end - 1}"
)
print(
    f"Lowest-hand Impact frame: "
    f"{impact_frame}"
)
print("")

# -------------------------
# Calculate Finish
# -------------------------
# Keep the existing Adam-specific finish timing for now,
# but calculate it from the corrected Impact frame.

impact_to_finish_frames = max(
    1,
    round(
        AVERAGE_IMPACT_TO_FINISH_FRAMES
        * fps_scale
    )
)

finish_frame = min(
    total_frames - 1,
    impact_frame
    + impact_to_finish_frames
)

# -------------------------
# Apply Phase Calibration
# -------------------------

address_frame = min(
    total_frames - 1,
    address_frame
    + round(
        ADDRESS_CALIBRATION_FRAMES
        * fps_scale
    )
)

top_frame = min(
    total_frames - 1,
    top_frame
    + round(
        TOP_CALIBRATION_FRAMES
        * fps_scale
    )
)

impact_frame = min(
    total_frames - 1,
    impact_frame
    + round(
        IMPACT_CALIBRATION_FRAMES
        * fps_scale
    )
)

finish_frame = min(
    total_frames - 1,
    finish_frame
    + round(
        FINISH_CALIBRATION_FRAMES
        * fps_scale
    )
)

# Keep phases in the correct chronological order.

top_frame = max(
    address_frame + 1,
    top_frame
)

impact_frame = max(
    top_frame + 1,
    impact_frame
)

finish_frame = max(
    impact_frame + 1,
    finish_frame
)

finish_frame = min(
    total_frames - 1,
    finish_frame
)

# -------------------------
# Results
# -------------------------

phase_report = {
    "video": video_path.name,
    "fps": round(
        video_fps,
        2
    ),
    "frames_analysed": total_frames,
    "takeaway": int(
        takeaway_frame
    ),
    "address": int(
        address_frame
    ),
    "top": int(
        top_frame
    ),
    "impact": int(
        impact_frame
    ),
    "finish": int(
        finish_frame
    ),
    "timings": {
        "address_to_top": int(
            top_frame - address_frame
        ),
        "top_to_impact": int(
            impact_frame - top_frame
        ),
        "impact_to_finish": int(
            finish_frame - impact_frame
        )
    }
}

with open(
    phase_report_path,
    "w",
    encoding="utf-8"
) as report_file:
    json.dump(
        phase_report,
        report_file,
        indent=4
    )


# -------------------------
# Save V2 phase images
# -------------------------

save_labelled_image(
    frames[address_frame],
    address_image_path,
    "Address",
    address_frame
)

save_labelled_image(
    frames[top_frame],
    top_image_path,
    "Top",
    top_frame
)

save_labelled_image(
    frames[impact_frame],
    impact_image_path,
    "Impact",
    impact_frame
)

save_labelled_image(
    frames[finish_frame],
    finish_image_path,
    "Finish",
    finish_frame
)


# -------------------------
# Print detected results
# -------------------------

print("")
print("PHASE DETECTOR V2")
print("-----------------")
print(f"Video: {video_path.name}")
print(f"FPS: {video_fps:.2f}")
print(f"Frames analysed: {total_frames}")
print(f"Takeaway frame: {takeaway_frame}")
print(f"Address frame: {address_frame}")
print(f"Top frame: {top_frame}")
print(f"Impact frame: {impact_frame}")
print(f"Finish frame: {finish_frame}")
print("")
print("DETECTED TIMINGS")
print("----------------")
print(
    f"Address to Top: "
    f"{top_frame - address_frame}"
)
print(
    f"Top to Impact: "
    f"{impact_frame - top_frame}"
)
print(
    f"Impact to Finish: "
    f"{finish_frame - impact_frame}"
)
print("")
print(
    f"V2 report saved to: "
    f"{phase_report_path}"
)
print(
    f"V2 images saved to: "
    f"{output_dir}"
)


# -------------------------
# Compare with manual labels
# -------------------------

manual_phases = load_manual_phases(
    manual_phases_path,
    video_path.name
)

if manual_phases:
    print("")
    print("MANUAL VS V2 COMPARISON")
    print("-----------------------")

    total_error = 0

    for phase_name in [
        "address",
        "top",
        "impact",
        "finish"
    ]:
        manual_frame = manual_phases[
            phase_name
        ]

        detected_frame = phase_report[
            phase_name
        ]

        frame_error = (
            detected_frame
            - manual_frame
        )

        total_error += abs(
            frame_error
        )

        print(
            f"{phase_name.title():<8} "
            f"Manual: {manual_frame:<4} "
            f"V2: {detected_frame:<4} "
            f"Difference: {frame_error:+d}"
        )

    mean_absolute_error = (
        total_error / 4
    )

    print("")
    print(
        f"Mean absolute error: "
        f"{mean_absolute_error:.2f} frames"
    )

else:
    print("")
    print(
        "No manual phase row was found "
        f"for {video_path.name}."
    )


# -------------------------
# Optional diagnostics
# -------------------------

if SHOW_DEBUG_WINDOWS:
    plt.figure(
        figsize=(12, 7)
    )

    plt.subplot(
        2,
        1,
        1
    )

    plt.plot(
        hand_y_smooth,
        label="Hand height"
    )

    plt.axvline(
        address_frame,
        color="blue",
        linestyle="--",
        label="Address"
    )

    plt.axvline(
        top_frame,
        color="green",
        linestyle="--",
        label="Top"
    )

    plt.axvline(
        impact_frame,
        color="red",
        linestyle="--",
        label="Impact"
    )

    plt.axvline(
        finish_frame,
        color="purple",
        linestyle="--",
        label="Finish"
    )

    plt.title(
        "Phase Detector V2: Hand Height"
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Normalised Y"
    )

    plt.legend()

    plt.subplot(
        2,
        1,
        2
    )

    plt.plot(
        hand_speed_smooth,
        label="Hand speed"
    )

    plt.axvline(
        address_frame,
        color="blue",
        linestyle="--",
        label="Address"
    )

    plt.axvline(
        top_frame,
        color="green",
        linestyle="--",
        label="Top"
    )

    plt.axvline(
        impact_frame,
        color="red",
        linestyle="--",
        label="Impact"
    )

    plt.axvline(
        finish_frame,
        color="purple",
        linestyle="--",
        label="Finish"
    )

    plt.title(
        "Phase Detector V2: Hand Speed"
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Normalised speed"
    )

    plt.legend()

    plt.tight_layout()
    plt.show()
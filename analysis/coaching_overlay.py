import cv2
import json
import math
import shutil
import subprocess
import sys

from pathlib import Path

import mediapipe as mp


# -------------------------
# Paths and arguments
# -------------------------

BASE_DIR = (
    Path(__file__).resolve().parent.parent
)

DATA_DIR = BASE_DIR / "data"

PHASE_REPORT_PATH = (
    DATA_DIR / "phase_report.json"
)

SWING_REPORT_PATH = (
    DATA_DIR / "swing_report.json"
)


if len(sys.argv) < 3:

    print(
        "Usage: python coaching_overlay.py "
        "<input_video> <output_video>"
    )

    sys.exit(1)


input_video_path = Path(sys.argv[1])
output_video_path = Path(sys.argv[2])

output_video_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


if not input_video_path.exists():

    print(
        f"Input video not found: "
        f"{input_video_path}"
    )

    sys.exit(1)


if not PHASE_REPORT_PATH.exists():

    print(
        f"Phase report not found: "
        f"{PHASE_REPORT_PATH}"
    )

    sys.exit(1)


if not SWING_REPORT_PATH.exists():

    print(
        f"Swing report not found: "
        f"{SWING_REPORT_PATH}"
    )

    sys.exit(1)


# -------------------------
# Load GolfAI reports
# -------------------------

with open(
    PHASE_REPORT_PATH,
    "r",
    encoding="utf-8"
) as file:

    phase_report = json.load(file)


with open(
    SWING_REPORT_PATH,
    "r",
    encoding="utf-8"
) as file:

    swing_report = json.load(file)


top_frame = int(
    phase_report["top"]
)

address_frame = int(
    phase_report.get("address", top_frame)
)

impact_frame = int(
    phase_report.get("impact", top_frame)
)

finish_frame = int(
    phase_report.get("finish", top_frame)
)

metrics = swing_report["metrics"]


def metric_value(key, default=0.0):
    """Safely read a numeric metric from the swing report."""

    try:
        return float(metrics[key])
    except (KeyError, TypeError, ValueError):
        return float(default)


current_lead_arm_angle = metric_value(
    "lead_arm_angle_at_top_degrees"
)

current_swing_width = metric_value(
    "swing_width"
)

current_trail_arm_angle = metric_value(
    "trail_arm_angle_at_top_degrees"
)

current_spine_angle_change = metric_value(
    "spine_angle_change_degrees"
)

current_hip_sway = metric_value(
    "hip_sway"
)

current_head_movement = metric_value(
    "head_movement_address_to_impact"
)

current_finish_stability = metric_value(
    "finish_stability"
)

current_tempo_ratio = metric_value(
    "tempo_ratio"
)

worst_metric = swing_report.get(
    "worst_metric",
    "Lead Arm Width"
)

# -------------------------
# Coaching configuration
# -------------------------

TARGET_LEAD_ARM_ANGLE = 150.0

TARGET_SWING_WIDTH = 0.08

USING_SWING_WIDTH = (
    worst_metric == "Swing Width"
)

USING_LEAD_ARM = (
    worst_metric == "Lead Arm Width"
)

USING_POSTURE = (
    worst_metric == "Posture Maintenance"
)

USING_HIP_SWAY = (
    worst_metric == "Hip Sway"
)

USING_HEAD_STABILITY = (
    worst_metric == "Head Stability"
)

USING_TRAIL_ARM = (
    worst_metric == "Trail Arm Position"
)

USING_FINISH_STABILITY = (
    worst_metric == "Finish Stability"
)

USING_TEMPO = (
    worst_metric == "Tempo"
)

# -------------------------------------------------
# Anchor the coaching overlay to the swing phase
# that actually matters for the metric being coached,
# rather than always using the Top of the backswing.
# -------------------------------------------------

if USING_POSTURE or USING_HEAD_STABILITY:

    # These are measured at Impact.
    anchor_frame = impact_frame

elif USING_FINISH_STABILITY:

    # This is measured at the Finish.
    anchor_frame = finish_frame

else:

    # Lead Arm, Trail Arm, Swing Width, Hip Sway and
    # Tempo are all judged at the Top of the backswing.
    anchor_frame = top_frame

# Existing lead arm logic

MAXIMUM_CORRECTION_DEGREES = 25.0

CORRECTION_REQUIRED = (
    current_lead_arm_angle < 145.0
)

requested_correction = max(
    0.0,
    TARGET_LEAD_ARM_ANGLE
    - current_lead_arm_angle
)

applied_correction = min(
    requested_correction,
    MAXIMUM_CORRECTION_DEGREES
)

suggested_lead_arm_angle = min(
    TARGET_LEAD_ARM_ANGLE,
    current_lead_arm_angle
    + applied_correction
)

suggested_lead_arm_angle = min(
    TARGET_LEAD_ARM_ANGLE,
    current_lead_arm_angle
    + applied_correction
)


print("")
print("COACHING OVERLAY")
print("----------------")

print(
    f"Worst metric: {worst_metric}"
)
print(
    f"USING_SWING_WIDTH = "
    f"{USING_SWING_WIDTH}"
)
print(f"Input video: {input_video_path}")
print(f"Output video: {output_video_path}")
print(f"Top frame: {top_frame}")
print(
    f"Current lead-arm angle: "
    f"{current_lead_arm_angle:.2f}"
)
print(
    f"Current swing width: "
    f"{current_swing_width:.3f}"
)
print(
    f"Suggested lead-arm angle: "
    f"{suggested_lead_arm_angle:.2f}"
)
print(
    f"Correction required: "
    f"{CORRECTION_REQUIRED}"
)
print("")


# -------------------------
# Helper functions
# -------------------------

def point_to_pixels(
    point,
    frame_width,
    frame_height
):

    x = int(
        max(
            0,
            min(
                frame_width - 1,
                point.x * frame_width
            )
        )
    )

    y = int(
        max(
            0,
            min(
                frame_height - 1,
                point.y * frame_height
            )
        )
    )

    return x, y


def normalised_to_pixels(
    point,
    frame_width,
    frame_height
):

    x = int(
        max(
            0,
            min(
                frame_width - 1,
                point[0] * frame_width
            )
        )
    )

    y = int(
        max(
            0,
            min(
                frame_height - 1,
                point[1] * frame_height
            )
        )
    )

    return x, y


def distance_2d(point_a, point_b):

    return math.sqrt(
        (
            point_a[0]
            - point_b[0]
        ) ** 2
        +
        (
            point_a[1]
            - point_b[1]
        ) ** 2
    )


def angle_of_vector(vector):

    return math.atan2(
        vector[1],
        vector[0]
    )


def wrap_angle_radians(angle):

    while angle > math.pi:
        angle -= 2 * math.pi

    while angle < -math.pi:
        angle += 2 * math.pi

    return angle


def interpolate_point(
    original_point,
    target_point,
    blend_amount
):

    return (
        original_point[0]
        + (
            target_point[0]
            - original_point[0]
        )
        * blend_amount,

        original_point[1]
        + (
            target_point[1]
            - original_point[1]
        )
        * blend_amount
    )


def correction_strength(
    frame_number,
    target_frame,
    window_frames
):

    distance_from_top = abs(
        frame_number
        - target_frame
    )

    if distance_from_top > window_frames:
        return 0.0

    return (
        1.0
        - (
            distance_from_top
            / window_frames
        )
    )


def calculate_corrected_wrist(
    shoulder,
    elbow,
    wrist,
    target_elbow_angle_degrees
):

    upper_arm_vector = (
        shoulder[0] - elbow[0],
        shoulder[1] - elbow[1]
    )

    forearm_vector = (
        wrist[0] - elbow[0],
        wrist[1] - elbow[1]
    )

    forearm_length = distance_2d(
        elbow,
        wrist
    )

    if forearm_length == 0:
        return wrist

    upper_arm_direction = angle_of_vector(
        upper_arm_vector
    )

    current_forearm_direction = angle_of_vector(
        forearm_vector
    )

    target_angle_radians = math.radians(
        target_elbow_angle_degrees
    )

    candidate_direction_1 = (
        upper_arm_direction
        + target_angle_radians
    )

    candidate_direction_2 = (
        upper_arm_direction
        - target_angle_radians
    )

    candidate_difference_1 = abs(
        wrap_angle_radians(
            candidate_direction_1
            - current_forearm_direction
        )
    )

    candidate_difference_2 = abs(
        wrap_angle_radians(
            candidate_direction_2
            - current_forearm_direction
        )
    )

    if (
        candidate_difference_1
        <= candidate_difference_2
    ):

        selected_direction = (
            candidate_direction_1
        )

    else:

        selected_direction = (
            candidate_direction_2
        )

    corrected_wrist = (
        elbow[0]
        + (
            math.cos(selected_direction)
            * forearm_length
        ),

        elbow[1]
        + (
            math.sin(selected_direction)
            * forearm_length
        )
    )

    return corrected_wrist


def draw_label(
    frame,
    text,
    position,
    colour,
    font_scale=0.65
):

    x, y = position

    text_size, baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        2
    )

    text_width, text_height = text_size

    cv2.rectangle(
        frame,
        (
            x - 8,
            y - text_height - 12
        ),
        (
            x + text_width + 8,
            y + baseline + 6
        ),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        colour,
        2,
        cv2.LINE_AA
    )


# -------------------------
# Open input video
# -------------------------

capture = cv2.VideoCapture(
    str(input_video_path)
)

if not capture.isOpened():

    print(
        f"Could not open video: "
        f"{input_video_path}"
    )

    sys.exit(1)


fps = capture.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30.0


frame_width = int(
    capture.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

frame_height = int(
    capture.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    capture.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


if frame_width <= 0 or frame_height <= 0:

    print(
        "Could not determine the video dimensions."
    )

    capture.release()
    sys.exit(1)


# Apply the correction around the Top.
correction_window_frames = max(
    6,
    round(fps * 0.40)
)


# -------------------------
# Create output video
# -------------------------

# Browsers (and Streamlit's st.video) can only play H.264 in an
# MP4 container. OpenCV's H.264 encoder is available on Windows
# but NOT in opencv-python-headless on Linux (our Azure container),
# where it silently falls back to "mp4v"/"FMP4" -- a file that
# opens in OpenCV but shows a black, unplayable frame in the
# browser.
#
# To behave identically everywhere we:
#   1. Write the overlay to a temporary file with "mp4v", which
#      is always available in OpenCV.
#   2. Transcode that to browser-friendly H.264 with ffmpeg into
#      the final output path (ffmpeg is installed in the image).
#   3. If ffmpeg is unavailable, try OpenCV's own H.264 encoders
#      as a best-effort fallback.
output_fps = fps / 2  # Suggested video plays at half speed.

temp_video_path = output_video_path.with_name(
    output_video_path.stem + "_raw.mp4"
)

writer = cv2.VideoWriter(
    str(temp_video_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    output_fps,
    (
        frame_width,
        frame_height
    )
)


if not writer.isOpened():

    print(
        "Could not create the output video."
    )

    capture.release()
    sys.exit(1)


# -------------------------
# MediaPipe setup
# -------------------------

mp_pose = mp.solutions.pose

frame_number = 0
frames_with_landmarks = 0


with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while capture.isOpened():

        success, frame = capture.read()

        if not success:
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = pose.process(
            rgb_frame
        )

        blend_amount = correction_strength(
            frame_number,
            anchor_frame,
            correction_window_frames
        )

        if (
            results.pose_landmarks
            and blend_amount > 0
        ):

            frames_with_landmarks += 1

            landmarks = (
                results
                .pose_landmarks
                .landmark
            )

            # GolfAI currently assumes:
            # right-handed golfer
            # left arm = lead arm

            left_shoulder_landmark = landmarks[
                mp_pose.PoseLandmark.LEFT_SHOULDER
            ]

            nose_landmark = landmarks[
                mp_pose.PoseLandmark.NOSE
            ]

            right_shoulder_landmark = landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER
            ]

            left_hip_landmark = landmarks[
                mp_pose.PoseLandmark.LEFT_HIP
            ]

            right_hip_landmark = landmarks[
                mp_pose.PoseLandmark.RIGHT_HIP
            ]

            left_elbow_landmark = landmarks[
                mp_pose.PoseLandmark.LEFT_ELBOW
            ]

            left_wrist_landmark = landmarks[
                mp_pose.PoseLandmark.LEFT_WRIST
            ]

            right_elbow_landmark = landmarks[
                mp_pose.PoseLandmark.RIGHT_ELBOW
            ]

            right_wrist_landmark = landmarks[
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]

            shoulder = (
                left_shoulder_landmark.x,
                left_shoulder_landmark.y
            )

            elbow = (
                left_elbow_landmark.x,
                left_elbow_landmark.y
            )

            wrist = (
                left_wrist_landmark.x,
                left_wrist_landmark.y
            )

            suggested_wrist = (
                calculate_corrected_wrist(
                    shoulder,
                    elbow,
                    wrist,
                    suggested_lead_arm_angle
                )
            )

            displayed_suggested_wrist = (
                interpolate_point(
                    wrist,
                    suggested_wrist,
                    blend_amount
                )
            )

            shoulder_pixels = (
                normalised_to_pixels(
                    shoulder,
                    frame_width,
                    frame_height
                )
            )

            elbow_pixels = (
                normalised_to_pixels(
                    elbow,
                    frame_width,
                    frame_height
                )
            )

            wrist_pixels = (
                normalised_to_pixels(
                    wrist,
                    frame_width,
                    frame_height
                )
            )

            nose_pixels = (
                normalised_to_pixels(
                    (
                        nose_landmark.x,
                        nose_landmark.y
                    ),
                    frame_width,
                    frame_height
                )
            )

            right_shoulder_landmark = landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER
            ]

            right_wrist_landmark = landmarks[
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]

            trail_shoulder = (
                right_shoulder_landmark.x,
                right_shoulder_landmark.y
            )

            trail_wrist = (
                right_wrist_landmark.x,
                right_wrist_landmark.y
            )

            trail_elbow = (
                right_elbow_landmark.x,
                right_elbow_landmark.y
            )

            trail_shoulder_pixels = (
                normalised_to_pixels(
                    trail_shoulder,
                    frame_width,
                    frame_height
                )
            )

            trail_wrist_pixels = (
                normalised_to_pixels(
                    trail_wrist,
                    frame_width,
                    frame_height
                )
            )

            trail_elbow_pixels = (
                normalised_to_pixels(
                    trail_elbow,
                    frame_width,
                    frame_height
                )
            )

            shoulder_centre = (
                (
                    left_shoulder_landmark.x
                    + right_shoulder_landmark.x
                ) / 2,

                (
                    left_shoulder_landmark.y
                    + right_shoulder_landmark.y
                ) / 2
            )

            hip_centre = (
                (
                    left_hip_landmark.x
                    + right_hip_landmark.x
                ) / 2,

                (
                    left_hip_landmark.y
                    + right_hip_landmark.y
                ) / 2
            )

            shoulder_centre_pixels = (
                normalised_to_pixels(
                    shoulder_centre,
                    frame_width,
                    frame_height
                )
            )

            hip_centre_pixels = (
                normalised_to_pixels(
                    hip_centre,
                    frame_width,
                    frame_height
                )
            )

            suggested_wrist_pixels = (
                normalised_to_pixels(
                    displayed_suggested_wrist,
                    frame_width,
                    frame_height
                )
            )

            if USING_TEMPO:

                # Tempo is a rhythm metric, not a body position,
                # so it is drawn as an always-on timeline bar
                # below (outside this per-frame landmark block)
                # rather than a pose overlay here.
                pass

            elif USING_FINISH_STABILITY:

                target_head = (
                    nose_pixels[0],
                    nose_pixels[1]
                )

                cv2.circle(
                    frame,
                    nose_pixels,
                    12,
                    (0, 0, 255),
                    -1
                )

                cv2.circle(
                    frame,
                    target_head,
                    12,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    "Hold Finish",
                    (
                        nose_pixels[0] + 20,
                        nose_pixels[1] - 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            elif USING_TRAIL_ARM:

                target_elbow = (
                    trail_elbow_pixels[0] - 30,
                    trail_elbow_pixels[1] - 30
                )

                cv2.line(
                    frame,
                    trail_shoulder_pixels,
                    trail_elbow_pixels,
                    (0, 0, 255),
                    6,
                    cv2.LINE_AA
                )

                cv2.line(
                    frame,
                    trail_elbow_pixels,
                    trail_wrist_pixels,
                    (0, 0, 255),
                    6,
                    cv2.LINE_AA
                )

                cv2.line(
                    frame,
                    trail_shoulder_pixels,
                    target_elbow,
                    (0, 255, 0),
                    6,
                    cv2.LINE_AA
                )

                cv2.line(
                    frame,
                    target_elbow,
                    trail_wrist_pixels,
                    (0, 255, 0),
                    6,
                    cv2.LINE_AA
                )

                cv2.arrowedLine(
                    frame,
                    trail_elbow_pixels,
                    target_elbow,
                    (0, 255, 255),
                    4,
                    cv2.LINE_AA
                )

            elif USING_HEAD_STABILITY:

                target_head = (
                    nose_pixels[0],
                    nose_pixels[1] - 30
                )

                cv2.circle(
                    frame,
                    nose_pixels,
                    12,
                    (0, 0, 255),
                    -1
                )

                cv2.circle(
                    frame,
                    target_head,
                    12,
                    (0, 255, 0),
                    -1
                )

                cv2.arrowedLine(
                    frame,
                    nose_pixels,
                    target_head,
                    (0, 255, 255),
                    5,
                    cv2.LINE_AA
                )

            elif USING_HIP_SWAY:

                target_hip = (
                    hip_centre_pixels[0] - 40,
                    hip_centre_pixels[1]
                )

                cv2.circle(
                    frame,
                    hip_centre_pixels,
                    12,
                    (0, 0, 255),
                    -1
                )

                cv2.circle(
                    frame,
                    target_hip,
                    12,
                    (0, 255, 0),
                    -1
                )

                cv2.arrowedLine(
                    frame,
                    hip_centre_pixels,
                    target_hip,
                    (0, 255, 255),
                    5,
                    cv2.LINE_AA
                )

            elif USING_POSTURE:

                # Current spine (hip -> shoulders) in red.
                cv2.line(
                    frame,
                    shoulder_centre_pixels,
                    hip_centre_pixels,
                    (0, 0, 255),
                    8,
                    cv2.LINE_AA
                )

                # Rotate the spine back toward its address
                # angle by the measured spine-angle change,
                # so the green target shows the posture the
                # golfer should maintain through impact.
                spine_vector_x = (
                    shoulder_centre_pixels[0]
                    - hip_centre_pixels[0]
                )

                spine_vector_y = (
                    shoulder_centre_pixels[1]
                    - hip_centre_pixels[1]
                )

                spine_screen_angle = math.degrees(
                    math.atan2(
                        spine_vector_x,
                        -spine_vector_y
                    )
                )

                # Reduce the forward tilt by the amount it
                # drifted from address, scaled by how close
                # we are to the anchor frame.
                correction_degrees = (
                    current_spine_angle_change
                    * blend_amount
                )

                desired_angle = math.radians(
                    spine_screen_angle
                    - correction_degrees
                )

                spine_pixel_length = math.hypot(
                    spine_vector_x,
                    spine_vector_y
                )

                target_shoulder = (
                    int(
                        hip_centre_pixels[0]
                        + spine_pixel_length
                        * math.sin(desired_angle)
                    ),
                    int(
                        hip_centre_pixels[1]
                        - spine_pixel_length
                        * math.cos(desired_angle)
                    )
                )

                cv2.line(
                    frame,
                    target_shoulder,
                    hip_centre_pixels,
                    (0, 255, 0),
                    8,
                    cv2.LINE_AA
                )

                cv2.arrowedLine(
                    frame,
                    shoulder_centre_pixels,
                    target_shoulder,
                    (0, 255, 255),
                    4,
                    cv2.LINE_AA
                )

            elif USING_SWING_WIDTH:

                target_width_pixels = 40

                target_wrist_x = (
                    trail_wrist_pixels[0]
                    + target_width_pixels
                )

                target_wrist_y = (
                    trail_wrist_pixels[1]
                )

                cv2.line(
                    frame,
                    trail_shoulder_pixels,
                    trail_wrist_pixels,
                    (0, 0, 255),
                    6,
                    cv2.LINE_AA
                )

                cv2.line(
                    frame,
                    trail_shoulder_pixels,
                    (
                        target_wrist_x,
                        target_wrist_y
                    ),
                    (0, 255, 0),
                    6,
                    cv2.LINE_AA
                )

                cv2.arrowedLine(
                    frame,
                    trail_wrist_pixels,
                    (
                        target_wrist_x,
                        target_wrist_y
                    ),
                    (0, 255, 255),
                    4,
                    cv2.LINE_AA,
                    tipLength=0.25
                )

            else:

                # Current detected arm: red
                cv2.line(
                    frame,
                    shoulder_pixels,
                    elbow_pixels,
                    (0, 0, 255),
                    5,
                    cv2.LINE_AA
                )

                cv2.line(
                    frame,
                    elbow_pixels,
                    wrist_pixels,
                    (0, 0, 255),
                    5,
                    cv2.LINE_AA
                )

                # Suggested lead arm: green
                cv2.line(
                    frame,
                    shoulder_pixels,
                    elbow_pixels,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA
                )

                cv2.line(
                    frame,
                    elbow_pixels,
                    suggested_wrist_pixels,
                    (0, 255, 0),
                    5,
                    cv2.LINE_AA
                )

                # Movement arrow from current wrist
                # to suggested wrist.
                if CORRECTION_REQUIRED:

                    cv2.arrowedLine(
                        frame,
                        wrist_pixels,
                        suggested_wrist_pixels,
                        (0, 255, 255),
                        3,
                        cv2.LINE_AA,
                        tipLength=0.20
                    )

                for point in [
                    shoulder_pixels,
                    elbow_pixels,
                    wrist_pixels
                ]:

                    cv2.circle(
                        frame,
                        point,
                        7,
                        (0, 0, 255),
                        -1,
                        cv2.LINE_AA
                    )

                cv2.circle(
                    frame,
                    suggested_wrist_pixels,
                    8,
                    (0, 255, 0),
                    -1,
                    cv2.LINE_AA
                )

        # -------------------------
        # On-screen coaching panel
        # -------------------------

        panel_height = 145

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (
                frame_width,
                panel_height
            ),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.68,
            frame,
            0.32,
            0,
            frame
        )

        if USING_TEMPO:

            overlay_title = (
                "GolfAI Suggested Tempo"
            )

        elif USING_FINISH_STABILITY:

            overlay_title = (
                "GolfAI Suggested Finish Stability"
            )

        elif USING_TRAIL_ARM:

            overlay_title = (
                "GolfAI Suggested Trail Arm Position"
            )

        elif USING_HEAD_STABILITY:

            overlay_title = (
                "GolfAI Suggested Head Stability"
            )

        elif USING_HIP_SWAY:

            overlay_title = (
                "GolfAI Suggested Hip Position"
            )

        elif USING_POSTURE:

            overlay_title = (
                "GolfAI Suggested Posture Maintenance"
            )

        elif USING_SWING_WIDTH:

            overlay_title = (
                "GolfAI Suggested Swing Width"
            )

        else:

            overlay_title = (
                "GolfAI Suggested Lead-Arm Position"
            )

        draw_label(
            frame,
            overlay_title,
            (20, 34),
            (255, 255, 255),
            font_scale=0.75
        )

        # -------------------------------------------------
        # Panel body: current value + target for the metric
        # being coached. Each metric shows its own numbers
        # and units rather than defaulting to lead-arm text.
        # -------------------------------------------------

        if USING_SWING_WIDTH:

            current_line = (
                f"Current Width: {current_swing_width:.3f}"
            )

            target_line = (
                f"Target Width: {TARGET_SWING_WIDTH:.3f}"
            )

        elif USING_TRAIL_ARM:

            current_line = (
                f"Current Trail Arm: "
                f"{current_trail_arm_angle:.1f} degrees"
            )

            target_line = "Target: 80-90 degrees"

        elif USING_POSTURE:

            current_line = (
                f"Spine moved: "
                f"{abs(current_spine_angle_change):.1f} degrees"
            )

            target_line = "Target: 0-8 degrees"

        elif USING_HIP_SWAY:

            current_line = (
                f"Current Hip Sway: {current_hip_sway:.3f}"
            )

            target_line = "Target: <= 0.15"

        elif USING_HEAD_STABILITY:

            current_line = (
                f"Head Movement: {current_head_movement:.3f}"
            )

            target_line = "Target: <= 0.02"

        elif USING_FINISH_STABILITY:

            current_line = (
                f"Finish Movement: {current_finish_stability:.3f}"
            )

            target_line = "Target: hold a stable finish"

        elif USING_TEMPO:

            current_line = (
                f"Current Tempo: {current_tempo_ratio:.1f} : 1"
            )

            target_line = "Target: 3.0 : 1"

        else:

            # Lead Arm Width (default)
            current_line = (
                f"Current at Top: "
                f"{current_lead_arm_angle:.1f} degrees"
            )

            target_line = (
                f"Suggested target: "
                f"{suggested_lead_arm_angle:.1f} degrees"
            )

        cv2.putText(
            frame,
            current_line,
            (20, 73),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            target_line,
            (20, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        if USING_TEMPO:

            guidance_text = (
                "Target rhythm = 3:1 "
            )

        elif USING_FINISH_STABILITY:

            guidance_text = (
                "Red = finish position | "
                "Green = stable finish | "
                "Hold your finish longer"
            )

        elif USING_TRAIL_ARM:

            guidance_text = (
                "Red = current trail arm | "
                "Green = target trail arm | "
                "Yellow = improve elbow position"
            )

        elif USING_HEAD_STABILITY:

            guidance_text = (
                "Red = head position | "
                "Green = target head position | "
                "Yellow = reduce movement"
            )

        elif USING_HIP_SWAY:

            guidance_text = (
                "Red = current hips | "
                "Green = target hips | "
                "Yellow = reduce sway"
            )

        elif USING_SWING_WIDTH:

            guidance_text = (
                "Red = current width | "
                "Green = target width | "
                "Yellow = move hands wider"
            )

        elif USING_POSTURE:

            guidance_text = (
                "Red = current spine | "
                "Green = address spine | "
                "Yellow = hold your posture"
            )

        elif USING_LEAD_ARM and CORRECTION_REQUIRED:

            guidance_text = (
                "Red = current | Green = suggested"
            )

        elif USING_LEAD_ARM:

            guidance_text = (
                "Lead arm is already within target range"
            )

        else:

            guidance_text = (
                "Red = current | Green = suggested"
            )

        cv2.putText(
            frame,
            guidance_text,
            (20, 133),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # -------------------------------------------------
        # Tempo timeline bar (always on for the Tempo metric).
        #
        # Tempo has no single body position to annotate, so we
        # show the rhythm visually: a bar split into the actual
        # backswing and downswing durations, a target bar showing
        # the ideal 3:1 split, and a playhead marking where the
        # current frame sits in the swing. This makes it obvious
        # which half of the swing is out of proportion.
        # -------------------------------------------------

        if USING_TEMPO:

            backswing_len = max(1, top_frame - address_frame)
            downswing_len = max(1, impact_frame - top_frame)
            total_len = backswing_len + downswing_len

            bar_left = 40
            bar_right = frame_width - 40
            bar_width = bar_right - bar_left

            current_bar_y = int(frame_height * 0.62)
            target_bar_y = current_bar_y + 54
            bar_thickness = 26

            # Where the Top sits along the CURRENT bar
            # (proportional to real durations).
            current_split_x = int(
                bar_left
                + bar_width * (backswing_len / total_len)
            )

            # Backswing portion (blue) and downswing portion
            # (orange) of the golfer's actual swing.
            cv2.rectangle(
                frame,
                (bar_left, current_bar_y),
                (current_split_x, current_bar_y + bar_thickness),
                (255, 160, 0),
                -1
            )

            cv2.rectangle(
                frame,
                (current_split_x, current_bar_y),
                (bar_right, current_bar_y + bar_thickness),
                (0, 140, 255),
                -1
            )

            # Target bar: ideal 3:1 split (3/4 backswing).
            target_split_x = int(
                bar_left + bar_width * (3.0 / 4.0)
            )

            cv2.rectangle(
                frame,
                (bar_left, target_bar_y),
                (target_split_x, target_bar_y + bar_thickness),
                (0, 200, 0),
                -1
            )

            cv2.rectangle(
                frame,
                (target_split_x, target_bar_y),
                (bar_right, target_bar_y + bar_thickness),
                (0, 90, 0),
                -1
            )

            # Dashed guide so you can compare where your Top is
            # versus where the target Top should be.
            cv2.line(
                frame,
                (target_split_x, current_bar_y - 6),
                (target_split_x, target_bar_y + bar_thickness + 6),
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Playhead: where the current frame is in the swing.
            if total_len > 0:
                progress = (
                    (frame_number - address_frame) / total_len
                )
                progress = max(0.0, min(1.0, progress))
                playhead_x = int(bar_left + bar_width * progress)

                cv2.line(
                    frame,
                    (playhead_x, current_bar_y - 10),
                    (playhead_x, current_bar_y + bar_thickness + 10),
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA
                )

            # Labels.
            cv2.putText(
                frame,
                f"Your tempo: {current_tempo_ratio:.1f} : 1",
                (bar_left, current_bar_y - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                "Target: 3.0 : 1 (longer, smoother backswing)",
                (bar_left, target_bar_y + bar_thickness + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        # Highlight the exact anchor frame for this metric.
        if frame_number == anchor_frame:

            if anchor_frame == impact_frame:
                phase_label = "IMPACT"
            elif anchor_frame == finish_frame:
                phase_label = "FINISH"
            else:
                phase_label = "TOP OF BACKSWING"

            cv2.putText(
                frame,
                phase_label,
                (
                    20,
                    frame_height - 30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.90,
                (0, 255, 255),
                3,
                cv2.LINE_AA
            )

        for _ in range(2):

            writer.write(frame)

        frame_number += 1


capture.release()
writer.release()


# -------------------------
# Transcode to browser-friendly H.264
# -------------------------
# OpenCV wrote an "mp4v" file (always available). Browsers need
# H.264, so transcode with ffmpeg if present. If ffmpeg is not
# available, fall back to re-encoding with OpenCV's own H.264
# codecs, and only as a last resort keep the mp4v file.

def transcode_to_h264(source_path, destination_path):
    """Transcode a video to H.264/MP4 using ffmpeg.

    Returns True on success. yuv420p + faststart make the file
    playable in browsers and seekable from the first byte.
    """

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        return False

    command = [
        ffmpeg,
        "-y",
        "-i", str(source_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(destination_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    return (
        result.returncode == 0
        and destination_path.exists()
        and destination_path.stat().st_size > 0
    )


def reencode_with_opencv_h264(source_path, destination_path):
    """Best-effort re-encode to H.264 using OpenCV directly."""

    reader = cv2.VideoCapture(str(source_path))

    if not reader.isOpened():
        return False

    source_fps = reader.get(cv2.CAP_PROP_FPS) or output_fps
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_writer = None

    for codec_name in ("avc1", "H264", "h264"):

        candidate = cv2.VideoWriter(
            str(destination_path),
            cv2.VideoWriter_fourcc(*codec_name),
            source_fps,
            (width, height)
        )

        if candidate.isOpened():
            out_writer = candidate
            break

        candidate.release()

    if out_writer is None:
        reader.release()
        return False

    while True:
        ok, frame = reader.read()
        if not ok:
            break
        out_writer.write(frame)

    reader.release()
    out_writer.release()

    return (
        destination_path.exists()
        and destination_path.stat().st_size > 0
    )


transcoded = transcode_to_h264(temp_video_path, output_video_path)

if transcoded:
    print("Transcoded to H.264 with ffmpeg.")
    temp_video_path.unlink(missing_ok=True)

elif reencode_with_opencv_h264(temp_video_path, output_video_path):
    print("Re-encoded to H.264 with OpenCV.")
    temp_video_path.unlink(missing_ok=True)

else:
    # Last resort: keep the mp4v file so there is still a video.
    # It may not play in all browsers, but nothing is lost.
    print(
        "WARNING: H.264 transcode unavailable; "
        "keeping mp4v output (may not play in browser)."
    )
    if temp_video_path.exists():
        shutil.move(str(temp_video_path), str(output_video_path))


# -------------------------
# Final validation
# -------------------------

if (
    output_video_path.exists()
    and output_video_path.stat().st_size > 0
):

    print("Coaching overlay created successfully.")
    print(f"Frames written: {frame_number}")
    print(
        f"Frames with coaching landmarks: "
        f"{frames_with_landmarks}"
    )
    print(
        f"Output size: "
        f"{output_video_path.stat().st_size} bytes"
    )
    print(
        f"Saved to: "
        f"{output_video_path}"
    )

else:

    print(
        "The coaching overlay file was not created."
    )

    sys.exit(1)
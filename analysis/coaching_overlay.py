import cv2
import json
import math
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

current_lead_arm_angle = float(
    swing_report["metrics"][
        "lead_arm_angle_at_top_degrees"
    ]
)


# -------------------------
# Coaching configuration
# -------------------------

TARGET_LEAD_ARM_ANGLE = 150.0

# Do not exaggerate the correction in the first version.
MAXIMUM_CORRECTION_DEGREES = 25.0

# Only suggest straightening when the detected arm is
# below the bottom of GolfAI's ideal range.
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


print("")
print("COACHING OVERLAY")
print("----------------")
print(f"Input video: {input_video_path}")
print(f"Output video: {output_video_path}")
print(f"Top frame: {top_frame}")
print(
    f"Current lead-arm angle: "
    f"{current_lead_arm_angle:.2f}"
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

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(output_video_path),
    fourcc,
    fps,
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
            top_frame,
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

            left_elbow_landmark = landmarks[
                mp_pose.PoseLandmark.LEFT_ELBOW
            ]

            left_wrist_landmark = landmarks[
                mp_pose.PoseLandmark.LEFT_WRIST
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

            suggested_wrist_pixels = (
                normalised_to_pixels(
                    displayed_suggested_wrist,
                    frame_width,
                    frame_height
                )
            )

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

        draw_label(
            frame,
            "GolfAI Suggested Lead-Arm Position",
            (20, 34),
            (255, 255, 255),
            font_scale=0.75
        )

        cv2.putText(
            frame,
            (
                f"Current at Top: "
                f"{current_lead_arm_angle:.1f} degrees"
            ),
            (20, 73),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            (
                f"Suggested target: "
                f"{suggested_lead_arm_angle:.1f} degrees"
            ),
            (20, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        if CORRECTION_REQUIRED:

            guidance_text = (
                "Red = current | Green = suggested"
            )

        else:

            guidance_text = (
                "Lead arm is already within "
                "GolfAI's target range"
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

        # Highlight the exact Top frame.
        if frame_number == top_frame:

            cv2.putText(
                frame,
                "TOP OF BACKSWING",
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

        writer.write(frame)

        frame_number += 1


capture.release()
writer.release()


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
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

LANDMARKS_PATH = (
    DATA_DIR / "landmarks.json"
)


# MediaPipe BlazePose landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24


if len(sys.argv) < 2:

    print(
        "Usage: python comparison_overlay.py "
        "<output_image>"
    )

    sys.exit(1)


output_image_path = Path(sys.argv[1])

output_image_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


# -------------------------
# Load reports
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


metrics = swing_report["metrics"]

worst_metric = swing_report.get(
    "worst_metric",
    "Swing Width"
)

worst_metric_score = swing_report.get(
    "worst_metric_score",
    0
)


# -------------------------
# Pose detection helpers
# -------------------------
#
# Overlays must line up with the golfer in the exact
# phase image we annotate. The most reliable way to
# guarantee that is to run MediaPipe directly on the
# source image, rather than trusting a separately
# generated landmarks file that may not match this frame.


def detect_pose_on_image(image):
    """Run MediaPipe Pose on a single BGR image.

    Returns the landmark list (normalised) or None if no
    pose was detected.
    """

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose:

        results = pose.process(rgb_image)

    if not results.pose_landmarks:
        return None

    return results.pose_landmarks.landmark


def landmark_to_pixels(
    landmark,
    frame_width,
    frame_height
):
    """Convert a normalised landmark to clamped pixel coordinates."""

    x = int(
        max(
            0,
            min(
                frame_width - 1,
                landmark.x * frame_width
            )
        )
    )

    y = int(
        max(
            0,
            min(
                frame_height - 1,
                landmark.y * frame_height
            )
        )
    )

    return x, y


def midpoint_pixels(
    landmark_a,
    landmark_b,
    frame_width,
    frame_height
):
    """Pixel midpoint of two normalised landmarks."""

    mid_x = (landmark_a.x + landmark_b.x) / 2.0
    mid_y = (landmark_a.y + landmark_b.y) / 2.0

    x = int(max(0, min(frame_width - 1, mid_x * frame_width)))
    y = int(max(0, min(frame_height - 1, mid_y * frame_height)))

    return x, y


def draw_arm_correction(
    panel,
    landmarks,
    shoulder_id,
    elbow_id,
    wrist_id,
    target_elbow_degrees
):
    """Draw an arm with a suggested elbow angle.

    Red = the golfer's current arm (shoulder-elbow-wrist).
    Green = the same upper arm with the forearm rotated about
    the elbow so the elbow angle becomes target_elbow_degrees.
    Yellow arrow = how the wrist should move. Works for both the
    lead arm (straighter target) and trail arm (folded target).
    """

    shoulder_px = landmark_to_pixels(
        landmarks[shoulder_id],
        image_width,
        image_height
    )

    elbow_px = landmark_to_pixels(
        landmarks[elbow_id],
        image_width,
        image_height
    )

    wrist_px = landmark_to_pixels(
        landmarks[wrist_id],
        image_width,
        image_height
    )

    # Current arm (red).
    cv2.line(panel, shoulder_px, elbow_px, (0, 0, 255), 6, cv2.LINE_AA)
    cv2.line(panel, elbow_px, wrist_px, (0, 0, 255), 6, cv2.LINE_AA)

    upper_x = shoulder_px[0] - elbow_px[0]
    upper_y = shoulder_px[1] - elbow_px[1]

    fore_x = wrist_px[0] - elbow_px[0]
    fore_y = wrist_px[1] - elbow_px[1]

    forearm_length = math.hypot(fore_x, fore_y)

    if forearm_length > 1:

        upper_direction = math.atan2(upper_y, upper_x)
        current_fore_direction = math.atan2(fore_y, fore_x)

        target_radians = math.radians(target_elbow_degrees)

        option_a = upper_direction + target_radians
        option_b = upper_direction - target_radians

        def ang_diff(a, b):
            d = a - b
            while d > math.pi:
                d -= 2 * math.pi
            while d < -math.pi:
                d += 2 * math.pi
            return abs(d)

        if (
            ang_diff(option_a, current_fore_direction)
            <= ang_diff(option_b, current_fore_direction)
        ):
            chosen_direction = option_a
        else:
            chosen_direction = option_b

        target_wrist_px = (
            int(elbow_px[0] + forearm_length * math.cos(chosen_direction)),
            int(elbow_px[1] + forearm_length * math.sin(chosen_direction))
        )

        # Suggested arm (green): same upper arm, adjusted forearm.
        cv2.line(panel, shoulder_px, elbow_px, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.line(panel, elbow_px, target_wrist_px, (0, 255, 0), 6, cv2.LINE_AA)

        cv2.arrowedLine(
            panel,
            wrist_px,
            target_wrist_px,
            (0, 255, 255),
            4,
            cv2.LINE_AA,
            tipLength=0.25
        )

    # Joint markers.
    for point in (shoulder_px, elbow_px, wrist_px):
        cv2.circle(panel, point, 7, (0, 0, 255), -1, cv2.LINE_AA)


# -------------------------
# Select relevant phase image
# -------------------------

PHASE_IMAGE_MAP = {

    "Lead Arm Width": "top.jpg",
    "Trail Arm Position": "top.jpg",
    "Swing Width": "top.jpg",

    "Posture Maintenance": "impact.jpg",
    "Head Stability": "impact.jpg",

    "Hip Sway": "top.jpg",

    "Finish Stability": "finish.jpg",

    "Tempo": "top.jpg"
}


phase_label_map = {

    "Lead Arm Width": "Top of Backswing",
    "Trail Arm Position": "Top of Backswing",
    "Swing Width": "Top of Backswing",

    "Posture Maintenance": "Impact",
    "Head Stability": "Impact",

    "Hip Sway": "Top of Backswing",

    "Finish Stability": "Finish",

    "Tempo": "Swing Timing"
}


# Frame index that each metric's phase image corresponds to.
PHASE_FRAME_MAP = {

    "Lead Arm Width": "top",
    "Trail Arm Position": "top",
    "Swing Width": "top",

    "Posture Maintenance": "impact",
    "Head Stability": "impact",

    "Hip Sway": "top",

    "Finish Stability": "finish",

    "Tempo": "top"
}


image_name = PHASE_IMAGE_MAP.get(
    worst_metric,
    "top.jpg"
)

phase_label = phase_label_map.get(
    worst_metric,
    "Relevant Swing Phase"
)

source_image_path = (
    DATA_DIR
    / image_name
)


if not source_image_path.exists():

    print(
        f"Comparison source image not found: "
        f"{source_image_path}"
    )

    sys.exit(1)


source_image = cv2.imread(
    str(source_image_path)
)


if source_image is None:

    print(
        "Could not read the comparison source image."
    )

    sys.exit(1)


# -------------------------
# Build panels
# -------------------------

current_panel = source_image.copy()
suggested_panel = source_image.copy()

image_height, image_width = (
    source_image.shape[:2]
)


# Dark panel at top of each image

for panel in [
    current_panel,
    suggested_panel
]:

    panel_overlay = panel.copy()

    cv2.rectangle(
        panel_overlay,
        (0, 0),
        (
            image_width,
            85
        ),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        panel_overlay,
        0.68,
        panel,
        0.32,
        0,
        panel
    )


cv2.putText(
    current_panel,
    "CURRENT POSITION",
    (20, 38),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.85,
    (0, 0, 255),
    2,
    cv2.LINE_AA
)

cv2.putText(
    current_panel,
    phase_label,
    (20, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.60,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)


cv2.putText(
    suggested_panel,
    "SUGGESTED POSITION",
    (20, 38),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.85,
    (0, 255, 0),
    2,
    cv2.LINE_AA
)

cv2.putText(
    suggested_panel,
    phase_label,
    (20, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.60,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)


# -------------------------
# Add initial metric-specific annotation
# -------------------------

if worst_metric == "Swing Width":

    current_width = float(
        metrics["swing_width"]
    )

    target_width = 0.08

    cv2.putText(
        current_panel,
        f"Width: {current_width:.3f}",
        (
            20,
            image_height - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        suggested_panel,
        f"Target: {target_width:.3f}",
        (
            20,
            image_height - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # -------------------------------------------------
    # Swing width = how far the trail hand is extended
    # from the trail shoulder at the top (a wider arc
    # stores more power). Red = current hand position,
    # green = a wider target, yellow arrow = extend the
    # hands further from the body.
    # -------------------------------------------------

    width_landmarks = detect_pose_on_image(source_image)

    if width_landmarks is not None:

        shoulder_px = landmark_to_pixels(
            width_landmarks[RIGHT_SHOULDER],
            image_width,
            image_height
        )

        wrist_px = landmark_to_pixels(
            width_landmarks[RIGHT_WRIST],
            image_width,
            image_height
        )

        # Current width line (red).
        cv2.line(
            suggested_panel,
            shoulder_px,
            wrist_px,
            (0, 0, 255),
            6,
            cv2.LINE_AA
        )

        # Extend the hand further from the shoulder (same
        # direction, longer) to show a wider arc.
        dx = wrist_px[0] - shoulder_px[0]
        dy = wrist_px[1] - shoulder_px[1]
        length = math.hypot(dx, dy)

        if length > 1:
            scale = 1.35
            target_wrist_px = (
                int(shoulder_px[0] + dx * scale),
                int(shoulder_px[1] + dy * scale)
            )

            cv2.line(
                suggested_panel,
                shoulder_px,
                target_wrist_px,
                (0, 255, 0),
                4,
                cv2.LINE_AA
            )

            cv2.arrowedLine(
                suggested_panel,
                wrist_px,
                target_wrist_px,
                (0, 255, 255),
                4,
                cv2.LINE_AA,
                tipLength=0.3
            )

        cv2.circle(
            suggested_panel, shoulder_px, 7, (0, 0, 255), -1, cv2.LINE_AA
        )
        cv2.circle(
            suggested_panel, wrist_px, 7, (0, 0, 255), -1, cv2.LINE_AA
        )


elif worst_metric == "Lead Arm Width":

    current_angle = float(
        metrics[
            "lead_arm_angle_at_top_degrees"
        ]
    )

    cv2.putText(
        current_panel,
        f"Angle: {current_angle:.1f} degrees",
        (
            20,
            image_height - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        suggested_panel,
        "Target: 145-160 degrees",
        (
            20,
            image_height - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # -------------------------------------------------
    # Draw the lead (left) arm: red = current, green =
    # suggested with the arm straightened toward the
    # target (~150 deg) for more width, yellow arrow
    # showing how the wrist should extend.
    # -------------------------------------------------

    lead_landmarks = detect_pose_on_image(source_image)

    if lead_landmarks is not None:
        draw_arm_correction(
            suggested_panel,
            lead_landmarks,
            LEFT_SHOULDER,
            LEFT_ELBOW,
            LEFT_WRIST,
            target_elbow_degrees=150.0
        )


elif worst_metric == "Trail Arm Position":

    current_angle = float(
        metrics[
            "trail_arm_angle_at_top_degrees"
        ]
    )

    cv2.putText(
        current_panel,
        f"Angle: {current_angle:.1f} degrees",
        (
            20,
            image_height - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        suggested_panel,
        "Target: 80-90 degrees",
        (
            20,
            image_height - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # -------------------------------------------------
    # Draw the trail (right) arm on the body using real
    # landmarks: red = current, green = suggested with the
    # elbow folded toward the target angle, yellow arrow
    # showing how the wrist should move.
    # -------------------------------------------------

    trail_landmarks = detect_pose_on_image(source_image)

    if trail_landmarks is not None:
        draw_arm_correction(
            suggested_panel,
            trail_landmarks,
            RIGHT_SHOULDER,
            RIGHT_ELBOW,
            RIGHT_WRIST,
            target_elbow_degrees=85.0
        )

elif worst_metric == "Posture Maintenance":

    spine_at_impact = float(
        metrics.get(
            "spine_angle_at_impact_degrees",
            0.0
        )
    )

    spine_at_address = float(
        metrics.get(
            "spine_angle_at_address_degrees",
            spine_at_impact
        )
    )

    spine_change = abs(
        float(
            metrics.get(
                "spine_angle_change_degrees",
                abs(spine_at_impact - spine_at_address)
            )
        )
    )

    cv2.putText(
        current_panel,
        f"Spine moved {spine_change:.0f} deg",
        (
            20,
            image_height - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        suggested_panel,
        "Hold address spine angle",
        (
            20,
            image_height - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # -------------------------------------------------
    # Draw the spine using the golfer's REAL landmarks
    # at the impact frame, instead of fixed screen
    # percentages. This keeps the overlay locked to the
    # body regardless of how the video is framed.
    # -------------------------------------------------

    posture_landmarks = detect_pose_on_image(source_image)

    drew_real_spine = False

    if posture_landmarks is not None:

        shoulder_mid = midpoint_pixels(
            posture_landmarks[LEFT_SHOULDER],
            posture_landmarks[RIGHT_SHOULDER],
            image_width,
            image_height
        )

        hip_mid = midpoint_pixels(
            posture_landmarks[LEFT_HIP],
            posture_landmarks[RIGHT_HIP],
            image_width,
            image_height
        )

        spine_length = math.hypot(
            shoulder_mid[0] - hip_mid[0],
            shoulder_mid[1] - hip_mid[1]
        )

        if spine_length > 1:

            # The spine angles are measured as the tilt from
            # vertical at each phase. Their difference is how
            # far the upper body has dropped from its address
            # posture by impact. We rotate the current spine
            # line back by that difference so the green line
            # shows the spine restored to its address angle.
            current_spine_angle = float(
                metrics.get(
                    "spine_angle_at_impact_degrees",
                    0.0
                )
            )

            target_spine_angle = float(
                metrics.get(
                    "spine_angle_at_address_degrees",
                    current_spine_angle
                )
            )

            # The current spine vector in screen pixels, measured
            # from the hip pivot up to the shoulders.
            vector_x = shoulder_mid[0] - hip_mid[0]
            vector_y = shoulder_mid[1] - hip_mid[1]

            # Current on-screen tilt from vertical, in the SAME
            # convention the stored angles use: atan2(dx, -dy).
            # Deriving the rotation this way keeps the correction
            # direction correct regardless of coordinate scaling.
            current_screen_angle = math.degrees(
                math.atan2(vector_x, -vector_y)
            )

            desired_screen_angle = (
                current_screen_angle
                + (target_spine_angle - current_spine_angle)
            )

            desired_radians = math.radians(desired_screen_angle)

            spine_pixel_length = math.hypot(vector_x, vector_y)

            # Rebuild the target shoulder point from the desired
            # tilt (dx = L*sin(theta), dy = -L*cos(theta)).
            target_shoulder = (
                int(
                    hip_mid[0]
                    + spine_pixel_length * math.sin(desired_radians)
                ),
                int(
                    hip_mid[1]
                    - spine_pixel_length * math.cos(desired_radians)
                )
            )

            # Current spine (red)
            cv2.line(
                suggested_panel,
                shoulder_mid,
                hip_mid,
                (0, 0, 255),
                6,
                cv2.LINE_AA
            )

            # Target spine (green)
            cv2.line(
                suggested_panel,
                target_shoulder,
                hip_mid,
                (0, 255, 0),
                6,
                cv2.LINE_AA
            )

            # Correction arrow from current shoulder
            # position toward the target shoulder position.
            cv2.arrowedLine(
                suggested_panel,
                shoulder_mid,
                target_shoulder,
                (0, 255, 255),
                4,
                cv2.LINE_AA,
                tipLength=0.3
            )

            # Mark the hip pivot and shoulder endpoints.
            cv2.circle(
                suggested_panel,
                hip_mid,
                8,
                (0, 255, 255),
                -1,
                cv2.LINE_AA
            )

            cv2.circle(
                suggested_panel,
                shoulder_mid,
                8,
                (0, 0, 255),
                -1,
                cv2.LINE_AA
            )

            cv2.circle(
                suggested_panel,
                target_shoulder,
                8,
                (0, 255, 0),
                -1,
                cv2.LINE_AA
            )

            drew_real_spine = True

    if not drew_real_spine:

        # Fallback: landmarks unavailable, keep a simple
        # centred guide rather than crashing.
        fallback_x = int(image_width * 0.47)
        fallback_top_y = int(image_height * 0.42)
        fallback_bottom_y = int(image_height * 0.68)

        cv2.line(
            suggested_panel,
            (fallback_x, fallback_top_y),
            (fallback_x, fallback_bottom_y),
            (0, 0, 255),
            6
        )

        cv2.line(
            suggested_panel,
            (fallback_x + 25, fallback_top_y),
            (fallback_x, fallback_bottom_y),
            (0, 255, 0),
            6
        )

        cv2.arrowedLine(
            suggested_panel,
            (fallback_x, fallback_top_y),
            (fallback_x + 25, fallback_top_y),
            (0, 255, 255),
            4
        )


elif worst_metric == "Hip Sway":

    # Hip sway = lateral movement of the hips during the
    # backswing. Red = current hip centre, green = a more
    # stable target (hips kept more centred over the lower
    # body), yellow arrow = reduce the slide.

    hip_landmarks = detect_pose_on_image(source_image)

    cv2.putText(
        current_panel,
        f"Hip Sway: {float(metrics.get('hip_sway', 0.0)):.3f}",
        (20, image_height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        suggested_panel,
        "Target: minimal lateral sway",
        (20, image_height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    if hip_landmarks is not None:

        hip_mid = midpoint_pixels(
            hip_landmarks[LEFT_HIP],
            hip_landmarks[RIGHT_HIP],
            image_width,
            image_height
        )

        ankle_mid = midpoint_pixels(
            hip_landmarks[LEFT_HIP],
            hip_landmarks[RIGHT_HIP],
            image_width,
            image_height
        )

        # Target: pull the hip centre back toward being
        # stacked over the mid-stance. For a right-handed
        # golfer the trail-side slide is away from target, so
        # nudge the hip marker a small amount toward centre.
        shift = int(image_width * 0.05)
        target_hip = (hip_mid[0] - shift, hip_mid[1])

        # Vertical reference line through the target position
        # so the viewer sees "keep hips here".
        cv2.line(
            suggested_panel,
            (target_hip[0], hip_mid[1] - int(image_height * 0.12)),
            (target_hip[0], hip_mid[1] + int(image_height * 0.12)),
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

        cv2.circle(
            suggested_panel, hip_mid, 11, (0, 0, 255), -1, cv2.LINE_AA
        )
        cv2.circle(
            suggested_panel, target_hip, 11, (0, 255, 0), -1, cv2.LINE_AA
        )

        cv2.arrowedLine(
            suggested_panel,
            hip_mid,
            target_hip,
            (0, 255, 255),
            5,
            cv2.LINE_AA,
            tipLength=0.3
        )


elif worst_metric == "Head Stability":

    # Head stability = how much the head moves from address to
    # impact. Red = current head position, green = where it
    # should stay (steady), yellow arrow = reduce movement.

    head_landmarks = detect_pose_on_image(source_image)

    cv2.putText(
        current_panel,
        f"Head Move: {float(metrics.get('head_movement_address_to_impact', 0.0)):.3f}",
        (20, image_height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        suggested_panel,
        "Target: keep head steady",
        (20, image_height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    if head_landmarks is not None:

        head_px = landmark_to_pixels(
            head_landmarks[NOSE],
            image_width,
            image_height
        )

        # Draw a "keep still" zone around the current head.
        cv2.circle(
            suggested_panel, head_px, 42, (0, 255, 0), 3, cv2.LINE_AA
        )

        cv2.circle(
            suggested_panel, head_px, 10, (0, 0, 255), -1, cv2.LINE_AA
        )

        cv2.putText(
            suggested_panel,
            "Keep head here",
            (head_px[0] + 50, head_px[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


elif worst_metric == "Finish Stability":

    # Finish stability = how settled the body is just after the
    # finish. Red = current body line, green = a balanced,
    # held finish over a stable base.

    finish_landmarks = detect_pose_on_image(source_image)

    cv2.putText(
        current_panel,
        f"Finish Move: {float(metrics.get('finish_stability', 0.0)):.3f}",
        (20, image_height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        suggested_panel,
        "Target: hold a balanced finish",
        (20, image_height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    if finish_landmarks is not None:

        shoulder_mid = midpoint_pixels(
            finish_landmarks[LEFT_SHOULDER],
            finish_landmarks[RIGHT_SHOULDER],
            image_width,
            image_height
        )

        hip_mid = midpoint_pixels(
            finish_landmarks[LEFT_HIP],
            finish_landmarks[RIGHT_HIP],
            image_width,
            image_height
        )

        # Current body line (red) shoulders -> hips.
        cv2.line(
            suggested_panel,
            shoulder_mid,
            hip_mid,
            (0, 0, 255),
            6,
            cv2.LINE_AA
        )

        # Green "balanced" reference: a vertical line down from
        # the shoulders through a stable base, showing the body
        # should be settled and upright over the lead side.
        balanced_bottom = (
            shoulder_mid[0],
            hip_mid[1] + int(image_height * 0.1)
        )

        cv2.line(
            suggested_panel,
            shoulder_mid,
            balanced_bottom,
            (0, 255, 0),
            4,
            cv2.LINE_AA
        )

        cv2.circle(
            suggested_panel, shoulder_mid, 9, (0, 0, 255), -1, cv2.LINE_AA
        )
        cv2.circle(
            suggested_panel, hip_mid, 9, (0, 0, 255), -1, cv2.LINE_AA
        )

        cv2.putText(
            suggested_panel,
            "Hold & balance",
            (shoulder_mid[0] + 20, shoulder_mid[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


elif worst_metric == "Tempo":

    # Tempo is a rhythm metric with no single body position to
    # mark, so instead of drawing on the pose we show the swing
    # TIMING: how the backswing/downswing split compares to the
    # ideal 3:1. The current panel shows the golfer's actual
    # split; the suggested panel shows the 3:1 target split.

    current_tempo = float(
        metrics.get("tempo_ratio", 0.0)
    )

    address_frame = int(phase_report.get("address", 0))
    top_frame = int(phase_report.get("top", 0))
    impact_frame = int(phase_report.get("impact", 0))

    backswing_len = max(1, top_frame - address_frame)
    downswing_len = max(1, impact_frame - top_frame)
    total_len = backswing_len + downswing_len

    bar_left = 30
    bar_right = image_width - 30
    bar_width = bar_right - bar_left
    bar_y = int(image_height * 0.5)
    bar_thickness = 34

    def draw_tempo_bar(panel, split_fraction, back_colour, down_colour):

        split_x = int(bar_left + bar_width * split_fraction)

        cv2.rectangle(
            panel,
            (bar_left, bar_y),
            (split_x, bar_y + bar_thickness),
            back_colour,
            -1
        )

        cv2.rectangle(
            panel,
            (split_x, bar_y),
            (bar_right, bar_y + bar_thickness),
            down_colour,
            -1
        )

        # Divider marking the Top of the backswing.
        cv2.line(
            panel,
            (split_x, bar_y - 10),
            (split_x, bar_y + bar_thickness + 10),
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            panel,
            "Backswing",
            (bar_left + 6, bar_y - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            panel,
            "Down",
            (split_x + 6, bar_y - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # Current: actual split (blue backswing, orange downswing).
    draw_tempo_bar(
        current_panel,
        backswing_len / total_len,
        (255, 160, 0),
        (0, 140, 255)
    )

    # Suggested: ideal 3:1 split (green).
    draw_tempo_bar(
        suggested_panel,
        3.0 / 4.0,
        (0, 200, 0),
        (0, 90, 0)
    )

    cv2.putText(
        current_panel,
        f"Tempo: {current_tempo:.1f} : 1",
        (
            20,
            image_height - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        suggested_panel,
        "Target: 3.0 : 1",
        (
            20,
            image_height - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


else:

    cv2.putText(
        current_panel,
        f"Score: {worst_metric_score}/100",
        (
            20,
            image_height - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        suggested_panel,
        "Target: 90+/100",
        (
            20,
            image_height - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


# -------------------------
# Combine images
# -------------------------

divider = 12

divider_image = (
    255
    * (
        current_panel[
            :,
            :divider
        ]
        * 0
        + 1
    )
)

comparison_image = cv2.hconcat(
    [
        current_panel,
        divider_image,
        suggested_panel
    ]
)


cv2.imwrite(
    str(output_image_path),
    comparison_image
)


print("")
print("COMPARISON IMAGE")
print("----------------")
print(f"Metric: {worst_metric}")
print(f"Score: {worst_metric_score}/100")
print(f"Phase: {phase_label}")
print(f"Saved to: {output_image_path}")
print("")
import cv2
import json
import sys

from pathlib import Path


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
# Select relevant phase image
# -------------------------

PHASE_IMAGE_MAP = {

    "Lead Arm Width": "top.jpg",
    "Trail Arm Position": "top.jpg",
    "Swing Width": "top.jpg",

    "Posture": "impact.jpg",
    "Head Stability": "impact.jpg",

    "Hip Sway": "top.jpg",

    "Finish Stability": "finish.jpg",

    "Tempo": "top.jpg"
}


phase_label_map = {

    "Lead Arm Width": "Top of Backswing",
    "Trail Arm Position": "Top of Backswing",
    "Swing Width": "Top of Backswing",

    "Posture": "Impact",
    "Head Stability": "Impact",

    "Hip Sway": "Top of Backswing",

    "Finish Stability": "Finish",

    "Tempo": "Swing Timing"
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

elif worst_metric == "Posture":

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
        "Improved Spine Angle",
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

    current_spine_x = int(
        image_width * 0.47
    )

    current_spine_top_y = int(
        image_height * 0.42
    )

    current_spine_bottom_y = int(
        image_height * 0.68
    )

    cv2.line(
        suggested_panel,
        (
            current_spine_x,
            current_spine_top_y
        ),
        (
            current_spine_x,
            current_spine_bottom_y
        ),
        (0, 0, 255),
        6
    )

    cv2.line(
        suggested_panel,
        (
            current_spine_x + 25,
            current_spine_top_y
        ),
        (
            current_spine_x,
            current_spine_bottom_y
        ),
        (0, 255, 0),
        6
    )

    cv2.arrowedLine(
        suggested_panel,
        (
            current_spine_x,
            current_spine_top_y
        ),
        (
            current_spine_x + 25,
            current_spine_top_y
        ),
        (0, 255, 255),
        4
    )

    # ---------------------
    # Visual suggestion
    # ---------------------

    elbow_x = int(image_width * 0.42)
    elbow_y = int(image_height * 0.42)

    target_elbow_x = elbow_x - 40
    target_elbow_y = elbow_y - 40

    cv2.circle(
        suggested_panel,
        (elbow_x, elbow_y),
        12,
        (0, 0, 255),
        -1
    )

    cv2.circle(
        suggested_panel,
        (target_elbow_x, target_elbow_y),
        12,
        (0, 255, 0),
        -1
    )

    cv2.arrowedLine(
        suggested_panel,
        (elbow_x, elbow_y),
        (
            target_elbow_x,
            target_elbow_y
        ),
        (0, 255, 255),
        5,
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
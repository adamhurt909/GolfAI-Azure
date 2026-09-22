import cv2
import mediapipe as mp
import math
import json
import sys
from pathlib import Path

with open(
    "data/phase_report.json",
    "r"
) as f:

    phase_report = json.load(f)

ADDRESS_FRAME = phase_report["address"]
TOP_FRAME = phase_report["top"]
IMPACT_FRAME = phase_report["impact"]
FINISH_FRAME = phase_report["finish"]

print("")
print("USING PHASE DETECTOR RESULTS")
print("----------------------------")
print(f"Address frame: {ADDRESS_FRAME}")
print(f"Top frame: {TOP_FRAME}")
print(f"Impact frame: {IMPACT_FRAME}")
print(f"Finish frame: {FINISH_FRAME}")
print("")

if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    video_path = r"videos\Adam_27thJune_26.mp4"



cap = cv2.VideoCapture(video_path)

video_fps = cap.get(
    cv2.CAP_PROP_FPS
)

if video_fps <= 0:
    video_fps = 30.0

print("")
print("VIDEO CHECK")
print("-----------")
print(f"Video path: {video_path}")
print(f"Video opened: {cap.isOpened()}")
print("")

mp_pose = mp.solutions.pose

landmarks_by_frame = []

def angle_between_three_points(point_a, point_b, point_c):
    """
    Calculates the angle at point_b formed by:
    point_a -> point_b -> point_c
    """

    a = (point_a.x, point_a.y)
    b = (point_b.x, point_b.y)
    c = (point_c.x, point_c.y)

    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]

    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)

    cosine_angle = max(min(cosine_angle, 1), -1)

    angle = math.degrees(math.acos(cosine_angle))

    return angle

def distance_between_points(point_a, point_b):
    return math.sqrt(
        (point_a.x - point_b.x) ** 2
        +
        (point_a.y - point_b.y) ** 2
    )

def midpoint(point_a, point_b):
    return {
        "x": (point_a.x + point_b.x) / 2,
        "y": (point_a.y + point_b.y) / 2
    }

def dictionary_distance(point_a, point_b):
    return math.sqrt(
        (
            point_a["x"]
            - point_b["x"]
        ) ** 2
        +
        (
            point_a["y"]
            - point_b["y"]
        ) ** 2
    )

def line_angle(point_a, point_b):
    dx = point_b["x"] - point_a["x"]
    dy = point_b["y"] - point_a["y"]

    return math.degrees(
        math.atan2(
            dy,
            dx
        )
    )

def landmark_line_angle(
    point_a,
    point_b
):

    dx = point_b.x - point_a.x
    dy = point_b.y - point_a.y

    return math.degrees(
        math.atan2(dy, dx)
    )

def angle_difference(angle_1, angle_2):
    difference = angle_2 - angle_1

    while difference > 180:
        difference -= 360

    while difference < -180:
        difference += 360

    return abs(difference)

with mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks_by_frame.append(results.pose_landmarks.landmark)
        else:
            landmarks_by_frame.append(None)

cap.release()

def get_landmarks(frame_number):

    print(
        f"Requested frame: {frame_number}"
    )

    print(
        f"Available frames: {len(landmarks_by_frame)}"
    )

    if (
        frame_number < 0
        or
        frame_number >= len(landmarks_by_frame)
    ):
        raise ValueError(
            f"Frame {frame_number} is outside "
            f"available range "
            f"0-{len(landmarks_by_frame)-1}"
        )

    landmarks = landmarks_by_frame[
        frame_number
    ]

    if landmarks is None:
        raise ValueError(
            f"No landmarks found at "
            f"frame {frame_number}"
        )

    return landmarks

def get_nearest_landmarks(
    frame_number,
    maximum_offset=10
):
    frame_number = min(
        len(landmarks_by_frame) - 1,
        max(
            0,
            frame_number
        )
    )

    if landmarks_by_frame[frame_number] is not None:
        return (
            landmarks_by_frame[frame_number],
            frame_number
        )

    for offset in range(
        1,
        maximum_offset + 1
    ):
        later_frame = frame_number + offset

        if (
            later_frame < len(landmarks_by_frame)
            and landmarks_by_frame[later_frame] is not None
        ):
            return (
                landmarks_by_frame[later_frame],
                later_frame
            )

        earlier_frame = frame_number - offset

        if (
            earlier_frame >= 0
            and landmarks_by_frame[earlier_frame] is not None
        ):
            return (
                landmarks_by_frame[earlier_frame],
                earlier_frame
            )

    return None, frame_number

address = get_landmarks(ADDRESS_FRAME)
top = get_landmarks(TOP_FRAME)
impact = get_landmarks(IMPACT_FRAME)
finish = get_landmarks(FINISH_FRAME)

finish_check_offset = max(
    3,
    round(
        video_fps * 0.20
    )
)

requested_finish_check_frame = min(
    len(landmarks_by_frame) - 1,
    FINISH_FRAME + finish_check_offset
)

finish_check, finish_check_frame = (
    get_nearest_landmarks(
        requested_finish_check_frame
    )
)

# -------------------------
# Tempo
# -------------------------

backswing_frames = (
    TOP_FRAME
    - ADDRESS_FRAME
)

downswing_frames = (
    IMPACT_FRAME
    - TOP_FRAME
)

if downswing_frames > 0:

    tempo_ratio = (
        backswing_frames
        / downswing_frames
    )

else:

    tempo_ratio = 0

# -------------------------
# Head movement
# -------------------------

address_nose = address[mp_pose.PoseLandmark.NOSE]
impact_nose = impact[mp_pose.PoseLandmark.NOSE]

head_movement = distance_between_points(
    address_nose,
    impact_nose
)

# -------------------------
# Arm structure at top
# -------------------------

top_left_shoulder = top[
    mp_pose.PoseLandmark.LEFT_SHOULDER
]

top_left_elbow = top[
    mp_pose.PoseLandmark.LEFT_ELBOW
]

top_left_wrist = top[
    mp_pose.PoseLandmark.LEFT_WRIST
]

top_right_shoulder = top[
    mp_pose.PoseLandmark.RIGHT_SHOULDER
]

top_right_elbow = top[
    mp_pose.PoseLandmark.RIGHT_ELBOW
]

top_right_wrist = top[
    mp_pose.PoseLandmark.RIGHT_WRIST
]

left_arm_angle = angle_between_three_points(
    top_left_shoulder,
    top_left_elbow,
    top_left_wrist
)

right_arm_angle = angle_between_three_points(
    top_right_shoulder,
    top_right_elbow,
    top_right_wrist
)

# Lead arm should generally be the straighter arm
# at the top of the backswing.
#
# Larger angle = straighter arm.

lead_arm_angle_top = max(
    left_arm_angle,
    right_arm_angle
)

trail_arm_angle_top = min(
    left_arm_angle,
    right_arm_angle
)

# -------------------------
# Swing Width
# -------------------------

swing_width = distance_between_points(
    top_right_shoulder,
    top_right_wrist
)

print("")
print("ARM STRUCTURE AT TOP")
print("--------------------")
print(
    f"Lead Arm : "
    f"{lead_arm_angle_top:.2f}"
)
print(
    f"Trail Arm: "
    f"{trail_arm_angle_top:.2f}"
)

print("")
print("SWING WIDTH")
print("----------------")
print(
    f"Shoulder to Wrist: "
    f"{swing_width:.4f}"
)

# -------------------------
# Spine angle change
# -------------------------

address_left_shoulder = address[mp_pose.PoseLandmark.LEFT_SHOULDER]
address_right_shoulder = address[mp_pose.PoseLandmark.RIGHT_SHOULDER]
address_left_hip = address[mp_pose.PoseLandmark.LEFT_HIP]
address_right_hip = address[mp_pose.PoseLandmark.RIGHT_HIP]

impact_left_shoulder = impact[mp_pose.PoseLandmark.LEFT_SHOULDER]
impact_right_shoulder = impact[mp_pose.PoseLandmark.RIGHT_SHOULDER]
impact_left_hip = impact[mp_pose.PoseLandmark.LEFT_HIP]
impact_right_hip = impact[mp_pose.PoseLandmark.RIGHT_HIP]

top_right_wrist = top[
    mp_pose.PoseLandmark.RIGHT_WRIST
]

top_left_hip = top[
    mp_pose.PoseLandmark.LEFT_HIP
]

top_right_hip = top[
    mp_pose.PoseLandmark.RIGHT_HIP
]

address_shoulder_midpoint = midpoint(
    address_left_shoulder,
    address_right_shoulder
)

address_hip_midpoint = midpoint(
    address_left_hip,
    address_right_hip
)

top_hip_midpoint = midpoint(
    top_left_hip,
    top_right_hip
)

# -------------------------
# Hip Sway
# -------------------------

hip_sway = abs(
    top_hip_midpoint["x"]
    - address_hip_midpoint["x"]
)

# -------------------------
# Finish Stability
# -------------------------
# This is a 2D stability proxy, not a direct measurement
# of physical balance.
#
# It measures average head and hip-centre movement during
# the first 0.2 seconds after the detected Finish frame.

if finish_check is not None:

    finish_nose = finish[
        mp_pose.PoseLandmark.NOSE
    ]

    finish_check_nose = finish_check[
        mp_pose.PoseLandmark.NOSE
    ]

    finish_left_hip = finish[
        mp_pose.PoseLandmark.LEFT_HIP
    ]

    finish_right_hip = finish[
        mp_pose.PoseLandmark.RIGHT_HIP
    ]

    finish_check_left_hip = finish_check[
        mp_pose.PoseLandmark.LEFT_HIP
    ]

    finish_check_right_hip = finish_check[
        mp_pose.PoseLandmark.RIGHT_HIP
    ]

    finish_hip_midpoint = midpoint(
        finish_left_hip,
        finish_right_hip
    )

    finish_check_hip_midpoint = midpoint(
        finish_check_left_hip,
        finish_check_right_hip
    )

    finish_head_movement = distance_between_points(
        finish_nose,
        finish_check_nose
    )

    finish_hip_movement = dictionary_distance(
        finish_hip_midpoint,
        finish_check_hip_midpoint
    )

    finish_stability = (
        finish_head_movement
        + finish_hip_movement
    ) / 2

else:

    finish_head_movement = 0
    finish_hip_movement = 0
    finish_stability = 0

impact_shoulder_midpoint = midpoint(
    impact_left_shoulder,
    impact_right_shoulder
)

impact_hip_midpoint = midpoint(
    impact_left_hip,
    impact_right_hip
)

address_spine_angle = line_angle(
    address_hip_midpoint,
    address_shoulder_midpoint
)

impact_spine_angle = line_angle(
    impact_hip_midpoint,
    impact_shoulder_midpoint
)

spine_angle_change = angle_difference(
    address_spine_angle,
    impact_spine_angle
)

# -------------------------
# Report
# -------------------------

print("")
print("GOLF SWING ANALYSIS REPORT")
print("--------------------------")

print(f"Address frame: {ADDRESS_FRAME}")
print(f"Top of backswing frame: {TOP_FRAME}")
print(f"Impact frame: {IMPACT_FRAME}")
print(f"Finish frame: {FINISH_FRAME}")

print("")
print("HEAD MOVEMENT")
print(f"Head movement, address to impact: {head_movement:.4f}")

print("")
print("TEMPO")
print("----------------")
print(
    f"Backswing frames: "
    f"{backswing_frames}"
)

print(
    f"Downswing frames: "
    f"{downswing_frames}"
)

print(
    f"Tempo Ratio: "
    f"{tempo_ratio:.2f}:1"
)

print("")
print("ARM STRUCTURE AT TOP")
print(f"Lead arm angle at top: {lead_arm_angle_top:.2f} degrees")
print(f"Trail arm angle at top: {trail_arm_angle_top:.2f} degrees")

print("")
print("HIP SWAY")
print("----------------")
print(
    f"Address Hip Midpoint X: "
    f"{address_hip_midpoint['x']:.4f}"
)

print(
    f"Top Hip Midpoint X: "
    f"{top_hip_midpoint['x']:.4f}"
)

print(
    f"Hip Movement: "
    f"{hip_sway:.4f}"
)

print("")
print("FINISH STABILITY")
print("----------------")
print(
    f"Finish Frame: "
    f"{FINISH_FRAME}"
)
print(
    f"Stability Check Frame: "
    f"{finish_check_frame}"
)
print(
    f"Head Movement After Finish: "
    f"{finish_head_movement:.4f}"
)
print(
    f"Hip Movement After Finish: "
    f"{finish_hip_movement:.4f}"
)
print(
    f"Finish Stability Movement: "
    f"{finish_stability:.4f}"
)

print("")
print("SPINE ANGLE")
print(f"Spine angle at address: {address_spine_angle:.2f} degrees")
print(f"Spine angle at impact: {impact_spine_angle:.2f} degrees")
print(f"Spine angle change: {spine_angle_change:.2f} degrees")

print("")
print("BASIC INTERPRETATION")

findings = []

if finish_stability <= 0.01:

    findings.append(
        "Finish position appears stable after the swing."
    )

elif finish_stability <= 0.025:

    findings.append(
        "Finish position shows moderate movement."
    )

else:

    findings.append(
        "Finish position moves noticeably after the swing."
    )

if head_movement < 0.05:
    findings.append("Head movement appears low.")
elif head_movement < 0.12:
    findings.append("Head movement appears moderate.")
else:
    findings.append("Head movement appears high.")

if lead_arm_angle_top > 160:
    findings.append("Lead arm appears well extended at the top.")
elif lead_arm_angle_top > 140:
    findings.append("Lead arm has a moderate bend at the top.")
else:
    findings.append("Lead arm appears quite bent at the top.")

if trail_arm_angle_top < 120:
    findings.append("Trail arm has a compact bend at the top.")
else:
    findings.append("Trail arm appears more extended at the top.")

if spine_angle_change < 8:
    findings.append("Spine angle appears well maintained from address to impact.")
elif spine_angle_change < 15:
    findings.append("Spine angle changes moderately through impact.")
else:
    findings.append("Spine angle changes significantly through impact.")

if tempo_ratio >= 2.7 and tempo_ratio <= 3.3:
    findings.append(
        "Tempo is close to the ideal 3:1 ratio."
    )
elif tempo_ratio < 2.7:
    findings.append(
        "Downswing may be too slow relative to backswing."
    )
else:
    findings.append(
        "Backswing may be too slow relative to downswing."
    )

for finding in findings:
    print(f"- {finding}")

swing_report = {
    "frames": {
        "address": ADDRESS_FRAME,
        "top_of_backswing": TOP_FRAME,
        "impact": IMPACT_FRAME,
        "finish": FINISH_FRAME
    },
    "metrics": {
        "head_movement_address_to_impact": round(head_movement, 4),
        "tempo_ratio": round(tempo_ratio, 2),
        "swing_width": round(swing_width, 4),
        "hip_sway": round(hip_sway, 4),
        "finish_stability": round(finish_stability, 4),
        "lead_arm_angle_at_top_degrees": round(lead_arm_angle_top, 2),
        "trail_arm_angle_at_top_degrees": round(trail_arm_angle_top, 2),
        "spine_angle_at_address_degrees": round(address_spine_angle, 2),
        "spine_angle_at_impact_degrees": round(impact_spine_angle, 2),
        "spine_angle_change_degrees": round(spine_angle_change, 2)
    },
    "findings": findings
}

print("")
print("STRUCTURED REPORT")
print(swing_report)

BASE_DIR = Path(__file__).resolve().parent.parent

output_path = (
    BASE_DIR
    / "data"
    / "swing_report.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        swing_report,
        file,
        indent=4
    )

print("")
print(
    f"Report saved to: {output_path}"
)
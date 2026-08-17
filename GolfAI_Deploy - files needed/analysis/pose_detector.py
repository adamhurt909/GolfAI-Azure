import cv2
import mediapipe as mp
import math
import json
import sys

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
    video_path = r"videos\adam_driver_1.mp4"



cap = cv2.VideoCapture(video_path)

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

    dx = (
        point_b.x
        - point_a.x
    )

    dy = (
        point_b.y
        - point_a.y
    )

    return math.degrees(
        math.atan2(
            dy,
            dx
        )
    )
    dx = point_b["x"] - point_a["x"]
    dy = point_b["y"] - point_a["y"]

    return math.degrees(math.atan2(dy, dx))

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
    landmarks = landmarks_by_frame[frame_number]

    if landmarks is None:
        print(f"ERROR: No landmarks found for frame {frame_number}")
        exit()

    return landmarks

address = get_landmarks(ADDRESS_FRAME)
top = get_landmarks(TOP_FRAME)
impact = get_landmarks(IMPACT_FRAME)
finish = get_landmarks(FINISH_FRAME)

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
# Lead arm extension at top
# Assumes right-handed golfer, so lead arm = left arm
# -------------------------

top_left_shoulder = top[mp_pose.PoseLandmark.LEFT_SHOULDER]
top_left_elbow = top[mp_pose.PoseLandmark.LEFT_ELBOW]
top_left_wrist = top[mp_pose.PoseLandmark.LEFT_WRIST]

lead_arm_angle_top = angle_between_three_points(
    top_left_shoulder,
    top_left_elbow,
    top_left_wrist
)

# -------------------------
# Trail arm bend at top
# For right-handed golfer, trail arm = right arm
# -------------------------

top_right_shoulder = top[mp_pose.PoseLandmark.RIGHT_SHOULDER]
top_right_elbow = top[mp_pose.PoseLandmark.RIGHT_ELBOW]
top_right_wrist = top[mp_pose.PoseLandmark.RIGHT_WRIST]

trail_arm_angle_top = angle_between_three_points(
    top_right_shoulder,
    top_right_elbow,
    top_right_wrist
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
# Shoulder turn
# -------------------------

address_shoulder_turn = landmark_line_angle(
    address_left_shoulder,
    address_right_shoulder
)

top_shoulder_turn = landmark_line_angle(
    top_left_shoulder,
    top_right_shoulder
)

shoulder_turn_change = angle_difference(
    address_shoulder_turn,
    top_shoulder_turn
)

# -------------------------
# Hip turn
# -------------------------

address_hip_turn = landmark_line_angle(
    address_left_hip,
    address_right_hip
)

top_hip_turn = landmark_line_angle(
    top_left_hip,
    top_right_hip
)

hip_turn_change = angle_difference(
    address_hip_turn,
    top_hip_turn
)

# -------------------------
# X-Factor
# -------------------------

x_factor = (
    shoulder_turn_change
    - hip_turn_change
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
print("ARM STRUCTURE AT TOP")
print(f"Lead arm angle at top: {lead_arm_angle_top:.2f} degrees")
print(f"Trail arm angle at top: {trail_arm_angle_top:.2f} degrees")

print("")
print("SHOULDER TURN")
print(
    f"Shoulder turn: "
    f"{shoulder_turn_change:.2f} degrees"
)

print("")
print("HIP TURN")
print(
    f"Hip turn: "
    f"{hip_turn_change:.2f} degrees"
)

print("")
print("X-FACTOR")
print(
    f"X-Factor: "
    f"{x_factor:.2f} degrees"
)

print("")
print("SPINE ANGLE")
print(f"Spine angle at address: {address_spine_angle:.2f} degrees")
print(f"Spine angle at impact: {impact_spine_angle:.2f} degrees")
print(f"Spine angle change: {spine_angle_change:.2f} degrees")

print("")
print("BASIC INTERPRETATION")

findings = []

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
        "lead_arm_angle_at_top_degrees": round(lead_arm_angle_top, 2),
        "trail_arm_angle_at_top_degrees": round(trail_arm_angle_top, 2),
        "shoulder_turn_change_degrees": round(shoulder_turn_change, 2),
        "hip_turn_change_degrees": round(hip_turn_change, 2),
        "x_factor_degrees": round(x_factor, 2),
        "spine_angle_at_address_degrees": round(address_spine_angle, 2),
        "spine_angle_at_impact_degrees": round(impact_spine_angle, 2),
        "spine_angle_change_degrees": round(spine_angle_change, 2)
    },
    "findings": findings
}

print("")
print("STRUCTURED REPORT")
print(swing_report)

output_path = r"data\swing_report.json"

with open(output_path, "w") as file:
    json.dump(swing_report, file, indent=4)

print("")
print(f"Report saved to: {output_path}")
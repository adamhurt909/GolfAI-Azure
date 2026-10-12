import cv2
import mediapipe as mp
import math
import json
import sys

from pathlib import Path

BASE_DIR = (
    Path(__file__).resolve().parent.parent
)

PHASE_REPORT = (
    BASE_DIR
    / "data"
    / "phase_report.json"
)

with open(
    PHASE_REPORT,
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
    video_path = "videos/Adam_27thJune_26.mp4"



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

def angle_between_three_points(point_a, point_b, point_c, x_scale=1.0):
    """
    Calculates the angle at point_b formed by:
    point_a -> point_b -> point_c

    x_scale corrects for the video aspect ratio. MediaPipe
    normalises x and y independently, so on a portrait video
    the horizontal axis is compressed relative to vertical.
    Scaling x by the aspect ratio restores true geometry so
    joint angles (like a folded elbow) are measured correctly.
    """

    a = (point_a.x * x_scale, point_a.y)
    b = (point_b.x * x_scale, point_b.y)
    c = (point_c.x * x_scale, point_c.y)

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

    frame_width = 0
    frame_height = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        if frame_width == 0:
            frame_height, frame_width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks_by_frame.append(results.pose_landmarks.landmark)
        else:
            landmarks_by_frame.append(None)

cap.release()


# Aspect ratio of the source video. MediaPipe landmarks are
# normalised to [0, 1] on each axis independently, so on a
# non-square frame the raw x and y are on different pixel
# scales. Correcting x by the aspect ratio makes measured
# angles reflect true on-screen geometry.
if frame_height > 0:
    aspect_ratio = frame_width / frame_height
else:
    aspect_ratio = 1.0

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

# -------------------------------------------------
# Robust elbow-angle measurement at the top.
#
# The trail arm is often partly hidden behind the head
# and body at the top of the backswing, so a single
# frame can give a wildly wrong angle (MediaPipe guesses
# the occluded joint). To stay accurate we:
#   1. Fix handedness: right-handed golfer => left arm is
#      the lead arm, right arm is the trail arm. (This
#      matches the stated assumption used elsewhere and
#      avoids a fragile "straightest arm is the lead"
#      guess that noisy landmarks can flip.)
#   2. Measure across a small window around the Top and
#      take the median, weighting toward frames where the
#      elbow and wrist are actually visible.
#   3. Correct for the video aspect ratio.
# -------------------------------------------------

def measured_elbow_angle(landmarks, shoulder_id, elbow_id, wrist_id):

    return angle_between_three_points(
        landmarks[shoulder_id],
        landmarks[elbow_id],
        landmarks[wrist_id],
        aspect_ratio
    )


def joint_visibility(landmarks, elbow_id, wrist_id):

    return min(
        landmarks[elbow_id].visibility,
        landmarks[wrist_id].visibility
    )


def robust_arm_angle_at_top(shoulder_id, elbow_id, wrist_id):
    """Median elbow angle over a window around the Top frame.

    Only frames where the elbow and wrist are reasonably
    visible are used; this rejects the occlusion spikes that
    otherwise corrupt a single-frame reading.
    """

    window = max(
        2,
        round(video_fps * 0.12)
    )

    visible_angles = []
    all_angles = []

    for frame_index in range(
        TOP_FRAME - window,
        TOP_FRAME + window + 1
    ):

        if (
            frame_index < 0
            or frame_index >= len(landmarks_by_frame)
        ):
            continue

        landmarks = landmarks_by_frame[frame_index]

        if landmarks is None:
            continue

        angle_value = measured_elbow_angle(
            landmarks,
            shoulder_id,
            elbow_id,
            wrist_id
        )

        all_angles.append(angle_value)

        if joint_visibility(landmarks, elbow_id, wrist_id) >= 0.6:
            visible_angles.append(angle_value)

    chosen = visible_angles if visible_angles else all_angles

    if not chosen:
        return 0.0

    chosen.sort()

    return chosen[len(chosen) // 2]


def angle_between_three_points_3d(point_a, point_b, point_c):
    """Angle at point_b in true 3D space (degrees).

    Used with MediaPipe pose_world_landmarks, which estimate
    joint positions in metres. A 3D angle is not fooled by
    foreshortening, so a trail elbow pointing toward the
    camera is still measured as its real anatomical bend.
    """

    bax = point_a.x - point_b.x
    bay = point_a.y - point_b.y
    baz = point_a.z - point_b.z

    bcx = point_c.x - point_b.x
    bcy = point_c.y - point_b.y
    bcz = point_c.z - point_b.z

    dot = (bax * bcx) + (bay * bcy) + (baz * bcz)

    magnitude_ba = math.sqrt(
        bax * bax + bay * bay + baz * baz
    )

    magnitude_bc = math.sqrt(
        bcx * bcx + bcy * bcy + bcz * bcz
    )

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0

    cosine = max(
        -1.0,
        min(1.0, dot / (magnitude_ba * magnitude_bc))
    )

    return math.degrees(math.acos(cosine))


def heavy_arm_angles_at_top():
    """Measure both elbow angles at the Top using the heavy model.

    The light model (used for the main pass) frequently places
    the trail arm on the club/hands at the top of the backswing,
    producing a far-too-straight angle. The heavy model
    (model_complexity=2) locates the arm much more reliably, and
    its 3D world landmarks give an angle that is immune to the
    foreshortening of a single side-on camera.

    We run the heavy model only on a short window of frames
    around the Top so the extra cost stays small, then take the
    median angle over frames where the joints are visible.

    Returns (lead_angle, trail_angle), or (None, None) if the
    heavy pass could not produce a reading.
    """

    # Keep the window tight around the genuine Top. Frames
    # further away are already moving out of the top position
    # and read straighter, which biases the median, so a
    # narrow window gives the most faithful reading.
    window = max(
        2,
        round(video_fps * 0.08)
    )

    start_frame = max(0, TOP_FRAME - window)
    end_frame = TOP_FRAME + window

    lead_angles = []
    trail_angles = []

    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as heavy_pose:

            frame_index = start_frame

            while frame_index <= end_frame:

                success, frame = capture.read()

                if not success:
                    break

                results = heavy_pose.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

                frame_index += 1

                if not results.pose_world_landmarks:
                    continue

                screen = results.pose_landmarks.landmark
                world = results.pose_world_landmarks.landmark

                # Lead (left) arm.
                if (
                    joint_visibility(
                        screen,
                        mp_pose.PoseLandmark.LEFT_ELBOW,
                        mp_pose.PoseLandmark.LEFT_WRIST
                    ) >= 0.5
                ):
                    lead_angles.append(
                        angle_between_three_points_3d(
                            world[mp_pose.PoseLandmark.LEFT_SHOULDER],
                            world[mp_pose.PoseLandmark.LEFT_ELBOW],
                            world[mp_pose.PoseLandmark.LEFT_WRIST]
                        )
                    )

                # Trail (right) arm.
                if (
                    joint_visibility(
                        screen,
                        mp_pose.PoseLandmark.RIGHT_ELBOW,
                        mp_pose.PoseLandmark.RIGHT_WRIST
                    ) >= 0.5
                ):
                    trail_angles.append(
                        angle_between_three_points_3d(
                            world[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                            world[mp_pose.PoseLandmark.RIGHT_ELBOW],
                            world[mp_pose.PoseLandmark.RIGHT_WRIST]
                        )
                    )

    finally:
        capture.release()

    def median(values):
        if not values:
            return None
        values.sort()
        return values[len(values) // 2]

    return median(lead_angles), median(trail_angles)


# Right-handed golfer convention.
LEAD_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
LEAD_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW
LEAD_WRIST = mp_pose.PoseLandmark.LEFT_WRIST

TRAIL_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
TRAIL_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW
TRAIL_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST

# Light-model 2D fallback values.
lead_arm_angle_2d = robust_arm_angle_at_top(
    LEAD_SHOULDER,
    LEAD_ELBOW,
    LEAD_WRIST
)

trail_arm_angle_2d = robust_arm_angle_at_top(
    TRAIL_SHOULDER,
    TRAIL_ELBOW,
    TRAIL_WRIST
)

# Preferred: heavy-model 3D measurement around the Top.
print("")
print("Measuring arm angles with heavy model (3D)...")

heavy_lead_angle, heavy_trail_angle = heavy_arm_angles_at_top()

lead_arm_angle = (
    heavy_lead_angle
    if heavy_lead_angle is not None
    else lead_arm_angle_2d
)

trail_arm_angle = (
    heavy_trail_angle
    if heavy_trail_angle is not None
    else trail_arm_angle_2d
)

lead_arm_measurement_source = (
    "heavy_3d" if heavy_lead_angle is not None else "light_2d"
)

trail_arm_measurement_source = (
    "heavy_3d" if heavy_trail_angle is not None else "light_2d"
)

print(
    f"Lead arm: {lead_arm_angle:.1f} "
    f"({lead_arm_measurement_source})"
)
print(
    f"Trail arm: {trail_arm_angle:.1f} "
    f"({trail_arm_measurement_source})"
)

lead_arm_side = "left"
trail_arm_side = "right"

# Kept for the diagnostic print below.
left_arm_angle = lead_arm_angle
right_arm_angle = trail_arm_angle
arm_angle_difference = abs(
    lead_arm_angle - trail_arm_angle
)

# -------------------------
# Swing Width
# -------------------------

if trail_arm_side == "left":

    trail_shoulder_at_top = top_left_shoulder
    trail_wrist_at_top = top_left_wrist

else:

    trail_shoulder_at_top = top_right_shoulder
    trail_wrist_at_top = top_right_wrist


swing_width = distance_between_points(
    trail_shoulder_at_top,
    trail_wrist_at_top
)

print("")
print("ARM STRUCTURE AT TOP")
print("--------------------")

print(
    f"Raw left arm angle: "
    f"{left_arm_angle:.2f}"
)

print(
    f"Raw right arm angle: "
    f"{right_arm_angle:.2f}"
)

print(
    f"Angle difference: "
    f"{arm_angle_difference:.2f}"
)

print(
    f"Detected lead arm side: "
    f"{lead_arm_side}"
)

print(
    f"Detected trail arm side: "
    f"{trail_arm_side}"
)

print(
    f"Lead Arm: "
    f"{lead_arm_angle:.2f}"
)

print(
    f"Trail Arm: "
    f"{trail_arm_angle:.2f}"
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

def spine_angle_aspect_corrected(hip_midpoint, shoulder_midpoint):
    """Spine line angle in degrees, corrected for frame aspect ratio.

    Measures the tilt of the hip->shoulder line relative to vertical
    (straight up = 0 degrees, leaning forward increases the value),
    using true on-screen geometry rather than normalised coordinates.
    """

    dx = (
        shoulder_midpoint["x"]
        - hip_midpoint["x"]
    ) * aspect_ratio

    dy = (
        shoulder_midpoint["y"]
        - hip_midpoint["y"]
    )

    # Angle from the vertical (upward) axis. dy is negative when the
    # shoulders are above the hips, as expected for a standing golfer.
    return math.degrees(
        math.atan2(dx, -dy)
    )


# Spine angle measured between Address and Impact.
# The change is what the "Posture Maintenance" metric reports:
# how much the spine tilt moves from its address position by impact.
address_spine_angle = spine_angle_aspect_corrected(
    address_hip_midpoint,
    address_shoulder_midpoint
)

impact_spine_angle = spine_angle_aspect_corrected(
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
print(
    f"Lead arm angle at top: "
    f"{lead_arm_angle:.2f} degrees"
)
print(
    f"Trail arm angle at top: "
    f"{trail_arm_angle:.2f} degrees"
)
print(
    f"Lead arm detected side: "
    f"{lead_arm_side}"
)
print(
    f"Trail arm detected side: "
    f"{trail_arm_side}"
)

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

if lead_arm_angle > 160:

    findings.append(
        "Lead arm appears well extended at the top."
    )

elif lead_arm_angle > 140:

    findings.append(
        "Lead arm has a moderate bend at the top."
    )

else:

    findings.append(
        "Lead arm appears quite bent at the top."
    )


if trail_arm_angle < 120:

    findings.append(
        "Trail arm has a compact bend at the top."
    )

else:

    findings.append(
        "Trail arm appears more extended at the top."
    )

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
        "left_arm_angle_at_top_degrees": round(
            left_arm_angle,
            2
        ),

        "right_arm_angle_at_top_degrees": round(
            right_arm_angle,
            2
        ),

        "lead_arm_angle_at_top_degrees": round(
            lead_arm_angle,
            2
        ),

        "trail_arm_angle_at_top_degrees": round(
            trail_arm_angle,
            2
        ),
        "lead_arm_detected_side": lead_arm_side,
        "trail_arm_detected_side": trail_arm_side,
        "spine_angle_at_address_degrees": round(address_spine_angle, 2),
        "spine_angle_at_impact_degrees": round(impact_spine_angle, 2),
        "spine_angle_change_degrees": round(spine_angle_change, 2)
    },
    "findings": findings
}

print("")
print("STRUCTURED REPORT")
print(swing_report)

from pathlib import Path

BASE_DIR = (
    Path(__file__).resolve().parent.parent
)

output_path = (
    BASE_DIR
    / "data"
    / "swing_report.json"
)

import os

with open(output_path, "w") as file:
    json.dump(
        swing_report,
        file,
        indent=4
    )

print("")
print("SWING REPORT SAVED")
print("------------------")
print("cwd:", os.getcwd())
print("file:", output_path)
print("exists:", output_path.exists())
print("")
import json
import sys
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    video_path = r"videos\adam_driver_1.mp4"

SHOW_DEBUG_WINDOWS = "--show" in sys.argv

cap = cv2.VideoCapture(video_path)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

frames = []
hand_x = []
hand_y = []

# -------------------------
# Helper functions
# -------------------------

def fill_missing(values):
    values = np.array(values, dtype=float)

    valid_indices = np.where(~np.isnan(values))[0]

    if len(valid_indices) == 0:
        raise ValueError("No valid values found.")

    first_valid = valid_indices[0]

    # Fill missing values before first valid point
    values[:first_valid] = values[first_valid]

    # Carry forward previous valid value
    for i in range(first_valid + 1, len(values)):
        if np.isnan(values[i]):
            values[i] = values[i - 1]

    return values


def smooth(values, window_size=9):
    values = np.array(values, dtype=float)

    if window_size < 3:
        return values

    padding = window_size // 2

    padded_values = np.pad(
        values,
        (padding, padding),
        mode="edge"
    )

    smoothed_values = np.convolve(
        padded_values,
        np.ones(window_size) / window_size,
        mode="valid"
    )

    return smoothed_values


def first_sustained_above(
    values,
    threshold,
    start,
    end,
    min_run=6
):
    end = min(
        end,
        len(values) - min_run
    )

    for i in range(start, end):

        window = values[
            i:i + min_run
        ]

        if np.all(window > threshold):
            return i

    return None


def last_above(
    values,
    threshold,
    start,
    end
):
    end = min(
        end,
        len(values)
    )

    candidates = np.where(
        values[start:end] > threshold
    )[0]

    if len(candidates) == 0:
        return None

    return start + candidates[-1]

def first_sustained_above(
    values,
    threshold,
    min_run=8
):
    for i in range(
        len(values) - min_run
    ):

        if np.all(
            values[
                i:i + min_run
            ] > threshold
        ):
            return i

    return None


def show_frame(frame, label, frame_number):
    display_frame = frame.copy()

    cv2.putText(
        display_frame,
        f"{label} - Frame {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(label, display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------
# Extract wrist positions
# -------------------------

with mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frames.append(frame.copy())

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = pose.process(rgb_frame)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            left_wrist = landmarks[
                mp_pose.PoseLandmark.LEFT_WRIST
            ]

            right_wrist = landmarks[
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]

            # Midpoint between both wrists = approximate hand position
            avg_x = (left_wrist.x + right_wrist.x) / 2
            avg_y = (left_wrist.y + right_wrist.y) / 2

            hand_x.append(avg_x)
            hand_y.append(avg_y)

        else:
            hand_x.append(np.nan)
            hand_y.append(np.nan)

cap.release()

hand_x = fill_missing(hand_x)
hand_y = fill_missing(hand_y)

hand_x_smooth = smooth(hand_x, window_size=9)
hand_y_smooth = smooth(hand_y, window_size=9)

# -------------------------
# Calculate hand speed
# -------------------------

dx = np.diff(
    hand_x_smooth,
    prepend=hand_x_smooth[0]
)

dy = np.diff(
    hand_y_smooth,
    prepend=hand_y_smooth[0]
)

hand_speed = np.sqrt(dx ** 2 + dy ** 2)

hand_speed_smooth = smooth(
    hand_speed,
    window_size=9
)

total_frames = len(frames)

# -------------------------
# Address Detection
# -------------------------

fps = 30

baseline_speed = hand_speed_smooth[:40]

speed_threshold = (
    np.mean(baseline_speed)
    + (
        np.std(baseline_speed) * 2
    )
)

takeaway_frame = (
    first_sustained_above(
        hand_speed_smooth,
        speed_threshold,
        min_run=8
    )
)

if takeaway_frame is None:
    takeaway_frame = 40

address_frame = max(
    0,
    takeaway_frame - int(fps * 0.5)
)

print("")
print("ADDRESS DETECTION")
print("----------------")
print(f"Takeaway frame: {takeaway_frame}")
print(f"Address frame : {address_frame}")
print("")


# -------------------------
# Top Of Backswing Detection
# -------------------------
# The top is the transition point immediately before the
# sustained downswing acceleration.
#
# We identify it by finding a low-speed turning point before
# the dominant downswing speed burst, while requiring the hands
# to be meaningfully displaced from their address position.

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

address_hand_x = hand_x_smooth[address_frame]
address_hand_y = hand_y_smooth[address_frame]

hand_displacement = np.sqrt(
    (hand_x_smooth - address_hand_x) ** 2
    +
    (hand_y_smooth - address_hand_y) ** 2
)

# Locate the dominant downswing acceleration.
# Ignore the initial setup and the final portion of the video.
impact_burst_search_start = min(
    total_frames - 1,
    address_frame + max(1, int(fps * 0.5))
)

impact_burst_search_end = max(
    impact_burst_search_start + 1,
    int(total_frames * 0.85)
)

impact_burst_frame = (
    impact_burst_search_start
    + int(
        np.argmax(
            hand_speed_smooth[
                impact_burst_search_start:
                impact_burst_search_end
            ]
        )
    )
)

# Search backwards from the acceleration burst.
# The bounds are based on seconds, not fixed frame numbers.
top_search_start = max(
    address_frame + 1,
    impact_burst_frame - int(fps * 2.0)
)

top_search_end = max(
    top_search_start + 1,
    impact_burst_frame - max(1, int(fps * 0.05))
)

search_speed = hand_speed_smooth[
    top_search_start:
    top_search_end
]

search_displacement = hand_displacement[
    top_search_start:
    top_search_end
]

if len(search_speed) == 0:

    top_frame = max(
        address_frame + 1,
        impact_burst_frame - max(1, int(fps * 0.25))
    )

else:

    # Normalise speed and displacement so they can be combined.
    speed_range = np.ptp(search_speed)
    displacement_range = np.ptp(search_displacement)

    if speed_range > 0:
        normalised_speed = (
            search_speed - np.min(search_speed)
        ) / speed_range
    else:
        normalised_speed = np.zeros_like(search_speed)

    if displacement_range > 0:
        normalised_displacement = (
            search_displacement
            - np.min(search_displacement)
        ) / displacement_range
    else:
        normalised_displacement = np.zeros_like(
            search_displacement
        )

    # A good top candidate has:
    # 1. Low hand speed, indicating the transition
    # 2. High displacement from address, indicating a completed backswing
    top_score = (
        normalised_displacement
        -
        normalised_speed
    )

    top_frame = (
        top_search_start
        + int(np.argmax(top_score))
    )

print("")
print("TOP OF BACKSWING DETECTION")
print("--------------------------")
print(f"Video FPS: {fps:.2f}")
print(f"Impact speed burst: {impact_burst_frame}")
print(
    f"Top search range: "
    f"{top_search_start} to {top_search_end}"
)
print(f"Top frame: {top_frame}")
print(
    f"Frames from top to speed burst: "
    f"{impact_burst_frame - top_frame}"
)
print("")

# -------------------------
# Impact Detection
# -------------------------

impact_search_start = top_frame

impact_search_end = min(
    total_frames,
    top_frame + 120
)

impact_region = hand_y_smooth[
    impact_search_start:
    impact_search_end
]

impact_frame = (
    impact_search_start
    + np.argmax(
        impact_region
    )
)

print("")
print("IMPACT DETECTION")
print("----------------")
print(f"Impact frame: {impact_frame}")
print("")

# -------------------------
# Estimate Finish
# -------------------------
# Do not use last_motion_frame here because it can include
# lowering the club, walking, or returning to a neutral stance.
#
# Search within a window scaled from the backswing duration.
# The finish is represented by the highest wrist position
# during the bounded follow-through period.

finish_search_start = (
    impact_frame + 20
)

finish_search_end = min(
    total_frames,
    impact_frame + 150
)

finish_region = (
    hand_speed_smooth[
        finish_search_start:
        finish_search_end
    ]
)

finish_frame = (
    finish_search_start
    + np.argmin(
        finish_region
    )
)

# -------------------------
# Safety checks
# -------------------------
# Keep the phases in the correct order.

top_frame = max(
    top_frame,
    address_frame + 1
)

impact_frame = max(
    impact_frame,
    top_frame + 1
)

finish_frame = max(
    finish_frame,
    impact_frame + 1
)

finish_frame = min(
    finish_frame,
    total_frames - 1
)

# -------------------------
# Print results
# -------------------------

print("")
print("PHASE DETECTION RESULTS")
print("----------------------")
print(f"Address frame: {address_frame}")
print(f"Top frame: {top_frame}")
print(f"Impact frame: {impact_frame}")
print(f"Finish frame: {finish_frame}")
print("")

# -------------------------

# Save phase report

# -------------------------

Path("data").mkdir(exist_ok=True)

phase_report = {
    "address": int(address_frame),
    "top": int(top_frame),
    "impact": int(impact_frame),
    "finish": int(finish_frame)
}

with open(
    "data/phase_report.json",
    "w"
) as f:

    json.dump(
        phase_report,
        f,
        indent=4
    )

print("Phase report saved")
print("data/phase_report.json")
print("")

# -------------------------
# Tempo Analysis
# -------------------------

backswing_frames = top_frame - address_frame
downswing_frames = impact_frame - top_frame

if downswing_frames > 0:

    tempo_ratio = (
        backswing_frames /
        downswing_frames
    )

    print("TEMPO ANALYSIS")
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
        f"Tempo ratio: "
        f"{tempo_ratio:.2f}:1"
    )

else:

    print(
        "Tempo could not "
        "be calculated"
    )

if SHOW_DEBUG_WINDOWS:

# -------------------------
# Plot diagnostics
# -------------------------

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(hand_y_smooth, label="Hand height")
    plt.axvline(address_frame, color="blue", linestyle="--", label="Address")
    plt.axvline(top_frame, color="green", linestyle="--", label="Top")
    plt.axvline(impact_frame, color="red", linestyle="--", label="Impact")
    plt.axvline(finish_frame, color="purple", linestyle="--", label="Finish")
    plt.title("Hand Height Through Swing")
    plt.xlabel("Frame")
    plt.ylabel("Hand Y Position")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(hand_speed_smooth, label="Hand speed")
    plt.axvline(address_frame, color="blue", linestyle="--", label="Address")
    plt.axvline(top_frame, color="green", linestyle="--", label="Top")
    plt.axvline(impact_frame, color="red", linestyle="--", label="Impact")
    plt.axvline(finish_frame, color="purple", linestyle="--", label="Finish")
    plt.title("Hand Speed Through Swing")
    plt.xlabel("Frame")
    plt.ylabel("Speed")
    plt.legend()

    plt.tight_layout()
    plt.show()

# -------------------------
# Show detected frames
# -------------------------

cv2.imwrite(
    "data/address.jpg",
    frames[address_frame]
)

cv2.imwrite(
    "data/top.jpg",
    frames[top_frame]
)

cv2.imwrite(
    "data/impact.jpg",
    frames[impact_frame]
)

cv2.imwrite(
    "data/finish.jpg",
    frames[finish_frame]
)

print("")
print("Phase images saved")
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
# Estimate Address
# -------------------------
# Address is the final slow/stable frame
# before the hands begin moving away.

address_search_end = int(total_frames * 0.25)

early_speed = hand_speed_smooth[:address_search_end]

takeaway_threshold = np.percentile(
    early_speed,
    75
)

takeaway_candidates = np.where(
    early_speed > takeaway_threshold
)[0]

if len(takeaway_candidates) > 0:

    takeaway_frame = takeaway_candidates[0]

    address_frame = max(
        0,
        takeaway_frame - 20
    )

else:

    takeaway_frame = -1
    address_frame = 0

print("")
print(f"Takeaway frame: {takeaway_frame}")
print(f"Address frame: {address_frame}")

# -------------------------
# Estimate impact acceleration area
# -------------------------
# Impact usually happens during the strongest early burst of hand speed.
# We ignore the final part of the video to avoid picking up finish movement.

impact_search_start = max(
    address_frame + 20,
    30
)

impact_search_end = int(total_frames * 0.70)

impact_speed_region = hand_speed_smooth[
    impact_search_start:impact_search_end
]

impact_burst_frame = impact_search_start + np.argmax(
    impact_speed_region
)

# -------------------------
# Estimate Top of Backswing
# -------------------------
# Top should happen before the impact burst.
# In image coordinates, smaller Y means higher on screen.

top_search_start = address_frame + 10
top_search_end = impact_burst_frame

if top_search_end <= top_search_start:
    top_frame = top_search_start
else:
    top_frame = top_search_start + np.argmin(
        hand_y_smooth[top_search_start:top_search_end]
    )

# -------------------------
# Estimate Impact
# -------------------------
# Search only the early part of the downswing.
# This helps avoid selecting a point deep in the follow-through.

impact_window_start = top_frame + 1

impact_window_end = min(
    impact_search_end,
    top_frame + 36
)

address_hand_x = hand_x_smooth[address_frame]
address_hand_y = hand_y_smooth[address_frame]

distance_to_address = np.sqrt(
    (
        hand_x_smooth[
            impact_window_start:impact_window_end
        ] - address_hand_x
    ) ** 2
    +
    (
        hand_y_smooth[
            impact_window_start:impact_window_end
        ] - address_hand_y
    ) ** 2
)

if len(distance_to_address) > 0:
    impact_frame = (
        impact_window_start
        + np.argmin(distance_to_address)
    )
else:
    impact_frame = impact_burst_frame

# -------------------------
# Estimate Finish
# -------------------------
# Finish is when hand movement settles after impact.
# Search after impact for a low-speed region.

finish_search_start = min(
    total_frames - 1,
    impact_frame + 10
)

finish_search_end = min(
    total_frames,
    impact_frame + 80
)

# -------------------------
# Estimate Finish
# -------------------------
# Finish is the highest hand position reached
# after impact before the golfer relaxes.

finish_search_start = impact_frame + 10

finish_search_end = min(
    total_frames,
    impact_frame + 80
)

finish_region_y = hand_y_smooth[
    finish_search_start:finish_search_end
]

if len(finish_region_y) > 0:

    finish_frame = (
        finish_search_start
        + np.argmin(finish_region_y)
    )

else:

    finish_frame = finish_search_start


# -------------------------
# Print results
# -------------------------

print("")
print("AUTOMATIC SWING PHASE DETECTION")
print("--------------------------------")
print(f"Frames analysed: {total_frames}")
print(f"Estimated address frame: {address_frame}")
print(f"Estimated top of backswing frame: {top_frame}")
print(f"Estimated impact frame: {impact_frame}")
print(f"Estimated finish frame: {finish_frame}")
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
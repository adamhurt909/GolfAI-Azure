import cv2

# Change this to whichever video you want to analyse
video_path = r"videos\adam_driver_1.mp4"

cap = cv2.VideoCapture(video_path)

frames = []

while True:
    success, frame = cap.read()

    if not success:
        break

    frames.append(frame)

cap.release()

print(f"Loaded {len(frames)} frames")

current_frame = 0

while True:

    frame = frames[current_frame].copy()

    cv2.putText(
        frame,
        f"Frame: {current_frame}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Frame Viewer", frame)

    key = cv2.waitKey(0)

    # Next frame (D key)
    if key == ord("d"):
        current_frame = min(
            len(frames) - 1,
            current_frame + 1
        )

    # Previous frame (A key)
    elif key == ord("a"):
        current_frame = max(
            0,
            current_frame - 1
        )

    # Jump forward 10 frames (W key)
    elif key == ord("w"):
        current_frame = min(
            len(frames) - 1,
            current_frame + 10
        )

    # Jump back 10 frames (S key)
    elif key == ord("s"):
        current_frame = max(
            0,
            current_frame - 10
        )

    # Quit (Q key)
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
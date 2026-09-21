import cv2
import csv
from pathlib import Path

# Change this to whichever video you want to analyse
video_path = r"videos\Adam_8.mp4"

video_name = Path(video_path).name

csv_path = Path("data") / "manual_phases.csv"

csv_path.parent.mkdir(
    exist_ok=True
)

if not csv_path.exists():

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "video",
                "address",
                "top",
                "impact",
                "finish"
            ]
        )

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

address_frame = None
top_frame = None
impact_frame = None
finish_frame = None

while True:

    frame = frames[current_frame].copy()
    display_frame = cv2.resize(
        frame,
        None,
        fx=0.5,
        fy=0.5
    )

    cv2.putText(
        display_frame,
        f"Frame: {current_frame} / {len(frames)-1}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    print(
    f"\rFrame: {current_frame}",
    end=""
    )

    cv2.imshow("Frame Viewer", display_frame)

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

    # Jump forward 50 frames (E key)
    elif key == ord("e"):
        current_frame = min(
            len(frames) - 1,
            current_frame + 50
        )

    # Jump back 50 frames (R key)
    elif key == ord("r"):
        current_frame = max(
            0,
            current_frame - 50
        )

    # Mark Address (1 key)
    elif key == ord("1"):

        address_frame = current_frame

        print(
            f"\nAddress = {address_frame}"
        )

    # Mark Top (2 key)
    elif key == ord("2"):

        top_frame = current_frame

        print(
            f"\nTop = {top_frame}"
        )

    # Mark Impact (3 key)
    elif key == ord("3"):

        impact_frame = current_frame

        print(
            f"\nImpact = {impact_frame}"
        )

        # Mark Finish (4 key)
    elif key == ord("4"):

        finish_frame = current_frame

        print(
            f"\nFinish = {finish_frame}"
        )

    # Quit queue key
    elif key == ord("q"):

        print("\n")
        print("PHASES")
        print("----------------")
        print(f"Address: {address_frame}")
        print(f"Top: {top_frame}")
        print(f"Impact: {impact_frame}")
        print(f"Finish: {finish_frame}")

        with open(
            csv_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    video_name,
                    address_frame,
                    top_frame,
                    impact_frame,
                    finish_frame
                ]
            )

        print(
            f"Address→Top: "
            f"{top_frame-address_frame}"
        )

        print(
            f"Top→Impact: "
            f"{impact_frame-top_frame}"
        )

        print(
            f"Impact→Finish: "
            f"{finish_frame-impact_frame}"
        )

        print("")
        print(
            f"Saved to: {csv_path}"
        )

        break

cv2.destroyAllWindows()
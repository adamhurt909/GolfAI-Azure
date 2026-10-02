import sys
import time
from pathlib import Path
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

start_time = time.time()

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from openai import AzureOpenAI

from config.azure_config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_DEPLOYMENT
)

def local_coach(report):

    print("")
    print("OFFLINE AI COACH FEEDBACK")
    print("-------------------------")
    print("")

    head_move = report["metrics"][
        "head_movement_address_to_impact"
    ]

    if head_move < 0.03:
        print(
            "✅ Excellent head stability "
            "through impact."
        )
    elif head_move < 0.06:
        print(
            "⚠ Moderate head movement."
        )
    else:
        print(
            "❌ Excessive head movement."
        )

    lead_arm = report["metrics"][
        "lead_arm_angle_at_top_degrees"
    ]

    if lead_arm > 160:
        print(
            "✅ Lead arm remains well extended."
        )
    else:
        print(
            "⚠ Lead arm could remain straighter."
        )

    spine_change = report["metrics"][
        "spine_angle_change_degrees"
    ]

    if abs(spine_change) < 8:
        print(
            "✅ Spine angle maintained effectively."
        )
    else:
        print(
            "⚠ Loss of posture detected."
        )

    print("")
    print(
        "Tip: Connect to Azure OpenAI "
        "for full AI coaching."
    )

print("")
print("CONNECTING TO GOLFAI...")
print("")

# Load swing report

BASE_DIR = (
    Path(__file__).resolve().parent.parent
)

SWING_REPORT = (
    BASE_DIR
    / "data"
    / "swing_report.json"
)

with open(
    SWING_REPORT,
    "r"
) as f:

    report = json.load(f)

print("")
print("CURRENT METRICS")
print(
    json.dumps(
        report["metrics"],
        indent=2
    )
)
print("")

metrics = report["metrics"]

print("")
print("SWING REPORT USED BY AI COACH")
print("----------------------------")
print(json.dumps(report["metrics"], indent=2))
print("")

prompt = f"""
You are GolfAI's coaching engine.

IMPORTANT:

Use the following GolfAI assessment rules.

HEAD MOVEMENT
Excellent <= 0.02
Good <= 0.05

LEAD ARM
Ideal = 145°-160°

TRAIL ARM
Ideal = 80°-90°

SPINE ANGLE CHANGE
Excellent <= 8°
Good <= 14°

TEMPO
Ideal = 2.7 to 3.3

SWING WIDTH
Excellent >= 0.08
Good >= 0.06

HIP SWAY
Excellent <= 0.015
Good <= 0.03

FINISH STABILITY
Excellent <= 0.01
Good <= 0.025

Only describe metrics as strengths if they meet the GolfAI targets.

Only describe metrics as weaknesses if they fall outside the GolfAI targets.

Base all coaching comments on GolfAI's definitions rather than generic golf opinions.

Swing metrics:

{json.dumps(metrics, indent=2)}

Provide:

1. Overall assessment
2. Strengths
3. Improvement opportunities
4. One recommended practice drill

Keep commentary concise and aligned with the GolfAI scoring system.
"""

try:

    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version="2025-04-01-preview"
    )

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content":
                "You are a professional PGA golf coach."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("")
    print("AI GOLF COACH REPORT")
    print("--------------------")
    print("")

    coach_text = (
        response.choices[0].message.content
    )

    print(coach_text)

    print("WRITING NEW COACH REPORT")

    COACH_REPORT = (
        BASE_DIR
        / "data"
        / "coach_report.txt"
    )

    with open(
        COACH_REPORT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(coach_text)

    print("coach_report.txt successfully written")

    print("")
    print("Coach report saved")
    print("data/coach_report.txt")

except Exception as e:

    print("")
    print(
        "Azure unavailable. "
        "Using offline coaching."
    )

    print(
        f"Reason: {str(e)}"
    )

    local_coach(report)

    offline_text = (
        "OFFLINE AI COACH FEEDBACK\n"
        "-------------------------\n\n"
        f"Reason: {str(e)}\n\n"
        "Azure was unavailable, so GolfAI used the local rule-based coach.\n\n"
    )

    head_move = report["metrics"][
        "head_movement_address_to_impact"
    ]

    if head_move < 0.03:
        offline_text += (
            "Excellent head stability through impact.\n"
        )
    elif head_move < 0.06:
        offline_text += (
            "Moderate head movement detected.\n"
        )
    else:
        offline_text += (
            "Excessive head movement detected.\n"
        )

    lead_arm = report["metrics"][
        "lead_arm_angle_at_top_degrees"
    ]

    if lead_arm > 160:
        offline_text += (
            "Lead arm remains well extended.\n"
        )
    else:
        offline_text += (
            "Lead arm could remain straighter at the top.\n"
        )

    spine_change = report["metrics"][
        "spine_angle_change_degrees"
    ]

    if abs(spine_change) < 8:
        offline_text += (
            "Spine angle maintained effectively.\n"
        )
    else:
        offline_text += (
            "Loss of posture detected.\n"
        )

    offline_text += (
        "\nTip: Connect to Azure OpenAI for full coaching.\n"
    )

    print("WRITING OFFLINE COACH REPORT")

    COACH_REPORT = (
        BASE_DIR
        / "data"
        / "coach_report.txt"
    )

    with open(
        COACH_REPORT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(offline_text)

print("")
print(
    f"AI COACH RUNTIME: "
    f"{time.time() - start_time:.2f} seconds"
)
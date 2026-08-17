import sys
from pathlib import Path
import json

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
            "Tip: Connect to the internet "
            "for full AI coaching."
        )

print("")
print("CONNECTING TO GOLFAI...")
print("")

# Load swing report

with open(
    "data/swing_report.json",
    "r"
) as f:

    report = json.load(f)

# Create client

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2025-04-01-preview"
)

prompt = f"""
You are an experienced PGA golf coach.

Analyse this golf swing report.

Provide:

1. Overall assessment
2. Strengths
3. Improvement opportunities
4. One recommended practice drill

Swing report:

{json.dumps(report, indent=2)}
"""

try:

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

    with open(
        "data/coach_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(coach_text)

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
        f"Reason: {type(e).__name__}"
    )

    local_coach(report)

    offline_text = (
    "OFFLINE AI COACH FEEDBACK\n"
    "-------------------------\n\n"
    "Azure was unavailable, so GolfAI used the local rule-based coach.\n\n"
)

head_move = report["metrics"][
    "head_movement_address_to_impact"
]

if head_move < 0.03:
    offline_text += "Excellent head stability through impact.\n"
elif head_move < 0.06:
    offline_text += "Moderate head movement detected.\n"
else:
    offline_text += "Excessive head movement detected.\n"

lead_arm = report["metrics"][
    "lead_arm_angle_at_top_degrees"
]

if lead_arm > 160:
    offline_text += "Lead arm remains well extended.\n"
else:
    offline_text += "Lead arm could remain straighter at the top.\n"

spine_change = report["metrics"][
    "spine_angle_change_degrees"
]

if abs(spine_change) < 8:
    offline_text += "Spine angle maintained effectively.\n"
else:
    offline_text += "Loss of posture detected.\n"

offline_text += (
    "\nTip: Connect to the internet for full AI coaching.\n"
)

with open(
    "data/coach_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(offline_text)
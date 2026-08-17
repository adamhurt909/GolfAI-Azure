import json

input_path = r"data\swing_report.json"

with open(input_path, "r") as file:
    swing_report = json.load(file)

prompt = f"""
You are a helpful golf coaching assistant.

You are reviewing a golf swing analysis report generated from a 2D down-the-line video.

Important limitations:
- This is based on a single camera angle.
- The metrics are 2D estimates, not full 3D biomechanics.
- Do not overstate certainty.
- Give practical, simple coaching advice.

Swing phase frames:
- Address frame: {swing_report["frames"]["address"]}
- Top of backswing frame: {swing_report["frames"]["top_of_backswing"]}
- Impact frame: {swing_report["frames"]["impact"]}
- Finish frame: {swing_report["frames"]["finish"]}

Measured metrics:
- Head movement from address to impact: {swing_report["metrics"]["head_movement_address_to_impact"]}
- Lead arm angle at top: {swing_report["metrics"]["lead_arm_angle_at_top_degrees"]} degrees
- Trail arm angle at top: {swing_report["metrics"]["trail_arm_angle_at_top_degrees"]} degrees
- Spine angle at address: {swing_report["metrics"]["spine_angle_at_address_degrees"]} degrees
- Spine angle at impact: {swing_report["metrics"]["spine_angle_at_impact_degrees"]} degrees
- Spine angle change: {swing_report["metrics"]["spine_angle_change_degrees"]} degrees

Initial findings:
{chr(10).join("- " + finding for finding in swing_report["findings"])}

Please produce a short coaching report with these sections:

1. Overall summary
2. Strengths
3. Areas to monitor
4. One recommended drill
5. What the golfer should feel
6. Caution about limitations of 2D video analysis
"""

print(prompt)
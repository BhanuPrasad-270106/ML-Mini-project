import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Ensure images directory exists
os.makedirs("images", exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'

def create_prediction_card(
    title,
    filename,
    inputs,
    prediction,
    probabilities,
    color_accent,
    badge_symbol,
    subtitle
):
    fig = plt.figure(figsize=(10, 7.5), dpi=200)
    fig.patch.set_facecolor('#0b0f19') # Sleek deep slate
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#0b0f19')
    ax.axis('off')

    # Card background
    card = patches.FancyBboxPatch(
        (0.04, 0.04), 0.92, 0.92,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor='#161e2e', edgecolor='#2d3748', linewidth=1.5
    )
    ax.add_patch(card)

    # Header banner
    banner = patches.FancyBboxPatch(
        (0.06, 0.83), 0.88, 0.11,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor='#0f172a', edgecolor=color_accent, linewidth=1.8
    )
    ax.add_patch(banner)

    # Title & subtitle
    ax.text(0.08, 0.90, f"STUDENT ACADEMIC STATUS PREDICTION SYSTEM", 
            fontsize=15, fontweight='bold', color='#f8fafc', va='center')
    ax.text(0.08, 0.855, f"Pipeline: XGBoost Early-Warning Model | Scenario: {subtitle}", 
            fontsize=9.5, color='#94a3b8', va='center')

    # Status badge (top right of banner)
    badge = patches.FancyBboxPatch(
        (0.70, 0.845), 0.22, 0.08,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=color_accent, edgecolor='none'
    )
    ax.add_patch(badge)
    ax.text(0.81, 0.885, f"[{badge_symbol}] {prediction.upper()}", 
            fontsize=12, fontweight='bold', color='#ffffff', ha='center', va='center')

    # Left Section: Input Parameters Panel
    left_panel = patches.FancyBboxPatch(
        (0.06, 0.08), 0.42, 0.72,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor='#0d131f', edgecolor='#243044', linewidth=1.2
    )
    ax.add_patch(left_panel)

    ax.text(0.08, 0.76, "USER INPUT PARAMETERS", 
            fontsize=12, fontweight='bold', color='#38bdf8', va='center')
    ax.plot([0.08, 0.46], [0.74, 0.74], color='#334155', lw=1)

    y_pos = 0.70
    for label, val in inputs.items():
        ax.text(0.08, y_pos, label, fontsize=9.5, color='#94a3b8', va='center')
        ax.text(0.45, y_pos, str(val), fontsize=10, fontweight='bold', color='#ffffff', ha='right', va='center')
        ax.plot([0.08, 0.45], [y_pos - 0.022, y_pos - 0.022], color='#1e293b', lw=0.8)
        y_pos -= 0.055

    # Right Section: Prediction Output & Probabilities Panel
    right_panel = patches.FancyBboxPatch(
        (0.52, 0.08), 0.42, 0.72,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor='#0d131f', edgecolor='#243044', linewidth=1.2
    )
    ax.add_patch(right_panel)

    ax.text(0.54, 0.76, "MODEL PREDICTION & CONFIDENCE", 
            fontsize=12, fontweight='bold', color='#38bdf8', va='center')
    ax.plot([0.54, 0.92], [0.74, 0.74], color='#334155', lw=1)

    # Big Result Display Box
    res_box = patches.FancyBboxPatch(
        (0.54, 0.59), 0.38, 0.12,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor='#1e293b', edgecolor=color_accent, linewidth=1.5
    )
    ax.add_patch(res_box)
    ax.text(0.56, 0.67, "PREDICTED STATUS:", fontsize=9, color='#94a3b8')
    ax.text(0.56, 0.62, f"{prediction.upper()}", fontsize=18, fontweight='bold', color=color_accent)

    # Probabilities Header
    ax.text(0.54, 0.54, "CLASS PROBABILITY DISTRIBUTION", fontsize=10.5, fontweight='bold', color='#e2e8f0')

    # Probability bars
    bar_y = 0.47
    bar_colors = {
        'Dropout': '#ef4444',   # Red
        'Enrolled': '#f59e0b',  # Amber
        'Graduate': '#10b981'   # Emerald
    }

    for cls, pct in probabilities.items():
        ax.text(0.54, bar_y + 0.025, cls, fontsize=9.5, fontweight='bold', color='#e2e8f0')
        ax.text(0.92, bar_y + 0.025, f"{pct:.2f}%", fontsize=10, fontweight='bold', color=bar_colors[cls], ha='right')

        # Background track
        track = patches.Rectangle((0.54, bar_y - 0.015), 0.38, 0.022, facecolor='#1f2937', edgecolor='none')
        ax.add_patch(track)

        # Active fill
        fill_width = 0.38 * (pct / 100.0)
        fill = patches.Rectangle((0.54, bar_y - 0.015), fill_width, 0.022, facecolor=bar_colors[cls], edgecolor='none')
        ax.add_patch(fill)

        bar_y -= 0.085

    # Bottom summary tag
    ax.text(0.54, 0.16, "Decision Logic / Impact Assessment:", fontsize=9, fontweight='bold', color='#94a3b8')
    if prediction == "Dropout":
        assessment = "CRITICAL RISK: Zero approved units with unpaid tuition fees.\nAction: Immediate academic intervention and advisory counseling needed."
    elif prediction == "Enrolled":
        assessment = "CONTINUING STUDENT: Partial units approved (5 of 6).\nAction: Monitor curriculum completion into the second semester."
    else:
        assessment = "HIGH SUCCESS TRAJECTORY: 100% units approved with high grade.\nAction: Student is on steady track toward timely graduation."
    
    ax.text(0.54, 0.10, assessment, fontsize=8.5, color='#cbd5e1', va='bottom')

    plt.savefig(f"images/{filename}", dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Generated: images/{filename}")

# 1. DROPOUT CASE
create_prediction_card(
    title="STUDENT DROPOUT RISK PREDICTION",
    filename="student_dropout_prediction_output.png",
    inputs={
        "Age at enrollment": "25 years",
        "Curricular units 1st sem (enrolled)": "6",
        "Curricular units 1st sem (approved)": "0",
        "Curricular units 1st sem (grade)": "0.0 / 20.0",
        "Tuition fees up to date": "No (0)",
        "Scholarship holder": "No (0)",
        "Admission grade": "115.0 / 200.0",
        "Course Code": "9147 (Management)",
        "Previous qualification": "Secondary education"
    },
    prediction="Dropout",
    probabilities={"Dropout": 89.26, "Enrolled": 10.11, "Graduate": 0.63},
    color_accent="#ef4444",
    badge_symbol="RISK ALERT",
    subtitle="High Risk Student (Dropped Out)"
)

# 2. ENROLLED CASE
create_prediction_card(
    title="STUDENT STATUS: ENROLLED",
    filename="student_enrolled_prediction_output.png",
    inputs={
        "Age at enrollment": "18 years",
        "Curricular units 1st sem (enrolled)": "6",
        "Curricular units 1st sem (approved)": "5",
        "Curricular units 1st sem (grade)": "10.0 / 20.0",
        "Tuition fees up to date": "Yes (1)",
        "Scholarship holder": "No (0)",
        "Admission grade": "120.7 / 200.0",
        "Course Code": "9238 (Social Service)",
        "Previous qualification": "Secondary education"
    },
    prediction="Enrolled",
    probabilities={"Dropout": 21.67, "Enrolled": 47.65, "Graduate": 30.68},
    color_accent="#f59e0b",
    badge_symbol="CONTINUING",
    subtitle="Continuing Student (Enrolled)"
)

# 3. GRADUATE CASE
create_prediction_card(
    title="STUDENT STATUS: GRADUATE",
    filename="student_graduate_prediction_output.png",
    inputs={
        "Age at enrollment": "18 years",
        "Curricular units 1st sem (enrolled)": "6",
        "Curricular units 1st sem (approved)": "6",
        "Curricular units 1st sem (grade)": "15.5 / 20.0",
        "Tuition fees up to date": "Yes (1)",
        "Scholarship holder": "Yes (1)",
        "Admission grade": "150.0 / 200.0",
        "Course Code": "9500 (Nursing)",
        "Previous qualification": "Secondary education"
    },
    prediction="Graduate",
    probabilities={"Dropout": 5.71, "Enrolled": 2.71, "Graduate": 91.57},
    color_accent="#10b981",
    badge_symbol="SUCCESS",
    subtitle="Successful Student (Graduate)"
)

# 4. TRIPLE COMPARISON DASHBOARD
def create_comparison_dashboard():
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), dpi=200)
    fig.patch.set_facecolor('#0b0f19')

    scenarios = [
        {
            "ax": axes[0],
            "title": "SCENARIO 1: DROPOUT",
            "prediction": "DROPOUT",
            "color": "#ef4444",
            "inputs": [("Age", "25"), ("Units Approved", "0 / 6"), ("Grade", "0.0 / 20.0"), ("Tuition Paid", "No"), ("Scholarship", "No"), ("Admission Grade", "115.0")],
            "probs": [("Dropout", 89.26, "#ef4444"), ("Enrolled", 10.11, "#f59e0b"), ("Graduate", 0.63, "#10b981")]
        },
        {
            "ax": axes[1],
            "title": "SCENARIO 2: ENROLLED",
            "prediction": "ENROLLED",
            "color": "#f59e0b",
            "inputs": [("Age", "18"), ("Units Approved", "5 / 6"), ("Grade", "10.0 / 20.0"), ("Tuition Paid", "Yes"), ("Scholarship", "No"), ("Admission Grade", "120.7")],
            "probs": [("Dropout", 21.67, "#ef4444"), ("Enrolled", 47.65, "#f59e0b"), ("Graduate", 30.68, "#10b981")]
        },
        {
            "ax": axes[2],
            "title": "SCENARIO 3: GRADUATE",
            "prediction": "GRADUATE",
            "color": "#10b981",
            "inputs": [("Age", "18"), ("Units Approved", "6 / 6"), ("Grade", "15.5 / 20.0"), ("Tuition Paid", "Yes"), ("Scholarship", "Yes"), ("Admission Grade", "150.0")],
            "probs": [("Dropout", 5.71, "#ef4444"), ("Enrolled", 2.71, "#f59e0b"), ("Graduate", 91.57, "#10b981")]
        }
    ]

    for item in scenarios:
        ax = item["ax"]
        ax.set_facecolor('#161e2e')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Card container
        card = patches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor='#161e2e', edgecolor='#2d3748', lw=1.5)
        ax.add_patch(card)

        # Header box
        hdr = patches.FancyBboxPatch((0.05, 0.84), 0.90, 0.12, boxstyle="round,pad=0.015,rounding_size=0.02", facecolor=item["color"], edgecolor='none')
        ax.add_patch(hdr)
        ax.text(0.5, 0.90, item["title"], ha='center', va='center', fontsize=12, fontweight='bold', color='#ffffff')

        # Subtitle
        ax.text(0.08, 0.79, "INPUT FEATURES:", fontsize=10, fontweight='bold', color='#38bdf8')
        y_pos = 0.74
        for k, v in item["inputs"]:
            ax.text(0.08, y_pos, k, fontsize=9, color='#94a3b8')
            ax.text(0.92, y_pos, v, fontsize=9.5, fontweight='bold', color='#ffffff', ha='right')
            ax.plot([0.08, 0.92], [y_pos - 0.015, y_pos - 0.015], color='#1f2937', lw=0.7)
            y_pos -= 0.048

        # Output prediction box
        p_box = patches.FancyBboxPatch((0.08, 0.38), 0.84, 0.08, boxstyle="round,pad=0.01,rounding_size=0.015", facecolor='#0d131f', edgecolor=item["color"], lw=1.2)
        ax.add_patch(p_box)
        ax.text(0.12, 0.42, "PREDICTED STATUS:", fontsize=8.5, color='#94a3b8')
        ax.text(0.88, 0.42, item["prediction"], fontsize=12, fontweight='bold', color=item["color"], ha='right')

        # Probability distribution bars
        ax.text(0.08, 0.32, "CLASS PROBABILITIES:", fontsize=9.5, fontweight='bold', color='#e2e8f0')
        by = 0.25
        for cname, cpct, ccol in item["probs"]:
            ax.text(0.08, by + 0.015, cname, fontsize=8.5, color='#cbd5e1')
            ax.text(0.92, by + 0.015, f"{cpct:.1f}%", fontsize=9, fontweight='bold', color=ccol, ha='right')
            track = patches.Rectangle((0.08, by - 0.01), 0.84, 0.016, facecolor='#1f2937', edgecolor='none')
            ax.add_patch(track)
            fill = patches.Rectangle((0.08, by - 0.01), 0.84 * (cpct / 100.0), 0.016, facecolor=ccol, edgecolor='none')
            ax.add_patch(fill)
            by -= 0.06

    plt.suptitle("STUDENT OUTCOME PREDICTION COMPARISON (DROPOUT vs ENROLLED vs GRADUATE)", 
                 fontsize=14, fontweight='bold', color='#ffffff', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("images/student_outcomes_comparison_dashboard.png", dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Generated: images/student_outcomes_comparison_dashboard.png")

create_comparison_dashboard()

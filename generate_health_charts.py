"""
generate_health_charts.py
=========================
Health Monitoring System – Automated Chart & Graph Generator
Run this script after every session of Health_Monitoring_System.m
It reads all saved .mat files and produces a full PDF + individual JPGs.

Usage (from MATLAB working directory):
    python generate_health_charts.py

Requirements:
    pip install matplotlib numpy scipy
    pip install mat73          # only needed if you use MATLAB R2019b+ -v7.3 mat files
"""

import os
import sys
import glob
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

# ── dependency check ──────────────────────────────────────────────────────────
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import MaxNLocator
    from scipy.io import loadmat
except ImportError as e:
    print(f"[ERROR] Missing library: {e}")
    print("Install with:  pip install matplotlib numpy scipy")
    sys.exit(1)

# ── optional mat73 for newer MATLAB formats ───────────────────────────────────
try:
    import mat73
    HAS_MAT73 = True
except ImportError:
    HAS_MAT73 = False


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "blue":   "#2A6EBB",
    "green":  "#27AE60",
    "orange": "#E67E22",
    "red":    "#E74C3C",
    "purple": "#8E44AD",
    "teal":   "#16A085",
    "navy":   "#1A2C4E",
    "light":  "#F0F4FA",
    "grey":   "#BDC3C7",
}

CHART_COLORS = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"],
                PALETTE["red"],  PALETTE["purple"], PALETTE["teal"]]

STYLE = {
    "facecolor":  "#F4F8FF",
    "panel_bg":   "#FFFFFF",
    "title_size": 13,
    "label_size": 10,
    "tick_size":  9,
}


def styled_ax(ax):
    """Apply consistent styling to an axes."""
    ax.set_facecolor(STYLE["panel_bg"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(labelsize=STYLE["tick_size"], colors="#444444")
    ax.yaxis.label.set_color("#444444")
    ax.xaxis.label.set_color("#444444")
    ax.title.set_color(PALETTE["navy"])
    ax.title.set_fontsize(STYLE["title_size"])
    ax.title.set_fontweight("bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    return ax


def save_fig(fig, name, out_dir):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor(), format="jpeg")
    plt.close(fig)
    print(f"  ✔  Saved  →  {path}")
    return path


def timestamp_label():
    return datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_mat_safe(path):
    """Load a .mat file using scipy; fall back to mat73 for v7.3 files."""
    try:
        return loadmat(path)
    except Exception:
        if HAS_MAT73:
            try:
                return mat73.loadmat(path)
            except Exception:
                pass
    return None


def extract_string(val):
    """Robustly pull a Python string out of a MATLAB char/cell array."""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, np.ndarray):
        flat = val.flatten()
        if flat.dtype.kind in ("U", "S", "O"):
            s = flat[0] if flat.size else ""
            return str(s).strip()
        return str(flat[0]).strip() if flat.size else ""
    return str(val).strip()


def extract_scalar(val):
    """Pull a float out of a MATLAB scalar / 1-element array."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, np.ndarray):
        return float(val.flatten()[0])
    return float(val)


def load_users(mat_dir):
    """Return list of dicts from users.mat."""
    path = os.path.join(mat_dir, "users.mat")
    data = load_mat_safe(path)
    if data is None:
        return []
    raw = data.get("users", None)
    if raw is None:
        return []

    users = []
    # MATLAB struct array → numpy structured array
    raw_flat = raw.flatten() if isinstance(raw, np.ndarray) else [raw]
    for u in raw_flat:
        try:
            name   = extract_string(u["Name"])
            mobile = extract_string(u["Mobile"])
            users.append({"Name": name, "Mobile": mobile})
        except Exception:
            pass
    return users


def load_master_records(mat_dir):
    """Return list of health-data dicts from master_health_records.mat."""
    path = os.path.join(mat_dir, "master_health_records.mat")
    data = load_mat_safe(path)
    if data is None:
        return []
    raw = data.get("masterRecords", None)
    if raw is None:
        return []

    records = []
    raw_flat = raw.flatten() if isinstance(raw, np.ndarray) else [raw]
    for cell in raw_flat:
        try:
            # cell may be a 0-d object array wrapping the struct
            item = cell.flatten()[0] if isinstance(cell, np.ndarray) else cell
            rec = {
                "PatientName":   extract_string(item["PatientName"]),
                "Mobile":        extract_string(item["Mobile"]),
                "BloodPressure": extract_string(item["BloodPressure"]),
                "BloodSugar":    extract_scalar(item["BloodSugar"]),
                "HeartRate":     extract_scalar(item["HeartRate"]),
                "SPO2":          extract_scalar(item["SPO2"]),
                "Cholesterol":   extract_scalar(item["Cholesterol"]),
                "Temperature":   extract_scalar(item["Temperature"]),
                "RecordDate":    extract_string(item["RecordDate"]),
            }
            records.append(rec)
        except Exception:
            pass
    return records


def load_individual_records(mat_dir):
    """
    Also scan HealthData_<Name>_<Mobile>.mat files so we don't miss
    records that weren't appended to master yet.
    """
    pattern = os.path.join(mat_dir, "HealthData_*.mat")
    extras  = []
    for fpath in glob.glob(pattern):
        data = load_mat_safe(fpath)
        if data is None:
            continue
        item = data.get("healthData", None)
        if item is None:
            continue
        try:
            item_s = item.flatten()[0] if isinstance(item, np.ndarray) else item
            rec = {
                "PatientName":   extract_string(item_s["PatientName"]),
                "Mobile":        extract_string(item_s["Mobile"]),
                "BloodPressure": extract_string(item_s["BloodPressure"]),
                "BloodSugar":    extract_scalar(item_s["BloodSugar"]),
                "HeartRate":     extract_scalar(item_s["HeartRate"]),
                "SPO2":          extract_scalar(item_s["SPO2"]),
                "Cholesterol":   extract_scalar(item_s["Cholesterol"]),
                "Temperature":   extract_scalar(item_s["Temperature"]),
                "RecordDate":    extract_string(item_s["RecordDate"]),
            }
            extras.append(rec)
        except Exception:
            pass
    return extras


def merge_records(master, individuals):
    """Merge, de-duplicate by (Name, Mobile, Date)."""
    seen = set()
    merged = []
    for r in master + individuals:
        key = (r["PatientName"], r["Mobile"], r["RecordDate"])
        if key not in seen:
            seen.add(key)
            merged.append(r)
    return merged


def parse_bp(bp_str):
    """Return (systolic, diastolic) floats or (None, None)."""
    try:
        parts = bp_str.split("/")
        return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        return None, None


# ═════════════════════════════════════════════════════════════════════════════
# DEMO DATA  (used when no .mat files are found – e.g., first-time testing)
# ═════════════════════════════════════════════════════════════════════════════

DEMO_USERS = [
    {"Name": "Alice", "Mobile": "9876543210"},
    {"Name": "Bob",   "Mobile": "9123456780"},
    {"Name": "Carol", "Mobile": "9000011111"},
    {"Name": "David", "Mobile": "9555577777"},
]

DEMO_RECORDS = [
    {"PatientName": "Alice", "Mobile": "9876543210",
     "BloodPressure": "118/76", "BloodSugar": 92.0,
     "HeartRate": 72.0, "SPO2": 98.0, "Cholesterol": 185.0,
     "Temperature": 36.8, "RecordDate": "01-06-2025 09:00:00"},
    {"PatientName": "Alice", "Mobile": "9876543210",
     "BloodPressure": "124/80", "BloodSugar": 105.0,
     "HeartRate": 78.0, "SPO2": 97.0, "Cholesterol": 192.0,
     "Temperature": 37.1, "RecordDate": "08-06-2025 09:30:00"},
    {"PatientName": "Bob",   "Mobile": "9123456780",
     "BloodPressure": "135/88", "BloodSugar": 148.0,
     "HeartRate": 85.0, "SPO2": 96.0, "Cholesterol": 220.0,
     "Temperature": 37.3, "RecordDate": "03-06-2025 10:15:00"},
    {"PatientName": "Carol", "Mobile": "9000011111",
     "BloodPressure": "112/72", "BloodSugar": 88.0,
     "HeartRate": 65.0, "SPO2": 99.0, "Cholesterol": 170.0,
     "Temperature": 36.5, "RecordDate": "05-06-2025 11:00:00"},
    {"PatientName": "David", "Mobile": "9555577777",
     "BloodPressure": "145/95", "BloodSugar": 182.0,
     "HeartRate": 92.0, "SPO2": 94.0, "Cholesterol": 260.0,
     "Temperature": 37.6, "RecordDate": "06-06-2025 08:45:00"},
    {"PatientName": "Bob",   "Mobile": "9123456780",
     "BloodPressure": "130/85", "BloodSugar": 138.0,
     "HeartRate": 82.0, "SPO2": 96.5, "Cholesterol": 215.0,
     "Temperature": 37.0, "RecordDate": "10-06-2025 09:00:00"},
    {"PatientName": "Alice", "Mobile": "9876543210",
     "BloodPressure": "120/78", "BloodSugar": 98.0,
     "HeartRate": 74.0, "SPO2": 98.5, "Cholesterol": 188.0,
     "Temperature": 36.9, "RecordDate": "15-06-2025 10:00:00"},
    {"PatientName": "David", "Mobile": "9555577777",
     "BloodPressure": "142/92", "BloodSugar": 175.0,
     "HeartRate": 89.0, "SPO2": 95.0, "Cholesterol": 255.0,
     "Temperature": 37.5, "RecordDate": "17-06-2025 11:30:00"},
]


# ═════════════════════════════════════════════════════════════════════════════
# CHART GENERATORS
# ═════════════════════════════════════════════════════════════════════════════

# ── CHART 1 : Patient Registration Overview ───────────────────────────────────
def chart_registration_overview(users, records, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                             facecolor=STYLE["facecolor"])
    fig.suptitle("Chart 1 — Patient Registration Overview",
                 fontsize=15, fontweight="bold", color=PALETTE["navy"], y=1.01)

    # 1a – Registered patients bar
    ax = styled_ax(axes[0])
    names  = [u["Name"] for u in users]
    counts = [sum(1 for r in records if r["Mobile"] == u["Mobile"]) for u in users]
    if not names:
        names, counts = ["No data"], [0]
    bars = ax.bar(names, counts, color=CHART_COLORS[:len(names)],
                  edgecolor="white", linewidth=1.2, width=0.55, zorder=3)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Health Records per Patient")
    ax.set_ylabel("Number of Records")
    ax.set_ylim(0, max(counts or [1]) + 2)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # 1b – Pie: patients with / without records
    ax2 = styled_ax(axes[1])
    ax2.grid(False)
    mobiles_with = {r["Mobile"] for r in records}
    with_rec  = sum(1 for u in users if u["Mobile"] in mobiles_with)
    without   = len(users) - with_rec
    if len(users) == 0:
        sizes, lbls = [1], ["No patients"]
        clrs = [PALETTE["grey"]]
    else:
        sizes = [with_rec, without] if without else [with_rec]
        lbls  = ["With Records", "No Records Yet"] if without else ["All have Records"]
        clrs  = [PALETTE["green"], PALETTE["grey"]] if without else [PALETTE["green"]]
    wedges, texts, autos = ax2.pie(
        sizes, labels=lbls, colors=clrs, autopct="%1.0f%%",
        startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops={"fontsize": 10})
    for a in autos:
        a.set_fontsize(10); a.set_fontweight("bold"); a.set_color("white")
    ax2.set_title("Patient Engagement")

    # 1c – Total system stats summary
    ax3 = styled_ax(axes[2])
    ax3.axis("off")
    ax3.grid(False)
    stats = [
        ("Registered Patients",    len(users)),
        ("Total Health Records",   len(records)),
        ("Unique Patients\nwith Records", with_rec),
        ("Avg Records / Patient",  f"{len(records)/max(len(users),1):.1f}"),
        ("Max Records (1 patient)",
         max(counts) if counts else 0),
        ("Generated",              timestamp_label()),
    ]
    y0 = 0.95
    for label, value in stats:
        ax3.text(0.05, y0, label + ":", fontsize=10, color="#666666",
                 transform=ax3.transAxes, va="top")
        ax3.text(0.95, y0, str(value), fontsize=11, fontweight="bold",
                 color=PALETTE["navy"], transform=ax3.transAxes, va="top", ha="right")
        y0 -= 0.16
    ax3.set_title("System Summary")

    fig.tight_layout()
    return save_fig(fig, "Chart1_Registration_Overview.jpg", out_dir)


# ── CHART 2 : Vital-Signs Distribution (Box + Violin) ────────────────────────
def chart_vital_distributions(records, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10),
                             facecolor=STYLE["facecolor"])
    fig.suptitle("Chart 2 — Vital Signs Distribution Across All Patients",
                 fontsize=15, fontweight="bold", color=PALETTE["navy"])

    vitals = [
        ("BloodSugar",   "Blood Sugar (mg/dL)",  70, 100,  125, "orange"),
        ("HeartRate",    "Heart Rate (bpm)",      60,  80,  100, "blue"),
        ("SPO2",         "SPO2 (%)",              95,  99,  100, "teal"),
        ("Cholesterol",  "Cholesterol (mg/dL)",   0,  200,  240, "purple"),
        ("Temperature",  "Temperature (°C)",      36,  37.5, 38, "green"),
    ]

    for idx, (key, title, low_norm, high_norm, max_concern, col) in enumerate(vitals):
        row, c = divmod(idx, 3)
        ax = styled_ax(axes[row][c])
        vals = [r[key] for r in records]
        if not vals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color=PALETTE["grey"])
            ax.set_title(title)
            continue

        # colour each point by health zone
        point_colors = []
        for v in vals:
            if v < low_norm or v > max_concern:
                point_colors.append(PALETTE["red"])
            elif low_norm <= v <= high_norm:
                point_colors.append(PALETTE["green"])
            else:
                point_colors.append(PALETTE["orange"])

        parts = ax.violinplot(vals, positions=[1], widths=0.6,
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(PALETTE[col])
            pc.set_alpha(0.35)
        parts["cmedians"].set_color(PALETTE["navy"])
        parts["cmedians"].set_linewidth(2)

        jitter = np.random.uniform(-0.08, 0.08, len(vals))
        ax.scatter(np.ones(len(vals)) + jitter, vals,
                   c=point_colors, s=50, zorder=5, edgecolors="white", linewidth=0.6)

        # reference lines
        ax.axhline(low_norm,  color=PALETTE["green"],  linewidth=1.2,
                   linestyle="--", alpha=0.7, label=f"Low normal ({low_norm})")
        ax.axhline(high_norm, color=PALETTE["orange"], linewidth=1.2,
                   linestyle="--", alpha=0.7, label=f"High normal ({high_norm})")

        ax.set_title(title)
        ax.set_xlim(0.5, 1.5)
        ax.set_xticks([])
        ax.set_ylabel(title.split("(")[1].rstrip(")") if "(" in title else "Value")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.8)

        # stats annotation
        txt = f"μ={np.mean(vals):.1f}  σ={np.std(vals):.1f}\nMin={np.min(vals):.1f}  Max={np.max(vals):.1f}"
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8,
                color="#555555", va="bottom")

    # hide unused subplot
    styled_ax(axes[1][2]).axis("off")
    axes[1][2].text(0.5, 0.5, "● Normal range\n● Borderline\n● Concern",
                    ha="center", va="center", transform=axes[1][2].transAxes,
                    fontsize=12, color=PALETTE["navy"],
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#EAF4FF",
                              edgecolor=PALETTE["blue"], linewidth=1.5))

    fig.tight_layout()
    return save_fig(fig, "Chart2_Vital_Distributions.jpg", out_dir)


# ── CHART 3 : Per-Patient Health Trends (Time-Series) ─────────────────────────
def chart_patient_trends(records, out_dir):
    # Group by patient
    patients = {}
    for r in records:
        key = r["PatientName"]
        patients.setdefault(key, []).append(r)

    metrics = ["BloodSugar", "HeartRate", "SPO2", "Cholesterol", "Temperature"]
    labels  = ["Sugar (mg/dL)", "Heart Rate (bpm)", "SPO2 (%)",
               "Cholesterol (mg/dL)", "Temp (°C)"]

    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(16, 4 * len(metrics)),
                             facecolor=STYLE["facecolor"])
    fig.suptitle("Chart 3 — Per-Patient Health Trends Over Time",
                 fontsize=15, fontweight="bold", color=PALETTE["navy"])

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = styled_ax(axes[i])
        ax.set_ylabel(label)
        ax.set_title(f"{label} Trend")

        for j, (pname, recs) in enumerate(patients.items()):
            if len(recs) < 1:
                continue
            # sort by date
            recs_s = sorted(recs, key=lambda x: x["RecordDate"])
            x_vals = list(range(1, len(recs_s) + 1))
            y_vals = [r[metric] for r in recs_s]
            col = CHART_COLORS[j % len(CHART_COLORS)]
            ax.plot(x_vals, y_vals, "o-", color=col, linewidth=2.2,
                    markersize=7, label=pname, zorder=3)
            ax.fill_between(x_vals, y_vals, alpha=0.07, color=col)
            # label last point
            ax.annotate(f"{y_vals[-1]:.1f}", (x_vals[-1], y_vals[-1]),
                        textcoords="offset points", xytext=(6, 2),
                        fontsize=8, color=col, fontweight="bold")

        ax.set_xlabel("Visit Number")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=9, framealpha=0.85, loc="upper right")

    fig.tight_layout()
    return save_fig(fig, "Chart3_Patient_Trends.jpg", out_dir)


# ── CHART 4 : Blood Pressure Analysis ────────────────────────────────────────
def chart_blood_pressure(records, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6),
                             facecolor=STYLE["facecolor"])
    fig.suptitle("Chart 4 — Blood Pressure Analysis",
                 fontsize=15, fontweight="bold", color=PALETTE["navy"])

    bp_data = []
    for r in records:
        s, d = parse_bp(r["BloodPressure"])
        if s is not None:
            bp_data.append({"Name": r["PatientName"], "sys": s, "dia": d,
                             "date": r["RecordDate"]})

    if not bp_data:
        for ax in axes:
            styled_ax(ax)
            ax.text(0.5, 0.5, "No BP data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color=PALETTE["grey"])
        fig.tight_layout()
        return save_fig(fig, "Chart4_BloodPressure.jpg", out_dir)

    names    = [d["Name"] for d in bp_data]
    systolic = [d["sys"]  for d in bp_data]
    diastolic= [d["dia"]  for d in bp_data]
    x        = np.arange(len(bp_data))

    # 4a – Grouped bar: systolic vs diastolic
    ax = styled_ax(axes[0])
    w = 0.38
    b1 = ax.bar(x - w/2, systolic,  w, label="Systolic",  color=PALETTE["red"],
                edgecolor="white", linewidth=1, alpha=0.88, zorder=3)
    b2 = ax.bar(x + w/2, diastolic, w, label="Diastolic", color=PALETTE["blue"],
                edgecolor="white", linewidth=1, alpha=0.88, zorder=3)
    for b, v in zip(list(b1)+list(b2), systolic+diastolic):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, str(int(v)),
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(140, color=PALETTE["red"],    linewidth=1.2, linestyle="--",
               alpha=0.7, label="High sys limit (140)")
    ax.axhline(90,  color=PALETTE["orange"], linewidth=1.2, linestyle="--",
               alpha=0.7, label="High dia limit (90)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("mmHg")
    ax.set_title("Systolic vs Diastolic BP")
    ax.legend(fontsize=8, framealpha=0.85)

    # 4b – Scatter: systolic vs diastolic coloured by risk zone
    ax2 = styled_ax(axes[1])
    ax2.grid(True, linestyle="--", alpha=0.35)
    for d in bp_data:
        if   d["sys"] >= 140 or d["dia"] >= 90:  zone_c = PALETTE["red"]
        elif d["sys"] >= 130 or d["dia"] >= 80:  zone_c = PALETTE["orange"]
        else:                                      zone_c = PALETTE["green"]
        ax2.scatter(d["sys"], d["dia"], c=zone_c, s=100, edgecolors="white",
                    linewidth=1, zorder=4)
        ax2.annotate(d["Name"], (d["sys"], d["dia"]),
                     textcoords="offset points", xytext=(5, 3),
                     fontsize=8, color="#333333")

    # risk-zone shading
    ax2.axhspan(0,  80, alpha=0.06, color=PALETTE["green"])
    ax2.axhspan(80, 90, alpha=0.06, color=PALETTE["orange"])
    ax2.axhspan(90, 200,alpha=0.06, color=PALETTE["red"])
    ax2.set_xlabel("Systolic (mmHg)")
    ax2.set_ylabel("Diastolic (mmHg)")
    ax2.set_title("BP Risk Zone Scatter")

    import matplotlib.patches as mpatches
    ax2.legend(handles=[
        mpatches.Patch(color=PALETTE["green"],  label="Normal"),
        mpatches.Patch(color=PALETTE["orange"], label="Elevated"),
        mpatches.Patch(color=PALETTE["red"],    label="High"),
    ], fontsize=8, framealpha=0.85)

    # 4c – Pulse pressure (sys - dia)
    ax3 = styled_ax(axes[2])
    pp = [s - d for s, d in zip(systolic, diastolic)]
    bar_clrs = [PALETTE["red"] if v > 60 else PALETTE["blue"] for v in pp]
    bars = ax3.bar(names, pp, color=bar_clrs, edgecolor="white",
                   linewidth=1.2, width=0.55, zorder=3)
    for b, v in zip(bars, pp):
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                 str(int(v)), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.axhline(60, color=PALETTE["red"], linewidth=1.3,
                linestyle="--", label="High PP limit (60)")
    ax3.set_ylabel("Pulse Pressure (mmHg)")
    ax3.set_title("Pulse Pressure per Record")
    ax3.legend(fontsize=9, framealpha=0.85)
    plt.setp(ax3.get_xticklabels(), rotation=20, ha="right", fontsize=9)

    fig.tight_layout()
    return save_fig(fig, "Chart4_BloodPressure.jpg", out_dir)


# ── CHART 5 : Health Risk Assessment Dashboard ───────────────────────────────
def chart_risk_assessment(records, out_dir):
    THRESHOLDS = {
        "BloodSugar":   {"normal": (70, 100), "concern": 126},
        "HeartRate":    {"normal": (60, 100),  "concern": 110},
        "SPO2":         {"normal": (95, 100),  "concern": 94},
        "Cholesterol":  {"normal": (0,  200),  "concern": 240},
        "Temperature":  {"normal": (36, 37.5), "concern": 38},
    }

    # Per patient, per metric: worst reading
    patients_set = sorted({r["PatientName"] for r in records})
    metric_keys  = list(THRESHOLDS.keys())

    # Build risk matrix
    risk_matrix = np.zeros((len(patients_set), len(metric_keys)))
    for pi, pname in enumerate(patients_set):
        precs = [r for r in records if r["PatientName"] == pname]
        for mi, mkey in enumerate(metric_keys):
            vals = [r[mkey] for r in precs]
            if not vals:
                continue
            lo, hi = THRESHOLDS[mkey]["normal"]
            concern = THRESHOLDS[mkey]["concern"]
            # risk score: 0=ok, 1=borderline, 2=concern
            worst = 0
            for v in vals:
                if v < lo or v > concern:
                    worst = max(worst, 2)
                elif v > hi:
                    worst = max(worst, 1)
            risk_matrix[pi, mi] = worst

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(patients_set) * 1.2 + 3)),
                             facecolor=STYLE["facecolor"])
    fig.suptitle("Chart 5 — Health Risk Assessment Dashboard",
                 fontsize=15, fontweight="bold", color=PALETTE["navy"])

    # 5a – Heatmap
    ax = axes[0]
    ax.set_facecolor(STYLE["panel_bg"])
    cmap = matplotlib.colors.ListedColormap([PALETTE["green"], PALETTE["orange"], PALETTE["red"]])
    im = ax.imshow(risk_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=2,
                   interpolation="nearest")

    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(["Sugar", "HR", "SPO2", "Chol", "Temp"],
                       fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(patients_set)))
    ax.set_yticklabels(patients_set, fontsize=10)
    ax.set_title("Risk Heat-Map\n(Green=OK, Orange=Borderline, Red=Concern)",
                 fontsize=11, fontweight="bold", color=PALETTE["navy"])

    # cell text
    for pi in range(len(patients_set)):
        for mi in range(len(metric_keys)):
            txt = ["✓", "!", "✗"][int(risk_matrix[pi, mi])]
            ax.text(mi, pi, txt, ha="center", va="center",
                    fontsize=13, fontweight="bold", color="white")

    # 5b – Overall risk score bar
    ax2 = styled_ax(axes[1])
    overall = risk_matrix.sum(axis=1)
    max_risk = len(metric_keys) * 2
    bar_clrs = []
    for v in overall:
        if v <= 1:            bar_clrs.append(PALETTE["green"])
        elif v <= max_risk/2: bar_clrs.append(PALETTE["orange"])
        else:                 bar_clrs.append(PALETTE["red"])

    bars = ax2.barh(patients_set, overall, color=bar_clrs,
                    edgecolor="white", linewidth=1.2, height=0.55, zorder=3)
    ax2.axvline(max_risk/2, color=PALETTE["red"], linewidth=1.5,
                linestyle="--", label=f"High risk threshold ({max_risk//2})")
    for b, v in zip(bars, overall):
        ax2.text(v + 0.05, b.get_y() + b.get_height()/2,
                 f"{v:.0f}/{max_risk}", va="center", fontsize=9, fontweight="bold")
    ax2.set_xlim(0, max_risk + 1.5)
    ax2.set_xlabel("Cumulative Risk Score")
    ax2.set_title("Overall Risk Score per Patient")
    ax2.legend(fontsize=9, framealpha=0.85)

    fig.tight_layout()
    return save_fig(fig, "Chart5_Risk_Assessment.jpg", out_dir)


# ── CHART 6 : Comparative Parameter Radar ────────────────────────────────────
def chart_radar_comparison(records, out_dir):
    patients = {}
    for r in records:
        patients.setdefault(r["PatientName"], []).append(r)

    metrics = ["BloodSugar", "HeartRate", "SPO2", "Cholesterol", "Temperature"]
    # Normalise each metric 0-1 using safe reference ranges
    norms = {
        "BloodSugar":  (50,  300),
        "HeartRate":   (40,  150),
        "SPO2":        (80,  100),
        "Cholesterol": (100, 350),
        "Temperature": (35,  40),
    }
    labels = ["Sugar", "Heart\nRate", "SPO2", "Chol", "Temp"]
    N = len(metrics)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True},
                           facecolor=STYLE["facecolor"])
    ax.set_facecolor("#F8FBFF")
    fig.suptitle("Chart 6 — Patient Comparison Radar\n(Normalised Health Parameters)",
                 fontsize=14, fontweight="bold", color=PALETTE["navy"], y=1.01)

    for i, (pname, recs) in enumerate(patients.items()):
        # average across visits
        avg = [np.mean([r[m] for r in recs]) for m in metrics]
        # normalise
        norm_vals = [(v - norms[m][0]) / (norms[m][1] - norms[m][0])
                     for v, m in zip(avg, metrics)]
        norm_vals = [max(0, min(1, v)) for v in norm_vals]
        norm_vals += norm_vals[:1]
        col = CHART_COLORS[i % len(CHART_COLORS)]
        ax.plot(angles, norm_vals, "o-", linewidth=2.2, color=col,
                label=pname, markersize=6)
        ax.fill(angles, norm_vals, alpha=0.10, color=col)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, color=PALETTE["navy"])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="grey")
    ax.grid(color="grey", alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=10, framealpha=0.9)

    fig.tight_layout()
    return save_fig(fig, "Chart6_Radar_Comparison.jpg", out_dir)


# ── CHART 7 : Session Run Statistics ─────────────────────────────────────────
def chart_session_statistics(users, records, out_dir):
    fig = plt.figure(figsize=(16, 10), facecolor=STYLE["facecolor"])
    fig.suptitle("Chart 7 — System Session Statistics",
                 fontsize=15, fontweight="bold", color=PALETTE["navy"])

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 7a – Records per patient pie
    ax1 = fig.add_subplot(gs[0, 0])
    ax1 = styled_ax(ax1); ax1.grid(False)
    counts_dict = {}
    for r in records:
        counts_dict[r["PatientName"]] = counts_dict.get(r["PatientName"], 0) + 1
    if counts_dict:
        ax1.pie(counts_dict.values(), labels=counts_dict.keys(),
                colors=CHART_COLORS[:len(counts_dict)], autopct="%1.0f%%",
                startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=2),
                textprops={"fontsize": 9})
    else:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax1.transAxes)
    ax1.set_title("Records Distribution\nby Patient")

    # 7b – Average vitals bar
    ax2 = fig.add_subplot(gs[0, 1])
    ax2 = styled_ax(ax2)
    metric_keys   = ["BloodSugar", "HeartRate", "SPO2", "Cholesterol", "Temperature"]
    metric_labels = ["Sugar", "HR", "SPO2", "Chol", "Temp"]
    avgs = [np.mean([r[k] for r in records]) if records else 0 for k in metric_keys]
    bars = ax2.bar(metric_labels, avgs, color=CHART_COLORS[:5],
                   edgecolor="white", linewidth=1.2, width=0.6, zorder=3)
    for b, v in zip(bars, avgs):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Average Value")
    ax2.set_title("Average Vitals Across\nAll Patients")

    # 7c – Visit frequency line
    ax3 = fig.add_subplot(gs[0, 2])
    ax3 = styled_ax(ax3)
    names_s = sorted(counts_dict.keys()) if counts_dict else []
    visits  = [counts_dict[n] for n in names_s]
    if names_s:
        ax3.plot(names_s, visits, "D-", color=PALETTE["purple"],
                 linewidth=2.5, markersize=10, markerfacecolor="white",
                 markeredgewidth=2.5, zorder=4)
        for n, v in zip(names_s, visits):
            ax3.text(n, v + 0.08, str(v), ha="center", va="bottom",
                     fontsize=10, fontweight="bold", color=PALETTE["navy"])
        ax3.set_ylim(0, max(visits) + 1.5)
    ax3.set_xlabel("Patient")
    ax3.set_ylabel("Number of Visits")
    ax3.set_title("Visit Frequency per Patient")
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.setp(ax3.get_xticklabels(), rotation=15, ha="right", fontsize=9)

    # 7d – Min/Max range bars
    ax4 = fig.add_subplot(gs[1, :])
    ax4 = styled_ax(ax4)
    if records:
        mins = [min(r[k] for r in records) for k in metric_keys]
        maxs = [max(r[k] for r in records) for k in metric_keys]
        x4 = np.arange(len(metric_keys))
        ax4.bar(x4, maxs, width=0.55, color=[c + "55" for c in CHART_COLORS[:5]],
                edgecolor="white", label="Max Observed", zorder=2)
        ax4.bar(x4, mins, width=0.55, color=CHART_COLORS[:5],
                edgecolor="white", label="Min Observed", zorder=3)
        for i, (mn, mx) in enumerate(zip(mins, maxs)):
            ax4.text(i, mx + 0.5, f"↑{mx:.0f}", ha="center", fontsize=8,
                     color="#333", fontweight="bold")
            ax4.text(i, mn - 2, f"↓{mn:.0f}", ha="center", fontsize=8,
                     color=CHART_COLORS[i], fontweight="bold", va="top")
        ax4.set_xticks(x4)
        ax4.set_xticklabels(["Blood Sugar\n(mg/dL)", "Heart Rate\n(bpm)",
                              "SPO2 (%)", "Cholesterol\n(mg/dL)",
                              "Temperature\n(°C)"], fontsize=9)
        ax4.set_ylabel("Value")
        ax4.set_title("Min / Max Range of Vital Parameters Observed")
        ax4.legend(fontsize=9, framealpha=0.85)
    else:
        ax4.text(0.5, 0.5, "No records available",
                 ha="center", va="center", transform=ax4.transAxes,
                 fontsize=14, color=PALETTE["grey"])

    return save_fig(fig, "Chart7_Session_Statistics.jpg", out_dir)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Health Monitoring System – Chart Generator")
    parser.add_argument("--dir", default=".",
                        help="Directory containing .mat files (default: current dir)")
    parser.add_argument("--out", default="HealthCharts_Output",
                        help="Output folder for JPG charts")
    parser.add_argument("--demo", action="store_true",
                        help="Force demo mode (ignore .mat files)")
    args = parser.parse_args()

    mat_dir = os.path.abspath(args.dir)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "═"*55)
    print("  Health Monitoring System – Chart Generator")
    print("═"*55)
    print(f"  Source dir : {mat_dir}")
    print(f"  Output dir : {out_dir}")
    print("─"*55)

    # ── Load data ──────────────────────────────────────────
    users   = [] if args.demo else load_users(mat_dir)
    master  = [] if args.demo else load_master_records(mat_dir)
    indiv   = [] if args.demo else load_individual_records(mat_dir)
    records = merge_records(master, indiv)

    demo_mode = args.demo or (len(users) == 0 and len(records) == 0)
    if demo_mode:
        print("  ⚠  No .mat files found – running with DEMO data.")
        users   = DEMO_USERS
        records = DEMO_RECORDS
    else:
        print(f"  ✔  Loaded {len(users)} user(s) and {len(records)} health record(s).")

    print("─"*55)
    print("  Generating charts …\n")

    saved_files = []

    # Run each chart function, catch individual failures gracefully
    generators = [
        ("Registration Overview",       chart_registration_overview, (users, records, out_dir)),
        ("Vital Signs Distribution",     chart_vital_distributions,   (records, out_dir)),
        ("Patient Trends",               chart_patient_trends,         (records, out_dir)),
        ("Blood Pressure Analysis",      chart_blood_pressure,         (records, out_dir)),
        ("Risk Assessment Dashboard",    chart_risk_assessment,        (records, out_dir)),
        ("Radar Comparison",             chart_radar_comparison,       (records, out_dir)),
        ("Session Statistics",           chart_session_statistics,     (users, records, out_dir)),
    ]

    for name, fn, fn_args in generators:
        try:
            path = fn(*fn_args)
            saved_files.append(path)
        except Exception as exc:
            print(f"  ✘  {name} failed: {exc}")

    # ── Summary manifest ───────────────────────────────────
    manifest = {
        "generated_at": timestamp_label(),
        "source_dir":   mat_dir,
        "demo_mode":    demo_mode,
        "patients":     len(users),
        "records":      len(records),
        "charts":       [os.path.basename(f) for f in saved_files],
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)

    print("\n" + "─"*55)
    print(f"  ✔  {len(saved_files)} chart(s) saved to: {out_dir}")
    print(f"  ✔  Manifest written  : {manifest_path}")
    print("═"*55 + "\n")


if __name__ == "__main__":
    main()

"""Explain the deployed Random Forest: SHAP (TreeExplainer), permutation importance and partial dependence.

Usage:  python model/explain.py path/to/AQI_Ready_For_Streamlit.csv [out_dir]
        out_dir defaults to docs/explainability

Rebuilds exactly the forest in model/model.json (same row order, split and hyperparameters as train.py),
checks it against the JSON the web app runs, then writes PNG charts and explain_results.json.
"""
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import train_test_split

CSV = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/explainability"
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

FEATURES = ["Temp", "Humidity", "PM2.5", "PM10", "NO2", "O3", "CO"]
COLMAP = {"Temp": "TempC", "Humidity": "Humidity", "PM2.5": "PM2.5", "PM10": "PM10",
          "NO2": "NO2 ppb", "O3": "O3 ppb", "CO": "CO ppm"}
UNITS = {"Temp": "°C", "Humidity": "%", "PM2.5": "µg/m³", "PM10": "µg/m³", "NO2": "ppb", "O3": "ppb", "CO": "ppm"}
RF_PARAMS = dict(n_estimators=40, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)

INK, MUTED, GRID, ACC, ACC2 = "#10212a", "#5b6b73", "#e3e9ec", "#0a6782", "#52c2dc"
AQI_COLORS = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11, "axes.edgecolor": "#c9d3d8", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 180,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- data + model (identical to train.py)
df = pd.read_csv(CSV)
df["TIME"] = pd.to_datetime(df["TIME"])
df = df.sort_values("TIME", kind="stable").reset_index(drop=True)
X = df[[COLMAP[f] for f in FEATURES]].to_numpy(dtype=float)
y = df["AQI"].to_numpy(dtype=float)
tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
rf = RandomForestRegressor(**RF_PARAMS).fit(X[tr], y[tr])
Xte, yte = X[te], y[te]


def predict_json(m, x):
    s = 0.0
    for t in m["trees"]:
        i = 0
        while t["l"][i] != -1:
            i = t["l"][i] if x[t["f"][i]] <= t["t"][i] else t["r"][i]
        s += t["v"][i]
    return s / len(m["trees"])


model_json = json.load(open(os.path.join(HERE, "model.json")))
sel = np.linspace(0, len(te) - 1, 500).astype(int)
diff = np.abs(rf.predict(Xte[sel]) - np.array([predict_json(model_json, x) for x in Xte[sel]]))
assert diff.max() < 0.01, f"rebuilt forest differs from model.json (max diff {diff.max()})"
print(f"forest matches model.json on 500 test rows (max diff {diff.max():.5f})")
res = {"model_check_max_abs_diff": round(float(diff.max()), 6), "n_test": int(len(te))}

# ---------------------------------------------------------------- SHAP
rng = np.random.default_rng(42)
idx = rng.choice(len(te), size=2000, replace=False)
Xs = Xte[idx]
explainer = shap.TreeExplainer(rf)
sv = explainer.shap_values(Xs)
base = float(np.ravel(explainer.expected_value)[0])
additivity = np.abs(base + sv.sum(axis=1) - rf.predict(Xs)).max()
mean_abs = np.abs(sv).mean(axis=0)
res["shap"] = {"rows": 2000, "base_value": round(base, 3), "additivity_max_error": float(additivity),
               "mean_abs": {f: round(float(v), 4) for f, v in zip(FEATURES, mean_abs)},
               "range": {f: [round(float(sv[:, i].min()), 2), round(float(sv[:, i].max()), 2)] for i, f in enumerate(FEATURES)}}

expl = shap.Explanation(values=sv, base_values=np.full(len(Xs), base), data=Xs, feature_names=FEATURES)
plt.figure()
shap.plots.beeswarm(expl, max_display=7, show=False, color_bar_label="Feature value (low → high)")
fig = plt.gcf()
fig.set_size_inches(9.5, 5.2)
ax = plt.gca()
ax.set_title("SHAP summary: how each reading pushes the AQI prediction", pad=14)
ax.set_xlabel("SHAP value (change in predicted AQI, AQI points)")
save(fig, "shap_beeswarm.png")

order = np.argsort(mean_abs)
fig, ax = plt.subplots(figsize=(9.5, 4.6))
cols = [ACC if FEATURES[i] in ("PM2.5", "PM10") else "#9fb2ba" for i in order]
bars = ax.barh([FEATURES[i] for i in order], mean_abs[order], color=cols, height=0.62)
for b, v in zip(bars, mean_abs[order]):
    ax.text(b.get_width() + mean_abs.max() * 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", color=INK, fontsize=10.5)
ax.set_title("Mean |SHAP|: average impact on the predicted AQI", pad=12)
ax.set_xlabel("Mean |SHAP value| (AQI points)")
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
ax.set_xlim(0, mean_abs.max() * 1.12)
save(fig, "shap_importance.png")

pm25 = FEATURES.index("PM2.5")
pm10 = FEATURES.index("PM10")
fig, ax = plt.subplots(figsize=(9.5, 5))
scat = ax.scatter(Xs[:, pm25], sv[:, pm25], c=Xs[:, pm10], cmap="viridis", s=10, alpha=0.75, linewidths=0)
cb = fig.colorbar(scat, ax=ax, pad=0.01)
cb.set_label("PM10 (µg/m³)", color=MUTED)
ax.axhline(0, color="#c9d3d8", lw=1)
ax.set_title("SHAP dependence: PM2.5 drives the prediction up as it rises", pad=12)
ax.set_xlabel("PM2.5 (µg/m³)")
ax.set_ylabel("SHAP value for PM2.5 (AQI points)")
ax.grid(color=GRID)
ax.set_axisbelow(True)
save(fig, "shap_dependence_pm25.png")

# ---------------------------------------------------------------- permutation importance (whole test set)
pi = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=42, n_jobs=-1, scoring="r2")
res["permutation_r2_drop"] = {f: {"mean": round(float(m), 4), "std": round(float(s), 4)}
                              for f, m, s in zip(FEATURES, pi.importances_mean, pi.importances_std)}
order = np.argsort(pi.importances_mean)
fig, ax = plt.subplots(figsize=(9.5, 4.6))
cols = [ACC if FEATURES[i] in ("PM2.5", "PM10") else "#9fb2ba" for i in order]
bars = ax.barh([FEATURES[i] for i in order], pi.importances_mean[order], xerr=pi.importances_std[order],
               color=cols, height=0.62, error_kw={"ecolor": INK, "capsize": 3, "lw": 1})
for b, v in zip(bars, pi.importances_mean[order]):
    ax.text(max(b.get_width(), 0) + pi.importances_mean.max() * 0.015, b.get_y() + b.get_height() / 2,
            f"{abs(v) if abs(v) < 0.0005 else v:.3f}", va="center", color=INK, fontsize=10.5)
ax.set_title(f"Permutation importance: R² lost when a feature is shuffled ({len(te):,} test rows, 10 repeats)", pad=12)
ax.set_xlabel("Drop in R²")
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
ax.set_xlim(0, pi.importances_mean.max() * 1.15)
save(fig, "permutation_importance.png")

# ---------------------------------------------------------------- partial dependence + ICE


def epa_pm25_pre2024(c):
    rows = [(0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)]
    c = math.floor(c * 10 + 1e-6) / 10
    for lo, hi, ilo, ihi in rows:
        if c <= hi:
            return (ihi - ilo) / (hi - lo) * (max(c, lo) - lo) + ilo
    return 500


Xp = Xte[rng.choice(len(te), size=1500, replace=False)]
feats = ["PM2.5", "PM10", "Temp", "Humidity"]
fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
disp = PartialDependenceDisplay.from_estimator(
    rf, Xp, [FEATURES.index(f) for f in feats], feature_names=FEATURES, kind="both", subsample=120,
    grid_resolution=60, random_state=42, ax=axes.ravel(),
    ice_lines_kw={"color": ACC2, "alpha": 0.12, "linewidth": 0.6}, pd_line_kw={"color": ACC, "linewidth": 2.6})
res["pdp"] = {}
for k, (a, f) in enumerate(zip(disp.axes_.ravel(), feats)):
    grid = disp.pd_results[k]["grid_values"][0]
    avg = disp.pd_results[k]["average"][0]
    res["pdp"][f] = {"grid_min": round(float(grid.min()), 2), "grid_max": round(float(grid.max()), 2),
                     "pd_min": round(float(avg.min()), 2), "pd_max": round(float(avg.max()), 2)}
    a.set_xlabel(f"{f} ({UNITS[f]})")
    a.set_ylabel("Predicted AQI" if k % 2 == 0 else "")
    a.grid(color=GRID)
    a.set_axisbelow(True)
    leg = a.get_legend()
    if leg:
        leg.remove()
    if f == "PM2.5":
        a.plot(grid, [epa_pm25_pre2024(v) for v in grid], ls="--", color="#c0392b", lw=1.6, label="EPA formula (PM2.5 only)")
        a.legend(loc="upper left", fontsize=9.5, frameon=False)
fig.suptitle("Partial dependence (thick line) and ICE curves (thin lines)", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.tight_layout()
save(fig, "pdp_ice.png")

json.dump(res, open(os.path.join(OUT, "explain_results.json"), "w"), indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2, ensure_ascii=False))

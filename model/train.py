"""Train + evaluate the AQI Random Forest, export compact JSON, verify it.

Usage: python -I train.py <csv_path> <out_dir>
Writes: model.json, metrics.json, verify_rows.json (for the JS check) into out_dir.
"""
import json, math, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

CSV, OUT = sys.argv[1], sys.argv[2]
FEATURES = ["Temp", "Humidity", "PM2.5", "PM10", "NO2", "O3", "CO"]
COLMAP = {"Temp": "TempC", "Humidity": "Humidity", "PM2.5": "PM2.5", "PM10": "PM10",
          "NO2": "NO2 ppb", "O3": "O3 ppb", "CO": "CO ppm"}
RF_PARAMS = dict(n_estimators=40, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)

df = pd.read_csv(CSV)
df["TIME"] = pd.to_datetime(df["TIME"])
df = df.sort_values("TIME", kind="stable").reset_index(drop=True)
X = df[[COLMAP[f] for f in FEATURES]].to_numpy(dtype=float)
y = df["AQI"].to_numpy(dtype=float)


def metrics(y_true, y_pred):
    return {"R2": round(float(r2_score(y_true, y_pred)), 4),
            "RMSE": round(float(math.sqrt(mean_squared_error(y_true, y_pred))), 3),
            "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
            "n_test": int(len(y_true))}


def fit_eval(Xtr, ytr, Xte, yte, model=None):
    m = model or RandomForestRegressor(**RF_PARAMS)
    m.fit(Xtr, ytr)
    return m, metrics(yte, m.predict(Xte))


res = {"n_rows": int(len(df)), "features": FEATURES, "rf_params": {k: v for k, v in RF_PARAMS.items() if k != "n_jobs"}}

# 1) random 80/20 split (what the thesis-style evaluation does)
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.2, random_state=42)
rf, res["random_split"] = fit_eval(X[tr], y[tr], X[te], y[te])

# 2) time-ordered split: first 80% of rows by TIME train, last 20% test
cut = int(len(df) * 0.8)
_, res["time_split"] = fit_eval(X[:cut], y[:cut], X[cut:], y[cut:])
res["time_split"]["train_until"] = str(df.TIME.iloc[cut - 1])
res["time_split"]["test_from"] = str(df.TIME.iloc[cut])

# 3) day split: train 3 Dec, test 4 Dec
d = df.TIME.dt.date.astype(str).to_numpy()
days = sorted(set(d))
m1, m2 = d == days[0], d == days[1]
_, res["day_split"] = fit_eval(X[m1], y[m1], X[m2], y[m2])
res["day_split"]["train_day"], res["day_split"]["test_day"] = days[0], days[1]

# 4) random split after removing exact duplicate rows (leakage check)
dd = df.drop_duplicates()
Xd = dd[[COLMAP[f] for f in FEATURES]].to_numpy(dtype=float); yd = dd["AQI"].to_numpy(dtype=float)
a, b = train_test_split(np.arange(len(dd)), test_size=0.2, random_state=42)
_, res["random_split_dedup"] = fit_eval(Xd[a], yd[a], Xd[b], yd[b])
res["random_split_dedup"]["n_rows_after_dedup"] = int(len(dd))

# 5) model without PM features (how much do gases + weather alone explain?)
nopm = [0, 1, 4, 5, 6]
_, res["time_split_no_pm"] = fit_eval(X[:cut][:, nopm], y[:cut], X[cut:][:, nopm], y[cut:])

# 6) LightGBM (default params) for comparison with the thesis claim
try:
    import lightgbm as lgb
    p = dict(random_state=42, verbose=-1)
    _, res["lgbm_random_split"] = fit_eval(X[tr], y[tr], X[te], y[te], lgb.LGBMRegressor(**p))
    _, res["lgbm_time_split"] = fit_eval(X[:cut], y[:cut], X[cut:], y[cut:], lgb.LGBMRegressor(**p))
except ImportError:
    res["lgbm"] = "lightgbm not installed"

# importances of the exported (random-split) model
res["importance_impurity"] = {f: round(float(v), 4) for f, v in zip(FEATURES, rf.feature_importances_)}
pi = permutation_importance(rf, X[te][:5000], y[te][:5000], n_repeats=3, random_state=42, n_jobs=-1)
res["importance_permutation_R2_drop"] = {f: round(float(v), 4) for f, v in zip(FEATURES, pi.importances_mean)}

# training range (the browser flags inputs outside it)
res["train_range"] = {f: [round(float(X[tr][:, i].min()), 3), round(float(X[tr][:, i].max()), 3)] for i, f in enumerate(FEATURES)}
res["train_target_range"] = [float(y[tr].min()), float(y[tr].max())]

# extrapolation probes
probes = {
    "thesis_case_study (T25,H50,PM2.5 20,PM10 30,NO2 10,O3 10,CO 0.5)": [25, 50, 20, 30, 10, 10, 0.5],
    "clean_air (T25,H50,PM2.5 5,PM10 10,NO2 5,O3 20,CO 0.3)": [25, 50, 5, 10, 5, 20, 0.3],
    "extreme (T30,H60,PM2.5 450,PM10 550,NO2 40,O3 2,CO 1)": [30, 60, 450, 550, 40, 2, 1],
}
res["probes"] = {k: round(float(rf.predict(np.array([v]))[0]), 2) for k, v in probes.items()}

# ---- export compact JSON ----
def r(x, nd):
    v = round(float(x), nd)
    return int(v) if v == int(v) else v

trees = []
for est in rf.estimators_:
    t = est.tree_
    leaf = t.children_left == -1
    trees.append({
        "f": [int(v) if not lf else -2 for v, lf in zip(t.feature, leaf)],
        "t": [0 if lf else r(v, 4) for v, lf in zip(t.threshold, leaf)],
        "l": [int(v) for v in t.children_left],
        "r": [int(v) for v in t.children_right],
        "v": [r(v, 2) if lf else r(v, 1) for v, lf in zip(t.value[:, 0, 0], leaf)],
    })
model = {"features": FEATURES, "units": ["degC", "%", "ug/m3", "ug/m3", "ppb", "ppb", "ppm"],
         "target": "AQI (US EPA, pre-2024 PM2.5 table, as in the thesis CSV)", "trees": trees}
js = json.dumps(model, separators=(",", ":"))
open(f"{OUT}/model.json", "w").write(js)
res["model_json_bytes"] = len(js)
res["n_nodes_total"] = int(sum(len(t["f"]) for t in trees))


def predict_json(m, x):
    s = 0.0
    for t in m["trees"]:
        i = 0
        while t["l"][i] != -1:
            i = t["l"][i] if x[t["f"][i]] <= t["t"][i] else t["r"][i]
        s += t["v"][i]
    return s / len(m["trees"])

# verify on the whole test set in Python
m2 = json.loads(js)
sk = rf.predict(X[te])
pj = np.array([predict_json(m2, x) for x in X[te]])
diff = np.abs(sk - pj)
res["verify_python"] = {"rows": int(len(te)), "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}

# 300 rows for the browser/JS check (spread across the test set, plus extremes)
sel = np.linspace(0, len(te) - 1, 300).astype(int)
rows = X[te][sel]
json.dump({"X": rows.tolist(), "sklearn": rf.predict(rows).tolist(), "aqi_true": y[te][sel].tolist()},
          open(f"{OUT}/verify_rows.json", "w"))

json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=2)
print(json.dumps(res, indent=2))

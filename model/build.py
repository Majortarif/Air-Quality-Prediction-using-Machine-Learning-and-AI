"""Assemble the single-file site: embed model.json + metadata (all numbers from the training run / CSV).

Usage: python build.py <csv_path> <model_dir> [out_dir]
Reads model.json + metrics.json from <model_dir>; writes index.html to out_dir (default: model_dir).
"""
import json, sys
import pandas as pd

CSV, BUILD = sys.argv[1], sys.argv[2]
OUTDIR = sys.argv[3] if len(sys.argv) > 3 else BUILD
SRC = __file__.replace("build.py", "app.src.html")
metrics = json.load(open(f"{BUILD}/metrics.json"))
model_js = open(f"{BUILD}/model.json").read()

df = pd.read_csv(CSV)
df["TIME"] = pd.to_datetime(df["TIME"])
day = df.TIME.dt.date
spans = [f"{d:%d %b %Y} {g.TIME.min():%H:%M}–{g.TIME.max():%H:%M}" for d, g in df.groupby(day)]
hours = sum((g.TIME.max() - g.TIME.min()).total_seconds() for _, g in df.groupby(day)) / 3600
dups = int(df.duplicated().sum())
cat = df.Category.value_counts()
pct = (cat / len(df) * 100).round(2)
sub = df[["AQI_PM25", "AQI_PM10", "AQI_NO2", "AQI_O3", "AQI_CO"]]
dom = sub.idxmax(axis=1).value_counts()
co_glitch = int((df["CO ppm"] > 50).sum())
per_min = df.groupby("TIME").size().median()
cols = {"temp": "TempC", "hum": "Humidity", "pm25": "PM2.5", "pm10": "PM10", "no2": "NO2 ppb", "o3": "O3 ppb", "co": "CO ppm"}
medians = {k: round(float(df[c].median()), 3) for k, c in cols.items()}

findings = [
    f"Only {hours:.1f} hours of data: {spans[0]} and {spans[1]}. The thesis describes 35 days, 20 days and five days in different places.",
    f"{dups:,} of {len(df):,} rows ({dups / len(df) * 100:.1f}%) are exact duplicates. The log holds about {per_min:.0f} rows per minute, so neighbouring rows are near copies; a random train/test split puts almost the same rows on both sides.",
    "Only three AQI categories occur: " + ", ".join(f"{k} {pct[k]}% ({cat[k]:,} rows)" for k in cat.index) + ". The model has never seen Good, Moderate or Unhealthy for Sensitive Groups air.",
    f"The AQI column equals the maximum EPA sub-index in every row, and PM2.5 is the dominant pollutant in {dom.get('AQI_PM25', 0):,} rows. The target is a formula of the inputs.",
    f"{co_glitch} rows have CO between {df.loc[df['CO ppm'] > 50, 'CO ppm'].min():,.0f} and {df['CO ppm'].max():,.0f} ppm (median {df['CO ppm'].median():.2f} ppm). These sensor glitches are the only reason those rows are labelled Hazardous on 3 December; outlier capping described in the thesis was not applied to this file.",
    f"NO2 reaches {df['NO2 ppb'].max():.0f} ppb while its median is {df['NO2 ppb'].median():.1f} ppb, another sign of uncalibrated spikes.",
    "The CSV has no site column, so Badda and Aftabnagar readings cannot be separated or compared.",
    "The xlsx files in Air_quality_dataset.zip have no PM or time columns, report CO around 18–61 ppm (about 30 times the CSV) and CO2 of 1–8 'ppm' (outdoor air is about 420 ppm), so their units or calibration are inconsistent with the CSV.",
]

meta = {
    **{k: metrics[k] for k in ["rf_params", "random_split", "time_split", "day_split", "random_split_dedup", "time_split_no_pm",
                               "lgbm_random_split", "lgbm_time_split", "importance_impurity", "train_range",
                               "train_target_range", "n_nodes_total"]},
    "n_train": int(round(metrics["n_rows"] * 0.8)),
    "medians": medians,
    "data": {"file": "AQI_Ready_For_Streamlit.csv", "n_rows": int(len(df)), "spans": spans, "hours": round(hours, 1),
             "duplicates": dups, "category_pct": pct.to_dict(), "category_counts": {k: int(v) for k, v in cat.items()},
             "dominant": {k: int(v) for k, v in dom.items()}, "co_glitch_rows": co_glitch, "findings": findings},
}
json.dump(meta, open(f"{BUILD}/meta.json", "w"), indent=2, ensure_ascii=False)

src = open(SRC, encoding="utf-8").read()
frag = src.replace("/*__MODEL__*/", model_js).replace("/*__META__*/", json.dumps(meta, ensure_ascii=False, separators=(",", ":")))
assert "/*__" not in frag
head, body = frag.split("<!--/HEAD-->")
index = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
         "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
         f"{head.strip()}\n</head>\n<body>\n{body.strip()}\n</body>\n</html>\n")
open(f"{OUTDIR}/index.html", "w", encoding="utf-8").write(index)
print("index.html bytes:", len(index.encode()), " findings:", len(findings))
print(json.dumps({k: meta["data"][k] for k in ["spans", "hours", "duplicates", "category_pct", "dominant", "co_glitch_rows"]}, indent=1))
print("medians", medians)

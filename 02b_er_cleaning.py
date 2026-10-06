# -*- coding: utf-8 -*-
"""
02b — تنظيف بيانات الطوارئ (Hospital_ER_Data) + تسلسل ساعي للوصولات.
مخرجات: hospital_er_cleaned.csv، hospital_er_hourly_series.csv
"""
import json

import numpy as np
import pandas as pd

from _paths import ER_CLEAN_CSV, ER_CLEANING_LOG, ER_HOURLY_CSV, PROCESSED_DIR, ensure_er_raw, RAW_ER_CSV


def main():
    ensure_er_raw()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    decisions: list[dict] = []

    df = pd.read_csv(RAW_ER_CSV, low_memory=False)
    n0 = len(df)
    df = df.replace({"?": np.nan, "None": np.nan, "": np.nan, "nan": np.nan})
    decisions.append({"step": "replace_tokens", "n_rows": n0})

    df = df.rename(
        columns={
            "Patient Id": "patient_id",
            "Patient Admission Date": "patient_admission_raw",
            "Patient First Inital": "patient_first_initial",
            "Patient Last Name": "patient_last_name",
            "Patient Gender": "patient_gender",
            "Patient Age": "patient_age",
            "Patient Race": "patient_race",
            "Department Referral": "department_referral",
            "Patient Admission Flag": "patient_admission_flag",
            "Patient Satisfaction Score": "patient_satisfaction_score",
            "Patient Waittime": "patient_waittime",
            "Patients CM": "patients_cm",
        }
    )

    df["admission_ts"] = pd.to_datetime(df["patient_admission_raw"], dayfirst=True, errors="coerce")
    bad_dt = df["admission_ts"].isna().sum()
    df = df.dropna(subset=["admission_ts"])
    decisions.append({"step": "parse_admission_datetime", "dropped_invalid_ts": int(bad_dt)})

    df["patient_waittime"] = pd.to_numeric(df["patient_waittime"], errors="coerce")
    df["patient_satisfaction_score"] = pd.to_numeric(df["patient_satisfaction_score"], errors="coerce")
    df["patient_age"] = pd.to_numeric(df["patient_age"], errors="coerce")

    flag = df["patient_admission_flag"].astype(str).str.strip().str.upper()
    df["patient_admission_flag"] = flag.isin(["TRUE", "1", "YES"]).astype(int)

    df["patients_cm"] = pd.to_numeric(df["patients_cm"], errors="coerce").fillna(0).astype(int)

    for c in ["patient_gender", "patient_race", "department_referral"]:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"nan": np.nan, "None": np.nan})

    # صف جاهز للـ ML (بدون معرّف المريض أو الاسم)
    ml_cols = [
        "admission_ts",
        "patient_gender",
        "patient_age",
        "patient_race",
        "department_referral",
        "patient_admission_flag",
        "patient_satisfaction_score",
        "patient_waittime",
        "patients_cm",
    ]
    out_ml = df[ml_cols].copy()
    out_ml["admission_ts"] = out_ml["admission_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out_ml.to_csv(ER_CLEAN_CSV, index=False)
    decisions.append({"step": "save_er_clean", "path": str(ER_CLEAN_CSV), "n_rows": len(out_ml)})

    # تسلسل ساعي: عدد الوصولات ومتوسط وقت الانتظار كمؤشر ضغط
    df["hour_bucket"] = df["admission_ts"].dt.floor("h")
    hourly = (
        df.groupby("hour_bucket", sort=True)
        .agg(arrivals=("patient_id", "count"), mean_waittime=("patient_waittime", "mean"))
        .reset_index()
    )
    t_min, t_max = hourly["hour_bucket"].min(), hourly["hour_bucket"].max()
    full_idx = pd.date_range(t_min, t_max, freq="h")
    full = hourly.set_index("hour_bucket").reindex(full_idx)
    full.index.name = "hour_start"
    full = full.reset_index()
    full["arrivals"] = full["arrivals"].fillna(0).astype(int)
    full["mean_waittime"] = full["mean_waittime"].fillna(0.0)
    full["hour_start"] = pd.to_datetime(full["hour_start"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    full.to_csv(ER_HOURLY_CSV, index=False)
    decisions.append(
        {
            "step": "save_hourly_series",
            "path": str(ER_HOURLY_CSV),
            "n_hours": len(full),
            "total_arrivals": int(full["arrivals"].sum()),
        }
    )

    ER_CLEANING_LOG.write_text(
        json.dumps({"decisions": decisions, "columns_clean": ml_cols}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Wrote:", ER_CLEAN_CSV)
    print("Wrote:", ER_HOURLY_CSV)
    print("Wrote:", ER_CLEANING_LOG)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
02 — Data Cleaning
توثيق القرارات: NA، تكرار، أعمدة شاذة، تجميع نادر، عمود الهدف الثنائي.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import CLEAN_CSV, OUTPUT_DIR, PROCESSED_DIR, RAW_CSV, ensure_raw_from_data_folder

LOG_PATH = PROCESSED_DIR / "cleaning_report.md"
LOG_JSON = PROCESSED_DIR / "cleaning_log.json"


def log_entry(log: list, msg: str):
    log.append(msg)


def collapse_rare_categories(s: pd.Series, min_freq: float = 0.01) -> pd.Series:
    vc = s.value_counts(normalize=True, dropna=False)
    rare = vc[vc < min_freq].index
    if len(rare) == 0:
        return s
    out = s.copy()
    out = out.where(~out.isin(rare), other="Other_rare")
    return out


def main():
    ensure_raw_from_data_folder()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    decisions: list[dict] = []

    df = pd.read_csv(RAW_CSV, low_memory=False)
    log_entry(log, f"1) تحميل خام: {len(df):,} صف، {len(df.columns)} عمود.")

    df = df.replace({"?": np.nan, "None": np.nan, "": np.nan})
    log_entry(log, "2) استبدال الرموز ?, None, فراغ بنقص (NaN).")
    decisions.append({"step": "replace_tokens", "detail": "? / None / empty -> NaN"})

    dup_enc = df["encounter_id"].duplicated().sum()
    if dup_enc > 0:
        df = df.drop_duplicates(subset=["encounter_id"], keep="first")
        log_entry(log, f"3) حذف {dup_enc} زيارة مكررة (encounter_id) — الإبقاء على أول ظهور.")
        decisions.append({"step": "drop_duplicate_encounter_id", "removed_rows": int(dup_enc)})
    else:
        log_entry(log, "3) لا تكرار على encounter_id.")

    df["readmitted"] = df["readmitted"].astype(str).str.strip()
    df["readmit_lt30"] = (df["readmitted"] == "<30").astype(int)
    pos_rate = df["readmit_lt30"].mean()
    log_entry(log, f"4) عمود readmit_lt30: نسبة الفئة الموجبة ≈ {pos_rate:.3%}.")
    decisions.append({"step": "target_binary", "rule": "readmit_lt30 = 1 iff readmitted == '<30'"})

    miss_w = df["weight"].isna().mean()
    df = df.drop(columns=["weight"])
    log_entry(log, f"5) حذف عمود weight (نسبة فقد ≈ {miss_w:.1%} عالية).")
    decisions.append({"step": "drop_weight", "reason": "high_missing_unstable"})

    log_entry(log, "6) الإبقاء على max_glu_serum و A1Cresult للتعويض في خط أنابيب التدريب.")

    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            pre = df[col].astype(str).str.slice(0, 3)
            df[col + "_grp"] = collapse_rare_categories(pre, 0.005)
            df = df.drop(columns=[col])
            df.rename(columns={col + "_grp": col}, inplace=True)
    log_entry(log, "7) تجميع نادر لبادئات diag_1/2/3 (أول 3 أحرف) بحد 0.5% — النادر -> Other_rare.")
    decisions.append({"step": "collapse_diag_prefix", "threshold": 0.005})

    if "race" in df.columns:
        vc = df["race"].value_counts(normalize=True)
        small = vc[vc < 0.02].index
        df["race"] = df["race"].where(~df["race"].isin(small), other="Other")
        log_entry(log, "8) تجميع فئات race الأقل من 2% تحت Other.")

    for col in ("medical_specialty", "payer_code"):
        if col in df.columns:
            df[col] = collapse_rare_categories(df[col].astype(str), 0.01)
            log_entry(log, f"8b) تجميع نادر {col} (أقل من 1%) -> Other_rare.")
            decisions.append({"step": f"collapse_rare_{col}", "threshold": 0.01})

    if "num_lab_procedures" in df.columns and "num_medications" in df.columns:
        nl = pd.to_numeric(df["num_lab_procedures"], errors="coerce").fillna(0).clip(lower=0)
        nm = pd.to_numeric(df["num_medications"], errors="coerce").fillna(0).clip(lower=0)
        df["fea_log1p_labs_meds"] = np.log1p(nl * nm)
        log_entry(log, "8c) ميزة fea_log1p_labs_meds = log1p(num_lab_procedures * num_medications).")
        decisions.append({"step": "feature_log1p_labs_meds"})

    if all(c in df.columns for c in ("number_outpatient", "number_emergency", "number_inpatient")):
        no = pd.to_numeric(df["number_outpatient"], errors="coerce").fillna(0).clip(lower=0)
        ne = pd.to_numeric(df["number_emergency"], errors="coerce").fillna(0).clip(lower=0)
        ni = pd.to_numeric(df["number_inpatient"], errors="coerce").fillna(0).clip(lower=0)
        tot = no + ne + ni
        df["fea_log1p_visit_load"] = np.log1p(tot)
        df["fea_emergency_share"] = (ne / (1.0 + tot)).astype(float)
        df["fea_inpatient_any"] = (ni > 0).astype(int)
        log_entry(log, "8d) ميزات fea_log1p_visit_load, fea_emergency_share, fea_inpatient_any.")
        decisions.append({"step": "features_visit_history"})

    if "time_in_hospital" in df.columns:
        tih = pd.to_numeric(df["time_in_hospital"], errors="coerce").fillna(0)
        df["fea_long_stay"] = (tih >= 7).astype(int)
        log_entry(log, "8e) fea_long_stay: إقامة >= 7 أيام.")
        decisions.append({"step": "feature_long_stay_ge7"})

    df.to_csv(CLEAN_CSV, index=False)
    log_entry(log, f"9) حفظ البيانات المنظّفة: {CLEAN_CSV} ({len(df):,} صف).")

    with open(LOG_JSON, "w", encoding="utf-8") as f:
        json.dump({"decisions": decisions, "n_rows_final": len(df), "columns_final": list(df.columns)}, f, ensure_ascii=False, indent=2)

    md = ["# Data Cleaning Report", "", "## قرارات منفّذة", "", *[f"- {x}" for x in log], ""]
    LOG_PATH.write_text("\n".join(md), encoding="utf-8")
    print("Wrote:", CLEAN_CSV)
    print("Wrote:", LOG_PATH)
    print("Wrote:", LOG_JSON)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
01 — Dataset Description (Diabetes 130-U.S. hospitals)
يولّد وصفاً منظماً: مصدر، فترة، أبعاد، مستوى التحليل (زيارة)، تعريف الهدف.
"""
import json
from pathlib import Path

import pandas as pd

from _paths import OUTPUT_DIR, RAW_CSV, ensure_raw_from_data_folder

# UCI official page (citation source)
SOURCE_URL = "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008"
SOURCE_NAME = "UCI ML Repository — Diabetes 130-US hospitals (1999–2008)"
TIME_RANGE = "1999–2008 (as documented by UCI for this extract)"
LEVEL = "encounter (hospital visit); multiple encounters may map to the same patient_nbr"


def main():
    ensure_raw_from_data_folder()
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV, na_values=["?", "None", ""], low_memory=False)
    n_rows, n_cols = len(df), len(df.columns)

    missing_pct = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    top_missing = missing_pct[missing_pct > 0].head(15)

    readmitted_dist = df["readmitted"].value_counts(dropna=False).to_dict()

    key_columns = {
        "encounter_id": "معرّف فريد للزيارة (Encounter).",
        "patient_nbr": "معرّف المريض — قد تتكرر عدة زيارات لنفس المريض.",
        "race": "العرق.",
        "gender": "الجنس.",
        "age": "الفئة العمرية كنطاق (مثل [50-60)).",
        "time_in_hospital": "أيام الإقامة في المستشفى لهذه الزيارة.",
        "num_lab_procedures": "عدد إجراءات المختبر.",
        "num_medications": "عدد الأدوية المسجّلة.",
        "diag_1": "التشخيص الأول (كود ICD تقريبي/نصي).",
        "readmitted": "الهدف الأصلي: NO أو <30 أو >30 (إعادة قبول خلال 30 يوماً).",
    }

    target_binary_def = (
        "للتنبؤ الثنائي في خطوات التنظيف/التدريب: readmit_lt30 = 1 إذا readmitted == '<30' وإلا 0. "
        "هذا يمثّل إعادة قبول مبكّرة مقابل غير ذلك."
    )

    doc = {
        "source_name": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "time_range_documented": TIME_RANGE,
        "analysis_level": LEVEL,
        "paths": {
            "raw_csv_relative": "Dataset/raw/diabetic_data.csv",
            "processed_clean_relative": "Dataset/processed/diabetic_data_cleaned.csv",
            "raw_note_ar": "يُفضّل عدم تعديل الملف في raw يدوياً؛ التعديلات في سكربت التنظيف فقط.",
        },
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "column_names": list(df.columns),
        "key_columns_ar": key_columns,
        "readmitted_distribution_raw": {str(k): int(v) for k, v in readmitted_dist.items()},
        "target_binary_definition_ar": target_binary_def,
        "missing_pct_top15": {str(k): float(v) for k, v in top_missing.items()},
        "duplicate_encounter_ids": int(df["encounter_id"].duplicated().sum()),
    }

    out_json = OUTPUT_DIR / "dataset_description.json"
    out_md = OUTPUT_DIR / "dataset_description.md"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# Dataset Description — Diabetes 130",
        "",
        f"**المصدر:** {SOURCE_NAME}",
        f"**الرابط:** {SOURCE_URL}",
        f"**الفترة الزمنية (حسب وصف UCI):** {TIME_RANGE}",
        f"**مستوى السجل:** {LEVEL}",
        "",
        "**هيكل المجلدات:** الخام في `Dataset/raw/` — بعد التنظيف في `Dataset/processed/` (يُنشأ تلقائياً عند تشغيل `02_data_cleaning.py`).",
        "",
        f"**عدد الصفوف:** {n_rows:,}",
        f"**عدد الأعمدة:** {n_cols}",
        "",
        "## توزيع الهدف الأصلي (readmitted)",
        "",
        *[f"- `{k}`: {v:,}" for k, v in readmitted_dist.items()],
        "",
        "## تعريف الهدف الثنائي للتدريب",
        "",
        target_binary_def,
        "",
        "## أعمدة مهمة (ملخص)",
        "",
        *[f"- **{k}:** {v}" for k, v in key_columns.items()],
        "",
        "## أعلى 15 عموداً من ناحية % القيم المفقودة (بعد اعتبار ? كـ NA في القراءة التالية)",
        "",
        *[f"- `{k}`: {v}%" for k, v in top_missing.items()],
        "",
        f"**تكرار encounter_id:** {doc['duplicate_encounter_ids']} صف مكرر",
        "",
    ]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print("Wrote:", out_json)
    print("Wrote:", out_md)


if __name__ == "__main__":
    main()

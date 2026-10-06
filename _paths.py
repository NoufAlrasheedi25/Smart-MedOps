"""مسارات المشروع — هيكل Dataset: raw (خام) / processed (بعد التنظيف)."""
import shutil
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent

# نسخة قديمة/احتياطية داخل data/ (يمكن الإبقاء عليها كمرآة)
DATA_DIR = PROJECT_ROOT / "data"

# الهيكل الموصى به للتقرير والمشرف
DATASET_DIR = PROJECT_ROOT / "Dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"

RAW_CSV = RAW_DIR / "diabetic_data.csv"
CLEAN_CSV = PROCESSED_DIR / "diabetic_data_cleaned.csv"

# ملفات إضافية خام (إن وُجدت في data/)
RAW_ER_CSV = RAW_DIR / "Hospital_ER_Data.csv"
LEGACY_ER = DATA_DIR / "Hospital_ER_Data.csv"

# طوارئ — بعد التنظيف والتجميع الزمني
ER_CLEAN_CSV = PROCESSED_DIR / "hospital_er_cleaned.csv"
ER_HOURLY_CSV = PROCESSED_DIR / "hospital_er_hourly_series.csv"
ER_CLEANING_LOG = PROCESSED_DIR / "er_cleaning_log.json"

OUTPUT_DIR = ML_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

for d in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, FIG_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def ensure_raw_from_data_folder() -> None:
    """ينسخ CSV الخام من data/ إلى Dataset/raw/ مرة واحدة إن لم يكن موجوداً."""
    LEGACY_MAIN = DATA_DIR / "diabetic_data.csv"
    if not RAW_CSV.exists() and LEGACY_MAIN.exists():
        shutil.copy2(LEGACY_MAIN, RAW_CSV)
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"لم يُعثر على {RAW_CSV}. ضع diabetic_data.csv في Dataset/raw أو في {LEGACY_MAIN}"
        )
    if LEGACY_ER.exists() and not RAW_ER_CSV.exists():
        shutil.copy2(LEGACY_ER, RAW_ER_CSV)


def ensure_er_raw() -> None:
    """يضمن وجود Hospital_ER_Data.csv في Dataset/raw (نسخة من data/ عند الحاجة)."""
    if not RAW_ER_CSV.exists() and LEGACY_ER.exists():
        shutil.copy2(LEGACY_ER, RAW_ER_CSV)
    if not RAW_ER_CSV.exists():
        raise FileNotFoundError(
            f"لم يُعثر على {RAW_ER_CSV}. ضع Hospital_ER_Data.csv في Dataset/raw أو في {LEGACY_ER}"
        )

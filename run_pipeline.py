# -*- coding: utf-8 -*-
"""تشغيل بالترتيب: وصف البيانات -> تنظيف -> EDA -> تدريب النماذج -> تنظيف طوارئ -> تدريب ازدحام/انتظار ER."""
import subprocess
import sys
from pathlib import Path

ML = Path(__file__).resolve().parent
scripts = [
    "01_dataset_description.py",
    "02_data_cleaning.py",
    "03_eda.py",
    "04_train_models.py",
    "02b_er_cleaning.py",
    "04b_train_er_congestion.py",
]


def main():
    for s in scripts:
        p = ML / s
        print("\n===", s, "===")
        r = subprocess.run([sys.executable, str(p)], cwd=str(ML))
        if r.returncode != 0:
            sys.exit(r.returncode)
    print("\nPipeline OK.")


if __name__ == "__main__":
    main()

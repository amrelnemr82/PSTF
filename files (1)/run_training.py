"""
Run this instead of `python aas_setting_time_pipeline.py`.

Why: if you run aas_setting_time_pipeline.py directly, Python loads it as
the '__main__' module, so joblib pickles the HybridModel class under
'__main__'. When ui_app.py later imports aas_setting_time_pipeline as a
normal module and tries to unpickle the saved model, Python looks for
HybridModel inside ui_app.py's own '__main__' and fails with:
    AttributeError: module '__main__' has no attribute 'HybridModel'

Running training through this tiny wrapper instead makes Python pickle
HybridModel under its real module name ('aas_setting_time_pipeline'),
which both this script and ui_app.py can resolve correctly.

Usage:
    python run_training.py --data Setting_timeX.xlsx
    python run_training.py --data Setting_timeX.xlsx --exclude-outliers
        (drops trials flagged by BOTH z-score and IQR outlier checks --
        currently Trial 16, whose FST=335min is 2x+ any other trial and
        has no close neighbor in the training data. This substantially
        improves FST accuracy but removes model coverage of that low-Na2O
        mix-design region. See the EDA outlier report / project report
        for the full tradeoff before using this flag.)
"""

import argparse
from aas_setting_time_pipeline import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Setting_timeX.xlsx")
    parser.add_argument("--exclude-outliers", action="store_true",
                         help="Drop trials flagged by both z-score and IQR outlier checks before training.")
    args = parser.parse_args()
    main(args.data, exclude_outliers=args.exclude_outliers)

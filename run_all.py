"""Run Modules A-G and all figures for the given cases (default: the mid-evaluation cases).

    .venv/bin/python run_all.py [case_id ...]
"""
import sys

from src import dose_response, events, plots, preprocess, recovery, robustness, sensitivity, synthesis
from src.common import load_config

if __name__ == "__main__":
    cases = sys.argv[1:] or load_config()["mid_eval_cases"]
    for cid in cases:
        preprocess.run(cid)                 # A
        df = events.run(cid)                # B
        sensitivity.run(cid, df)            # C
        sensitivity.run_terciles(cid, df)   # C (secondary contrast)
        rec = recovery.run(cid, df)         # D
        dose_response.run(cid, df, rec)     # E
        robustness.run(cid, df)             # F
    synthesis.run(cases)                    # G
    plots.run(cases)

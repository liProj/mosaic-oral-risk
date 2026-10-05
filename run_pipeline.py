"""Cross-platform entry point for the MOSAIC research workflow."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "code"))
from config import ROOT


def run(script, *args):
    subprocess.run([sys.executable, str(REPO / "code" / script), *args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "main", "ablation", "era", "extras", "stats", "figures", "reference-stats"])
    parser.add_argument("--features", choices=["base", "paper", "traj"], default="base")
    parser.add_argument("--extra", choices=["all", "weights", "hazard", "conformal", "importance", "curve", "featureset"], default="all")
    args = parser.parse_args()
    (ROOT / "results" / "experiments").mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        run("data_prep.py")
        run("features.py")
    elif args.stage in ("main", "ablation"):
        run("experiment.py", args.stage, args.features)
    elif args.stage == "era":
        run("era_analysis.py")
    elif args.stage == "extras":
        run("extras.py", args.extra)
    elif args.stage in ("stats", "reference-stats"):
        if args.stage == "reference-stats":
            # Isolated output avoids overwriting results from a user's own training run.
            target = ROOT / "reference_analysis" / "results" / "experiments"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / "reference_results" / "experiments" / "main_raw.csv", target / "main_raw.csv")
            os.environ["MOSAIC_ROOT"] = str(ROOT / "reference_analysis")
        run("stats_tests.py", "cox_multivar")
        run("stats_tests.py", "lightgbm")
    elif args.stage == "figures":
        required = [ROOT / "results" / "experiments" / "oof_predictions.csv", ROOT / "results" / "data_check.csv"]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            parser.error("Missing inputs: " + ", ".join(missing) + ". See README figure instructions.")
        run("plot_paper_figures.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Estimate a coeval population's age from its periods and dereddened colors."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from astrogyro.data import ROOT
from run_real_star_study import age_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV with Prot (days), BPRP_0, and optionally GaiaDR3_ID")
    parser.add_argument("--holdout-cluster", help="Use the three saved fits that exclude this entire reference cluster")
    args = parser.parse_args()
    root = ROOT/"project/results/real_stars_v2"
    sample = pd.read_csv(args.csv, dtype={"GaiaDR3_ID": str})
    if not {"Prot", "BPRP_0"} <= set(sample) or sample.empty:
        parser.error("Input needs nonempty Prot and BPRP_0 columns")
    if not (np.isfinite(sample.Prot) & (sample.Prot > 0) & np.isfinite(sample.BPRP_0)
            & sample.BPRP_0.between(.65, 1.5, inclusive="neither")).all():
        parser.error("Every row must have a positive finite period and 0.65 < BPRP_0 < 1.5; no rows are silently removed")
    primary = pd.read_csv(ROOT/"project/results/research_v1/selected_stars.csv", dtype={"GaiaDR3_ID": str})
    if args.holdout_cluster:
        paths = sorted((root/"folds").glob(f"holdout_{args.holdout_cluster}_seed*/result.json"))
        if len(paths) != 3:
            parser.error("No complete three-fit holdout exists for that cluster")
        fits = [json.loads(p.read_text()) for p in paths]
        models = [fit["models"]["mixture_quadratic"] for fit in fits]
        known = primary[primary.cluster != args.holdout_cluster]
        age_grid = np.geomspace(80, 5000, 401)
    else:
        models = [json.loads((root/"application_model.json").read_text())["model"]]
        known = primary
        age_grid = np.geomspace(135, 3870, 401)
    if "GaiaDR3_ID" in sample and set(sample.GaiaDR3_ID) & set(known.GaiaDR3_ID):
        parser.error("Input overlaps fitted/validation catalogue stars; use the appropriate cluster holdout for evaluation")
    sample["log_color"], sample["log_period"] = np.log10(sample.BPRP_0), np.log10(sample.Prot)
    indices = [int(np.argmax(age_profile(model, sample, age_grid))) for model in models]
    estimates = age_grid[indices]
    print(json.dumps({"n_stars": len(sample), "inferred_cluster_age_myr": float(10**np.mean(np.log10(estimates))),
                      "per_fit_age_myr": estimates.tolist(), "boundary_hits": int(sum(i in [0, 400] for i in indices)),
                      "id_overlap_checked": "GaiaDR3_ID" in sample,
                      "interpretation": "Maximum-likelihood age for an assumed coeval population. Per-fit variation is not a calibrated confidence interval. Systematic age and color uncertainties remain."}, indent=2))


if __name__ == "__main__": main()

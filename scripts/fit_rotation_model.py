#!/usr/bin/env python3
"""Fit an application model after evaluation; never use it to report holdout scores."""
import json
from collections import Counter

import pandas as pd

from astrogyro.data import ROOT, equal_cluster_weights, features, sha256
from astrogyro.distributions import fit_distribution
from astrogyro.experiments import write_json


def main():
    root = ROOT/"project/results/real_stars_v2"
    paths = sorted((root/"folds").glob("*/result.json"))
    penalties = Counter(json.loads(p.read_text())["models"]["mixture_quadratic"]["penalty"] for p in paths)
    penalty = max(penalties, key=lambda value: (penalties[value], value))
    sample_path = ROOT/"project/results/research_v1/selected_stars.csv"
    sample = pd.read_csv(sample_path)
    model = fit_distribution(features(sample), sample.log_period.to_numpy(),
                             equal_cluster_weights(sample.cluster), "mixture_quadratic", penalty)
    write_json(root/"application_model.json", {"model": model, "sample_sha256": sha256(sample_path),
               "penalty_rule": "Most frequent inner-validation choice across 48 folds; ties use stronger regularization",
               "penalty_counts": dict(penalties), "n_stars": len(sample), "n_clusters": sample.cluster.nunique(),
               "age_domain_myr": [135, 3870], "color_domain": [.65, 1.5],
               "scope": "Conditional rotation distribution. Trained on the full primary sample; not a held-out model or calibrated individual-star age estimator."})
    print("Saved application_model.json;", dict(penalties))


if __name__ == "__main__": main()

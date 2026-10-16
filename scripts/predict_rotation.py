#!/usr/bin/env python3
"""Predict a rotation distribution at a supplied age and dereddened Gaia color."""
import argparse
import json
from pathlib import Path

import numpy as np

from astrogyro.data import ROOT
from astrogyro.distributions import components, quantiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--age-myr", type=float, required=True)
    parser.add_argument("--bp-rp", type=float, required=True, help="Dereddened Gaia DR3 BP-RP")
    parser.add_argument("--model", type=Path, default=ROOT/"project/results/real_stars_v2/application_model.json")
    args = parser.parse_args()
    saved = json.loads(args.model.read_text())
    if not saved["age_domain_myr"][0] <= args.age_myr <= saved["age_domain_myr"][1]:
        parser.error("Age must be within the application's observed 135-3870 Myr domain")
    if not saved["color_domain"][0] < args.bp_rp < saved["color_domain"][1]:
        parser.error("Dereddened BP-RP must be strictly between 0.65 and 1.5")
    X = np.array([[np.log10(args.age_myr/1000), np.log10(args.bp_rp)]])
    q = 10**quantiles(saved["model"], X)[0]
    mu, sigma, weights = components(saved["model"], X)
    print(json.dumps({"age_myr": args.age_myr, "dereddened_bp_rp": args.bp_rp,
                      "median_period_days": float(q[2]), "prediction_interval_68_days": [float(q[1]), float(q[3])],
                      "prediction_interval_90_days": [float(q[0]), float(q[4])],
                      "slow_component_weight": float(weights[0, 0]),
                      "interpretation": "Population spread in rotation period at the supplied age/color; includes observed scatter, not uncertainty on the mean or a stellar age estimate."}, indent=2))


if __name__ == "__main__": main()

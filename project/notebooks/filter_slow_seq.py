"""
Refactored slow-sequence filtering module.

Extracted from phase2_data_exploration.py for reuse in sensitivity analyses.
The thresholds (young_frac, old_frac) are now parameters.
"""

import numpy as np


def filter_slow_sequence(sub_df, age_myr, young_frac=0.40, old_frac=0.30,
                         color_col="BPRP_0", period_col="Prot"):
    """
    Filter a single cluster's data to keep slow-sequence rotators.

    Parameters:
        sub_df: DataFrame for one cluster
        age_myr: cluster age in Myr
        young_frac: fraction of 75th-percentile ridge to use as cutoff for young clusters
        old_frac: fraction of median to use as cutoff for old clusters
        color_col: column name for color
        period_col: column name for period

    Returns: boolean mask (True = keep)
    """
    colors = sub_df[color_col].values
    periods = sub_df[period_col].values
    keep = np.ones(len(sub_df), dtype=bool)

    color_min, color_max = np.nanpercentile(colors, [2, 98])
    bin_width = 0.3
    bin_step = 0.1

    if age_myr >= 600:
        bin_centers = np.arange(color_min + bin_width / 2, color_max, bin_step)
        for bc in bin_centers:
            in_bin = (colors >= bc - bin_width / 2) & (colors < bc + bin_width / 2)
            if in_bin.sum() < 5:
                continue
            median_p = np.median(periods[in_bin])
            cut = median_p * old_frac
            keep[in_bin & (periods < cut)] = False
    else:
        bin_centers = np.arange(color_min + bin_width / 2, color_max, bin_step)
        for bc in bin_centers:
            in_bin = (colors >= bc - bin_width / 2) & (colors < bc + bin_width / 2)
            if in_bin.sum() < 5:
                continue
            p75 = np.percentile(periods[in_bin], 75)
            cut = p75 * young_frac
            keep[in_bin & (periods < cut)] = False

    return keep


def apply_slow_seq_filter(df, young_frac=0.40, old_frac=0.30):
    """Apply slow-sequence filter to all clusters in a DataFrame.

    Returns: filtered DataFrame.
    """
    mask = np.zeros(len(df), dtype=bool)
    for cluster_name, sub_df in df.groupby("cluster"):
        age = sub_df["fiducial_age_Myr"].iloc[0]
        keep = filter_slow_sequence(sub_df, age, young_frac=young_frac, old_frac=old_frac)
        mask[sub_df.index[keep]] = True
    return df[mask].reset_index(drop=True)

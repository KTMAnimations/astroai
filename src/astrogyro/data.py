"""Load the original catalogue without using rotation to define membership."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "project/data/prot_catalogue.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_catalog(path: Path = CATALOG) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"GaiaDR3_ID": str})
    df["membership"] = df["mem_prob_hdbscan"].fillna(df["mem_prob_source"])
    return df


def select_sample(df: pd.DataFrame, membership: float = 0.8,
                  min_age: float = 100, color_lo: float = 0.65,
                  color_hi: float = 1.5, min_cluster_size: int = 10):
    """Return selected rows and a sequential exclusion ledger.

    No upper period limit or slow-sequence cut is applied. This population
    includes rotational scatter; it is not asserted to be a pure slow sequence.
    """
    ledger = [{"step": "catalogue", "stars": len(df), "clusters": df.cluster.nunique()}]
    result = df.copy()
    cuts = [
        ("photometry_and_astrometry", lambda d: (d.cmd_exclude == 0) &
         (d.dered_exclude == 0) & (d.phot_exclude == 0) & (d.dr3_ruwe < 1.4)),
        ("finite_positive_measurements", lambda d: np.isfinite(d.Prot) &
         np.isfinite(d.BPRP_0) & np.isfinite(d.fiducial_age_Myr) & (d.Prot > 0)),
        ("age_and_color_domain", lambda d: (d.fiducial_age_Myr >= min_age) &
         (d.BPRP_0 > color_lo) & (d.BPRP_0 < color_hi)),
        ("membership_threshold", lambda d: d.membership >= membership),
        ("cluster_sample_size", lambda d: d.groupby("cluster").Prot.transform("size") >= min_cluster_size),
    ]
    for name, rule in cuts:
        result = result.loc[rule(result)].copy()
        ledger.append({"step": name, "stars": len(result), "clusters": result.cluster.nunique()})
    if result.GaiaDR3_ID.duplicated().any():
        raise ValueError("Duplicate Gaia IDs in selected sample")
    result = result.sort_values(["fiducial_age_Myr", "cluster", "GaiaDR3_ID"]).reset_index(drop=True)
    result["log_age_gyr"] = np.log10(result.fiducial_age_Myr / 1000)
    result["log_color"] = np.log10(result.BPRP_0)
    result["log_period"] = np.log10(result.Prot)
    return result, pd.DataFrame(ledger)


def features(df: pd.DataFrame) -> np.ndarray:
    return df[["log_age_gyr", "log_color"]].to_numpy(dtype=float)


def equal_cluster_weights(groups) -> np.ndarray:
    groups = np.asarray(groups)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    return weights / weights.mean()


def cluster_split(df: pd.DataFrame, held_out: str, seed: int):
    """Outer test is one entire cluster; inner validation is age-stratified.

    The remaining clusters are divided into three age strata, from each of
    which one validation cluster is drawn. No periods enter the split.
    """
    test = df.cluster.eq(held_out).to_numpy()
    groups = (df.loc[~test].groupby("cluster").fiducial_age_Myr.first()
              .sort_values().index.to_numpy())
    if len(groups) < 6:
        raise ValueError("Need at least seven clusters for this split")
    rng = np.random.default_rng(seed)
    val_groups = [str(rng.choice(chunk)) for chunk in np.array_split(groups, 3)]
    validation = df.cluster.isin(val_groups).to_numpy()
    discovery = ~(test | validation)
    return discovery, validation, test

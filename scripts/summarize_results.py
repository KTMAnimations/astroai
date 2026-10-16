#!/usr/bin/env python3
"""Build complete scenario tables and trace every headline number to saved trials."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from astrogyro.data import ROOT
from astrogyro.experiments import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=ROOT/"project/results/research_v1")
    args=parser.parse_args();root=args.results
    config=json.loads((root/"manifest.json").read_text())["config"]
    threshold=config["synthetic"]["recovery_threshold_dex"]
    scores=pd.read_csv(root/"synthetic_scores.csv")
    # Strict JSON represents nonfinite scores as null. Restore the failure
    # penalty on read; pandas would otherwise silently omit those trials.
    scores["truth_grid_rmse"]=scores.truth_grid_rmse.fillna(np.inf)
    scores["invalid_grid_prediction"]=~np.isfinite(scores.truth_grid_rmse)
    scores.to_csv(root/"synthetic_scores.csv",index=False)
    summary=scores.groupby(["truth","design","noise_dex","model"]).agg(
        n=("trial","size"),n_recovered=("recovered","sum"),n_invalid=("invalid_grid_prediction","sum"),mean_grid_rmse=("truth_grid_rmse","mean"),
        median_grid_rmse=("truth_grid_rmse","median"),min_grid_rmse=("truth_grid_rmse","min"),max_grid_rmse=("truth_grid_rmse","max")).reset_index()
    summary.to_csv(root/"synthetic_summary.csv",index=False)
    available=[]
    for path in sorted((root/"synthetic").glob("*/candidates.csv")):
        table=pd.read_csv(path)
        meta=scores.loc[scores.trial==path.parent.name].iloc[0]
        available.append({"trial":path.parent.name,"truth":meta.truth,"design":meta.design,"noise_dex":meta.noise_dex,
                          "any_candidate_recovers":bool((table.truth_grid_rmse<threshold).any()),
                          "best_available_truth_rmse":float(table.truth_grid_rmse.min()),
                          "n_candidates":len(table)})
    candidate_table=pd.DataFrame(available)
    candidate_table.to_csv(root/"candidate_recovery.csv",index=False)
    counts=summary.pivot(index=["truth","design","noise_dex"],columns="model",values="n_recovered")
    counts["any_candidate"]=candidate_table.groupby(["truth","design","noise_dex"]).any_candidate_recovers.sum()
    counts["n_trials"]=config["synthetic"]["replicates"]
    counts.to_csv(root/"recovery_counts.csv")
    candidate_joins=scores[scores.model=="pysr_default"].merge(candidate_table,on=["trial","truth","design","noise_dex"])
    selection_misses=int(((~candidate_joins.recovered)&candidate_joins.any_candidate_recovers).sum())
    search_misses=int((~candidate_joins.any_candidate_recovers).sum())
    paired=scores.pivot(index=["truth","design","noise_dex","replicate"],columns="model",values="truth_grid_rmse")
    both_invalid=~np.isfinite(paired.pysr_validation)&~np.isfinite(paired.pysr_default)
    delta=paired.pysr_validation-paired.pysr_default
    comparison={"n_trials":len(paired),"validation_lower_error":int((delta < -1e-10).sum()),
                "default_lower_error":int((delta > 1e-10).sum()),"numerically_tied":int((delta.abs()<=1e-10).sum()),
                "both_invalid":int(both_invalid.sum()),
                "invalid_selected_functions":scores.loc[scores.invalid_grid_prediction,["trial","model","expression"]].to_dict("records"),
                "default_selection_misses_despite_accurate_candidate":selection_misses,
                "no_candidate_recovers":search_misses}
    write_json(root/"selection_diagnostics.json",comparison)
    run_files=list((root/"synthetic").glob("*/result.json"))+list((root/"observational").glob("*/result.json"))
    elapsed=sum(json.loads(p.read_text())["elapsed_seconds"] for p in run_files)
    write_json(root/"run_totals.json",{"n_searches":len(run_files),"sum_search_elapsed_seconds":elapsed,
                                     "note":"Sum across independent processes, not wall-clock elapsed time."})
    print(counts.to_string())
    print(json.dumps(comparison,indent=2))


if __name__=="__main__":main()

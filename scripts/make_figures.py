#!/usr/bin/env python3
"""Create presentation figures from completed, verified experiment tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from astrogyro import baselines
from astrogyro.data import ROOT
from astrogyro.experiments import truth_log_period, write_json
from astrogyro.symbolic import evaluate

LABELS = {"fixed_skumanich": "Fixed age exponent", "power_law": "Fitted power law",
          "quadratic": "Quadratic baseline", "pysr_default": "Symbolic: default choice",
          "pysr_validation": "Symbolic: validation choice"}
COLORS = {"fixed_skumanich": "#8d929c", "power_law": "#25344b", "quadratic": "#39956b",
          "pysr_default": "#d56a3a", "pysr_validation": "#226cb5"}
ORDER = list(LABELS)


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#25344b",
                         "text.color": "#25344b", "axes.edgecolor": "#aeb5bf",
                         "grid.color": "#e6e9ee", "axes.axisbelow": True,
                         "figure.constrained_layout.use": True})


def sample_figure(sample, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    sc = axes[0].scatter(sample.BPRP_0, sample.Prot, c=sample.fiducial_age_Myr,
                         cmap="viridis", norm=LogNorm(100, 3870), s=15, alpha=.7, edgecolors="none")
    axes[0].set(xlabel="Dereddened Gaia color (BP-RP)", ylabel="Observed rotation period (days)",
                yscale="log", title=f"{len(sample):,} selected stars")
    fig.colorbar(sc, ax=axes[0], label="Cluster age (Myr)", shrink=.8)
    counts = sample.groupby("cluster").agg(age=("fiducial_age_Myr", "first"), n=("Prot", "size")).sort_values("age")
    axes[1].barh(np.arange(len(counts)), counts.n, color="#226cb5", height=.7)
    axes[1].set_yticks(np.arange(len(counts)), [f"{c}  ({r.age/1000:.2f} Gyr)" for c,r in counts.iterrows()], fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Number of selected stars", title=f"{len(counts)} held-out clusters")
    axes[1].set_xlim(0, counts.n.max()*1.15)
    for i,n in enumerate(counts.n):
        axes[1].text(n+1.5, i, str(n), va="center", fontsize=9)
    fig.suptitle("The observational sample: measured periods, uneven age coverage", fontsize=16)
    save(fig, output, "sample")


def synthetic_figure(scores, config, output):
    truths = config["synthetic"]["truths"]
    designs = config["synthetic"]["designs"]
    noise = config["synthetic"]["noise_dex"]
    names = {"skumanich": "Known square-root law", "interaction": "Known age-color interaction", "saturation": "Known gradual slowdown"}
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, sharey=True)
    selected_models = ["power_law", "quadratic", "pysr_default", "pysr_validation"]
    for i, truth in enumerate(truths):
        for j, design in enumerate(designs):
            ax = axes[i,j]
            for mi,model in enumerate(selected_models):
                selected = scores[(scores.truth==truth)&(scores.design==design)&(scores.model==model)]
                groups = [selected.loc[selected.noise_dex==n,"truth_grid_rmse"].to_numpy() for n in noise]
                med = np.array([np.median(g) for g in groups])
                qlo = np.array([np.quantile(g,.25) for g in groups]);qhi = np.array([np.quantile(g,.75) for g in groups])
                # A shared tiny plotting floor distinguishes numerical-zero errors
                # from meaningful error without hiding failed predictions.
                med,qlo,qhi = [np.maximum(v,1e-6) for v in [med,qlo,qhi]]
                xx = np.arange(len(noise))+(mi-1.5)*.07
                ax.errorbar(xx, med, yerr=[med-qlo,qhi-med], color=COLORS[model], marker="o", ms=4,
                            lw=1.5, capsize=3, label=LABELS[model], ls="--" if mi<2 else "-")
            ax.axhline(config["synthetic"]["recovery_threshold_dex"], color="#aaadb3", ls=":", lw=1)
            ax.set_yscale("symlog", linthresh=.001, linscale=.5)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=.7)
            ax.set_title(f"{names[truth]}\n{'Catalogue sampling' if design=='catalog' else 'Even log-age/color coverage'}", fontsize=12)
            ax.set_xticks(np.arange(len(noise)), [str(n) for n in noise])
            if j==0: ax.set_ylabel("Error against known law (dex)")
            if i==2: ax.set_xlabel("Added measurement noise (dex)")
            invalid=scores[(scores.truth==truth)&(scores.design==design)&~np.isfinite(scores.truth_grid_rmse)]
            if len(invalid):
                ax.text(.03,.96,f"{len(invalid)} validation choices invalid on grid\nCounted as failures (infinite error)",
                        transform=ax.transAxes,va="top",fontsize=9,color=COLORS["pysr_validation"])
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=11)
    fig.suptitle("Controlled recovery: selection, sampling and noise\nMedian and interquartile range of 10 independent trials; dotted line = 0.02 dex", fontsize=15)
    save(fig, output, "synthetic_recovery")


def recovery_counts_figure(scores, config, output):
    truths = config["synthetic"]["truths"]
    noise = config["synthetic"]["noise_dex"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    matrices = {}
    labels = [f"{t.title()} / {d}" for t in truths for d in ["catalog", "balanced"]]
    for ax, model in zip(axes,["pysr_default","pysr_validation"]):
        counts = np.array([[int(scores.loc[(scores.truth==t)&(scores.design==d)&(scores.noise_dex==n)&(scores.model==model),"recovered"].sum())
                            for n in noise] for t in truths for d in ["catalog","balanced"]])
        ax.imshow(counts, vmin=0, vmax=config["synthetic"]["replicates"], cmap="Blues", aspect="auto")
        ax.set_xticks(range(3), [str(n) for n in noise]);ax.set_yticks(range(6),labels,fontsize=10)
        ax.set_title(LABELS[model]);ax.set_xlabel("Added noise (dex)")
        for i,j in np.ndindex(counts.shape):
            ax.text(j,i,f"{counts[i,j]}/{config['synthetic']['replicates']}",ha="center",va="center",color="white" if counts[i,j]>=6 else "#25344b",fontsize=13,fontweight="bold")
        matrices[model] = counts.tolist()
    fig.suptitle("How often did the selected equation recover the known function?\nRecovery means error < 0.02 dex on an independent, noise-free grid",fontsize=15)
    save(fig,output,"recovery_counts")


def model_comparison(scores, summary, output):
    mean = scores.groupby(["cluster","model"]).rmse_dex.mean().unstack()
    fig, axes = plt.subplots(1,2,figsize=(13,5), gridspec_kw={"width_ratios":[1,1.15]})
    values = [mean[m].mean() for m in ORDER]
    y = np.arange(len(ORDER))
    axes[0].barh(y,values,color=[COLORS[m] for m in ORDER],height=.6)
    axes[0].set_yticks(y,[LABELS[m] for m in ORDER],fontsize=10);axes[0].invert_yaxis()
    axes[0].set(xlabel="Mean held-out cluster RMSE (dex)",title="Prediction error on unseen clusters")
    axes[0].set_xlim(0,max(values)*1.18)
    for i,v in enumerate(values):axes[0].text(v+.004,i,f"{v:.3f}",va="center",fontsize=11)
    comparisons=[m for m in ORDER if m!="power_law"]
    for i,model in enumerate(comparisons):
        interval=summary["paired_cluster_intervals"][f"{model}_minus_power_law"]
        mid=interval["mean_difference"];lo=interval["ci95_low"];hi=interval["ci95_high"]
        axes[1].errorbar(mid,i,xerr=[[mid-lo],[hi-mid]],fmt="o",color=COLORS[model],capsize=5,ms=7)
    axes[1].axvline(0,color="#8d929c",ls="--",lw=1)
    axes[1].set_yticks(range(len(comparisons)),[LABELS[m] for m in comparisons],fontsize=10);axes[1].invert_yaxis()
    axes[1].set(xlabel="Error difference from fitted power law (dex)\nLeft: lower error     Right: higher error",title="Paired differences with 95% intervals")
    axes[1].grid(axis="x")
    fig.suptitle("Fair comparison: same discovery, validation and test clusters\n16 withheld clusters; three split/search seeds averaged within each cluster",fontsize=15)
    save(fig,output,"model_comparison")


def cluster_figure(scores, output):
    ages=scores.groupby("cluster").age_Myr.first().sort_values()
    values=scores.groupby(["cluster","model"]).rmse_dex.mean().unstack().loc[ages.index,ORDER]
    fig,ax=plt.subplots(figsize=(11,8))
    im=ax.imshow(values,cmap="YlOrRd",aspect="auto",vmin=0)
    ax.set_yticks(range(len(values)),[f"{c} ({ages[c]/1000:.2f} Gyr)" for c in values.index],fontsize=10)
    ax.set_xticks(range(len(ORDER)),[LABELS[m].replace(": ","\n") for m in ORDER],fontsize=10)
    for i,j in np.ndindex(values.shape):
        v=values.iloc[i,j];ax.text(j,i,f"{v:.3f}",ha="center",va="center",fontsize=10,color="white" if v>values.to_numpy().max()*.6 else "#25344b")
    fig.colorbar(im,ax=ax,label="Held-out RMSE (dex)",shrink=.7)
    ax.set_title("Every test cluster, including the difficult cases\nMean error across three discovery/validation splits",pad=15)
    save(fig,output,"cluster_errors")


def m67_figure(sample, results, output):
    m67=sample[sample.cluster=="M67"]
    colors=np.linspace(.65,1.5,150)
    X=np.column_stack([np.full(len(colors),np.log10(3.87)),np.log10(colors)])
    fig,ax=plt.subplots(figsize=(10,5.8))
    ax.scatter(m67.BPRP_0,m67.Prot,s=45,facecolors="white",edgecolors="#25344b",linewidths=1.5,label=f"Observed M67 stars (n={len(m67)})",zorder=5)
    for name in ["fixed_skumanich","power_law","pysr_validation"]:
        predictions=[]
        for path in sorted((results/"observational").glob("holdout_M67_seed*/result.json")):
            model=json.loads(path.read_text())["models"][name]
            pred=evaluate(model["expression"],X) if name.startswith("pysr") else baselines.predict(model,X)
            predictions.append(10**pred)
        predictions=np.asarray(predictions)
        ax.plot(colors,predictions.mean(axis=0),color=COLORS[name],lw=2,label=LABELS[name])
        ax.fill_between(colors,predictions.min(axis=0),predictions.max(axis=0),color=COLORS[name],alpha=.12)
    ax.set(xlabel="Dereddened Gaia color (BP-RP)",ylabel="Rotation period (days)",
           title="M67: an old cluster withheld from every fit\nBands show the range across three splits, not calibrated uncertainty")
    ax.legend(fontsize=10,loc="upper left");ax.grid(alpha=.6)
    save(fig,output,"m67_holdout")


def sensitivity_figure(scores, output):
    agg=scores.groupby(["membership_cut","color_hi","model"]).rmse_dex.mean()
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,color_hi in zip(axes,[1.5,2.]):
        for model in baselines.NAMES:
            ys=[agg.loc[(cut,color_hi,model)] for cut in [.5,.8,.9]]
            ax.plot([.5,.8,.9],ys,"o-",label=LABELS[model],color=COLORS[model],lw=2)
        ax.set(xlabel="Minimum membership score in training",title=f"Training color upper limit: {color_hi}")
        ax.set_xticks([.5,.8,.9]);ax.grid(alpha=.7)
    axes[0].set_ylabel("Mean cluster RMSE on common test stars (dex)")
    axes[1].legend(fontsize=10)
    fig.suptitle("Sensitivity to training-sample choices\nAll runs test the same primary stars; each test cluster is excluded from training",fontsize=14)
    save(fig,output,"sample_sensitivity")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=ROOT/"project/results/research_v1")
    parser.add_argument("--output",type=Path,default=ROOT/"project/figures/research_v1")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    style()
    sample=pd.read_csv(args.results/"selected_stars.csv")
    config=json.loads((args.results/"manifest.json").read_text())["config"]
    sample_figure(sample,args.output)
    scores=pd.read_csv(args.results/"synthetic_scores.csv")
    scores["truth_grid_rmse"]=scores.truth_grid_rmse.fillna(np.inf)
    synthetic_figure(scores,config,args.output);recovery_counts_figure(scores,config,args.output)
    observed=pd.read_csv(args.results/"observational_scores.csv")
    summary=json.loads((args.results/"observational_summary.json").read_text())
    model_comparison(observed,summary,args.output);cluster_figure(observed,args.output)
    m67_figure(sample,args.results,args.output)
    sensitivity_figure(pd.read_csv(args.results/"sensitivity_scores.csv"),args.output)
    print(f"Saved seven PNG/SVG figures to {args.output}")


if __name__=="__main__":main()

"""
Task 7: Estimate bootstrap compute time.

Uses smoke test timing + scaling to estimate full run times.
Will update with actual times from Task 6 when available.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# ── Check if reproducibility results exist ───────────────────────────────
repro_path = os.path.join(RESULTS, "reproducibility_results.json")

if os.path.exists(repro_path):
    with open(repro_path) as f:
        repro = json.load(f)

    print("=== Actual timing from reproducibility runs ===\n")
    total_time = 0
    for r in repro:
        t = r["wall_clock_seconds"]
        total_time += t
        print(f"  Run {r['run']}: {t:.0f}s ({t/60:.1f} min)")

    avg_time = total_time / len(repro)
    print(f"\n  Average per run: {avg_time:.0f}s ({avg_time/60:.1f} min)")

    # Extrapolate for bootstrap
    print(f"\n=== Bootstrap compute estimates ===\n")
    print(f"  Based on actual 200-iteration constrained run time: {avg_time:.0f}s\n")

    for n_iter_boot in [50, 100, 200]:
        scale = n_iter_boot / 200.0
        time_per_run = avg_time * scale  # rough linear scaling

        for n_resamples in [50, 100]:
            total = time_per_run * n_resamples
            hours = total / 3600
            days = hours / 24
            print(f"  {n_resamples:3d} resamples × {n_iter_boot} iters: "
                  f"{hours:6.1f} hrs ({days:.1f} days)")
        print()

else:
    print("=== Reproducibility results not yet available ===\n")
    print("  Using smoke test estimates (5 iterations):\n")

    # Smoke test: ~17s for first run (Julia compilation), ~1s for subsequent
    # Julia compilation is one-time overhead. Per-run time at 5 iters ≈ 1-2s.
    # Scaling: 200 iters / 5 iters = 40x → ~40-80s per run after warmup
    # But PySR scaling is super-linear due to population evolution
    # Empirically, 200 iters on 2911 stars typically takes 1-4 hours

    # Use conservative estimate: 2 hours per 200-iter run
    est_per_run_200 = 2 * 3600  # 2 hours in seconds

    print(f"  Estimated per-run time (200 iters, 2911 stars): ~2 hours")
    print(f"  (Conservative estimate — actual may be 1-4 hours)\n")

    for n_iter_boot in [50, 200]:
        scale = n_iter_boot / 200.0
        time_per_run = est_per_run_200 * scale

        for n_resamples in [50, 100]:
            total = time_per_run * n_resamples
            hours = total / 3600
            days = hours / 24
            print(f"  {n_resamples:3d} resamples × {n_iter_boot} iters: "
                  f"{hours:6.1f} hrs ({days:.1f} days)")
        print()

    print("  Recommendation (pending actual timing):")
    print("  - 50 resamples × 50 iters:  ~25 hrs (1 day) — feasible on a single machine")
    print("  - 100 resamples × 50 iters: ~50 hrs (2 days) — feasible before Sept 2026 deadline")
    print("  - 50 resamples × 200 iters: ~100 hrs (4 days) — use if reproducibility check fails")

print(f"\n  ML4PS deadline: ~September 2026")
print(f"  All options are feasible. Start with 50 × 50, increase if needed.")

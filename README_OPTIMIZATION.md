# FROR optimization
Keep ml_results and the unchanged recipes_foodcom.csv. Copy these files into the existing project. The previous locally generated rankers.joblib must be trusted. No new dependencies.

Run in PowerShell:

    .venv\Scripts\python optimize_models.py --train-queries 2000 --validation-queries 200 --test-queries 500
    .venv\Scripts\python evaluate_final.py
    .venv\Scripts\python -m streamlit run app.py

Optimization writes optimized_results; it refuses to overwrite an existing directory. Final evaluation writes optimized_results/final, also without overwriting. For a failed run, rename the incomplete folder before retrying. Do not repeatedly tune against final results. App uses optimized_results/rankers.joblib if present, otherwise ml_results/rankers.joblib.

The original seed 42, 1000 training and 300 test queries are replayed to identify old test targets; this assumes the original training script and unchanged dataset and NumPy sampling behavior. The final 20% target partition is sampled from the old test partition excluding previously sampled test targets; training uses 60% of the catalog drawn from the old training partition. Validation is the remaining 20%. All recipes remain searchable, so this is held-out target retrieval, not an unseen catalog. Rating features come from historical training ratings only. Synthetic inventories are generated from targets; unselected recipes are not confirmed dislikes.

Three logistic regression C values and three random forest leaf sizes are compared by validation MRR. Then 70 nonnegative weight vectors (increments of .25, sum 1) are compared using MRR + .5 utilization - .5 missing/5 + .25 complete rate. FROR combines logistic score, Jaccard, coverage, utilization and negative capped missing count/5. Weights can be zero, including the learned component; report the selected weights honestly. These objectives are design choices, not measured consumer preferences. Final evaluation compares seven methods on identical inventories, with 2000 paired bootstrap resamples per contrast (unadjusted for multiple comparisons). Timing excludes shared feature computation. App splits exact and near matches; benchmark ranks the unrestricted catalog, so these metrics do not evaluate the filtered UI. Quantities and real-world waste are not measured. Feature importance is descriptive, not causal. No ablation experiment is included yet.

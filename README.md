# Fridge Rescue AI

**Developer handoff and research guide | Phase 5 — optimized recommendation system**

Fridge Rescue AI recommends recipes based on ingredients a person already has, identifies additional ingredients needed, suggests pantry staples and maintains a local grocery list. It is a Python/Streamlit research prototype comparing seven ranking methods on a structured Food.com recipe catalog. Its research question is how to balance recipe retrieval, ingredient utilization and missing ingredient requirements.

**Start here:** If you received an already configured project folder, follow [Quick start](#2-quick-start). If you are installing from scratch, follow [Installation](#3-installation-from-scratch). To reproduce the research or change recipe counts, use Sections 7–9. No knowledge of machine learning is required to launch the application.

## Contents

1. [Requirements and project layout](#1-requirements-and-project-layout)
2. [Quick start](#2-quick-start)
3. [Installation from scratch](#3-installation-from-scratch)
4. [Using the application](#4-using-the-application)
5. [Data and preprocessing](#5-data-and-preprocessing)
6. [How the recommendation models work](#6-how-the-recommendation-models-work)
7. [Training and reproducing experiments](#7-training-and-reproducing-experiments)
8. [Changing recipe amounts and experiment settings](#8-changing-recipe-amounts-and-experiment-settings)
9. [Testing and interpreting results](#9-testing-and-interpreting-results)
10. [File reference](#10-file-reference)
11. [Troubleshooting and maintenance](#11-troubleshooting-and-maintenance)
12. [Research findings and limitations](#12-research-findings-and-limitations)

---

## 1. Requirements and project layout

- Windows 10/11 and PowerShell (examples use Windows); macOS/Linux can use equivalent Python commands.
- Python 3.11 or 3.12 recommended; use the same Python and package versions as the original training environment for exact reproduction.
- Visual Studio Code is optional but recommended; any terminal and web browser will work.
- Internet access for package installation and dataset download; offline use is possible after setup.
- No GPU, cloud account, API key or paid AI service is needed. Random forest uses available CPU cores; memory and runtime grow with catalog and experiment size.
- Packages: Streamlit (web UI), pandas (tables), NumPy (numerical operations), scikit-learn (vectors and ML), joblib (model persistence, installed with scikit-learn). SQLite, JSON, CSV and argparse are part of Python's standard library.

The active project should have this structure after applying all updates through Phase 5. Generated files may be absent on a new installation.

```text
FridgeRescueAI/
  .venv/                         local Python environment
  Archived/                      retired RecipeNLG files
  data/FoodCom/                  original Food.com download
  app.py                         Streamlit interface
  engine.py                      recommendation engine
  ingredient_names.py            shared ingredient normalization
  inspect_foodcom.py             raw data schema inspection
  prepare_foodcom.py             data conversion and sampling
  audit_dataset.py               vocabulary quality audit
  evaluate.py                    legacy four-method evaluation
  learned_ranker.py              shared supervised features
  train_models.py                preliminary supervised experiment
  optimized_ranker.py            FROR scoring and ranking metrics
  optimize_models.py             validation and parameter search
  evaluate_final.py              held-out evaluation
  recipes.csv                    small demonstration catalog
  recipes_foodcom.csv            generated main catalog
  recipes_foodcom.report.json    generated conversion report
  recipes_foodcom.vocabulary.csv generated ingredient counts
  foodcom_schema.json            generated schema report
  grocery.db                     local grocery list, generated on use
  ml_results/                    preliminary model and results
  optimized_results/             optimized model and results
  requirements.txt               Python dependencies
  README.md                      this guide
  README_ML.md                   earlier training notes
  README_OPTIMIZATION.md         earlier optimization notes
```

Use the most recent `app.py` and `engine.py` (Phase 5), `ingredient_names.py` and `prepare_foodcom.py` (normalization update), and the remaining modules from their respective updates. Do not copy an older engine over the current one. This README is documentation only; it does not install or change code.

## 2. Quick start

1. Open the project folder in VS Code (File > Open Folder). Open Terminal > New Terminal.
2. Confirm that the terminal is in the directory containing `app.py` and `requirements.txt`:

```powershell
Get-Location
Get-ChildItem
```

3. If `.venv`, the dataset and trained models already exist, run:

```powershell
.venv\Scripts\python -m streamlit run app.py
```

4. Open the local URL printed in the terminal (normally `http://localhost:8501`). Leave the terminal running. Stop the server with Ctrl+C.

Select `recipes_foodcom.csv` for the research catalog, or `recipes.csv` for the small demonstration. The app prefers `optimized_results/rankers.joblib` over `ml_results/rankers.joblib`. With a matching optimized bundle, all seven methods are available. With only the preliminary bundle, six methods are available. Without either bundle, the four non-trained methods are available. The demo dataset intentionally does not load trained models.

## 3. Installation from scratch

### 3.1 Create the environment

Install Python from https://www.python.org/downloads/ and ensure the Windows Python launcher is available. From the project root:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip check
```

If you installed Python 3.12, use `py -3.12` instead. If `py` is unavailable, use the installed `python` command to create the environment. Optional activation: `.venv\Scripts\Activate.ps1`. All examples explicitly use the virtual environment's Python, so activation is not necessary and no PowerShell execution-policy change is needed.

On macOS/Linux, create the environment with `python3 -m venv .venv` and replace `.venv\Scripts\python` with `.venv/bin/python` in every command.

The existing requirements specify `streamlit>=1.40,<2`, `scikit-learn>=1.5,<2`, `pandas>=2,<4`, `numpy>=1.26,<3`. These are version ranges, not an exact lockfile. Record the installed versions for research reproducibility:

```powershell
.venv\Scripts\python --version
.venv\Scripts\python -m pip freeze > environment_versions.txt
```

### 3.2 Download the data

Download **Food.com Recipes and Interactions** from https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions. Extract the eight files into `data/FoodCom/` (not a further nested folder). See Section 5 for their purposes. Respect the dataset's license and attribution requirements when distributing the project.

```powershell
.venv\Scripts\python inspect_foodcom.py
.venv\Scripts\python prepare_foodcom.py --size 20000 --seed 42
```

The first command inspects the schema; the second scans the raw recipes, validates and normalizes the ingredient lists, and writes a reproducible sample. It does not train a model. The inspection script expects the listed raw/preprocessed recipe and interaction CSVs; the converter itself requires only `RAW_recipes.csv`.

### 3.3 Train the models (optional for basic app use)

If matching trained bundles were not supplied, execute these commands in order:

```powershell
.venv\Scripts\python train_models.py --train-queries 1000 --test-queries 300 --seed 42
.venv\Scripts\python optimize_models.py --train-queries 2000 --validation-queries 200 --test-queries 500 --seed 123
.venv\Scripts\python evaluate_final.py
```

Do not run these blindly over an existing research installation. `train_models.py` can overwrite preliminary results; optimization and final evaluation refuse to overwrite their output directories. Section 8 shows safe isolated experiments. Finally launch the app using Section 2.

## 4. Using the application

| Control | What it does |
|---|---|
| Research dataset (sidebar) | Select the Food.com catalog or demonstration catalog. |
| Ingredients available | Search and select known ingredient names from the catalog. |
| Other ingredients | Enter additional comma-separated ingredient names; the same normalization is applied. |
| Recommendation algorithm | Select one of the available ranking methods. Current UI defaults to Hybrid, not FROR. |
| Maximum missing ingredients | Set 0–5; controls which incomplete recipes appear, default 2. |
| All extracted ingredients matched | Shows up to five recipes with zero missing listed ingredient types. |
| Almost available | Shows up to five recipes with 1 through the selected maximum missing ingredients. |
| Recipe expander | Shows ingredient list, coverage, missing items, score, source and cooking steps when available. |
| Add missing to grocery list | Inserts missing ingredients into the persistent list without duplicates. |
| Suggested staples | Suggests up to five additions based on recipes newly unlocked; selected sequentially. |
| Ingredient association rules | Shows up to 15 co-occurrence rules for inventory ingredients. |
| Grocery list | Add arbitrary items; check an item to delete it from the list. |

Example: select egg, milk, butter, flour, and salt if available in the catalog. Try Baseline, KNN and FROR with a missing limit of 2. Compare the two recipe sections, inspect a recipe, and add its missing items. Restart the server to confirm that grocery items persist.

The limit filters ranked recipes; it does not change how a model is trained. Each section currently shows five recipes, hard-coded as `subset.head(5)` in `app.py`. To display ten, change this to `subset.head(10)`; the research evaluation still measures top five. To change the slider range, edit `st.slider('Maximum missing ingredients',0,5,2)` in `app.py`. Neither UI change requires retraining. Do not confuse the number of displayed recommendations with the number of recipes in the catalog.

**Important:** the application tracks ingredient *types*, not weights, serving sizes, expiry dates or dietary safety. Food.com raw ingredient lists do not include quantities. A zero-missing result is not a guarantee that enough food is available or that the recipe is safe to cook; consult the original recipe. The grocery database is shared by users of the same local installation and has no account system.

## 5. Data and preprocessing

### 5.1 Why Food.com?

The earlier RecipeNLG catalog contained missing required ingredients, section headings and malformed extracted names. Our custom parser still produced kitchen equipment and incomplete phrases. We changed datasets to focus on recommendation research rather than ingredient extraction. Food.com supplies structured ingredient names, steps, recipe IDs and ratings. It is not error-free.

| Downloaded file | Purpose | Used by current code? |
|---|---|---|
| RAW_recipes.csv | Recipe IDs, titles, ingredient lists, steps, minutes, nutrition, tags | Yes: converter |
| RAW_interactions.csv | Original user ratings/reviews | Schema inspection only |
| PP_recipes.csv | Preprocessed recipe IDs, tokens, techniques and ingredient IDs | Schema inspection only |
| PP_users.csv | Preprocessed user information | No |
| ingr_map.pkl | Existing ingredient mapping | Existence/size checked only; not loaded |
| interactions_train.csv | Training user ratings and recipe IDs | Yes: rating features |
| interactions_validation.csv | Published rating validation split | Schema inspection only |
| interactions_test.csv | Published rating test split | Schema inspection only |

The current recipe split is our own synthetic inventory experiment, not the published interaction split. Never unpickle an untrusted file; pickle and joblib may execute arbitrary code.

### 5.2 Conversion pipeline

`prepare_foodcom.py` reads Python-literal lists using `ast.literal_eval`, checks IDs, names, ingredients and instructions, rejects invalid records, and removes repeated IDs. It calls `normalize_items()` and uses seeded reservoir sampling, which scans the full raw catalog and selects up to the requested count without keeping all valid recipes in memory. Original structured names and steps remain available for display.

`ingredient_names.py` uses conservative aliases and known final-word plurals (egg/eggs, garlic cloves/garlic, scallions/green onion, pie crust/crusts), selected equipment exclusions and known composite entries (salt and pepper). It keeps distinct products such as garlic powder versus garlic and baked versus unbaked pie crust separate. It does not use a trained NLP parser or Food.com's official ingredient IDs. Some duplicate variants and incorrect names may remain. Changing normalization can change scores, even if the dataset filename is unchanged; retrain and document the revision.

Generated catalog columns: `name` (recipe title), `ingredients` (pipe-separated normalized names), `original_ingredients` (JSON list of raw names, not quantities), `directions` (JSON list of steps), `source_link` (Food.com URL), `source`, `recipe_id`, and `minutes`. The report records counts, sampling seed and method. The vocabulary CSV counts recipes containing each normalized name; counts may overlap across ingredients.

The initial raw conversion scanned 231,637 recipes, retained 231,635 valid records, and sampled 20,000. It produced 7,252 unique raw ingredient names before the later normalization patch. These historical counts are not guaranteed for a different source version or sample.

## 6. How the recommendation models work

### 6.1 Common representation and features

Let A be the set of available ingredient types and B a recipe's required set. Let o = |A ∩ B|, a = |A| and b = |B|. `Engine` creates a sparse binary ingredient matrix using `MultiLabelBinarizer` and a separate sparse TF-IDF matrix. Sparse matrices avoid storing huge arrays of zeros. A user inventory is normalized and transformed into matching vectors; ingredients unknown to the catalog cannot match but still contribute to inventory size.

| Feature | Definition and reason |
|---|---|
| Coverage C | o / b; measures how much of the recipe is available. |
| Utilization U | o / max(a, 1); measures how much of the inventory is used. |
| Cosine T | Dot product of unit-normalized TF-IDF vectors; rewards informative matching ingredients. |
| Jaccard J | o / max(a + b − o, 1); balances overlap against total distinct ingredients. |
| Missing M | b − o; approximates additional grocery requirements. |
| Recipe size | b; helps distinguish small and complex recipes. |
| Overlap | o; absolute number of matches. |
| Minutes | Numeric preparation time, clipped to 0–1440; missing/invalid becomes zero. |
| Historical rating | (sum of valid training ratings + 15) / (count + 5); shrinks sparse ratings toward 3. |
| Rating count | log(1 + count); limits the effect of very popular recipes. |

Ratings are 1–5 from the training interaction file only; zero ratings are excluded. If the ratings file or recipe IDs are missing, the code uses neutral rating 3 and count feature 0. Features are computed in `learned_ranker.py` in a fixed order shared by training and inference.

### 6.2 The seven recommendation methods

**Baseline — rule-based ingredient coverage.** Score = C. A recipe with 4 of 5 ingredients scores 0.8. This directly rewards recipes that need little shopping, but can favor small recipes and overlook how much of a large inventory gets used. It requires no model training and is our comparison reference.

**TF-IDF — content-based similarity.** Each distinct normalized ingredient phrase is a token. Scikit-learn's default smoothed inverse document frequency is `log((1 + N)/(1 + df)) + 1`, where N is catalog size and df is the number of recipes containing a token. TF-IDF vectors are L2-normalized; cosine similarity compares the direction of inventory and recipe vectors. Rare matches may be more informative than ubiquitous salt or water. It does not explicitly penalize the number of missing ingredients. The vectorizer is fitted on the searchable catalog, not trained on relevance labels.

**KNN — exact nearest-neighbor retrieval using Jaccard similarity.** Score = J. The code calculates similarity to all catalog recipes and sorts them; it does not instantiate a scikit-learn KNeighborsClassifier, use neighbor voting or train a predictive classifier. This approach is called KNN because the highest-ranked recipes are the nearest neighbors of the inventory. It balances recipe and inventory overlap, but may rank recipes with several missing ingredients highly. The top five are selected for research metrics.

**Hybrid — manually weighted ranking.** Score = 0.40T + 0.25U + 0.35C. Combines informative ingredient similarity, inventory utilization and recipe completeness. The original weights were chosen by design rather than learned. `Engine.rank()` accepts an optional `weights` tuple in that order and divides the weighted sum by its total. The standard training/evaluation scripts use the fixed weights shown above; changing only the app/engine does not change those experiments.

**Logistic regression — supervised binary classifier used as a ranker.** StandardScaler centers and scales the ten features, then logistic regression learns coefficients and an intercept. Its score is sigmoid(bias + sum(coefficient × standardized feature)). Higher predicted positive-class probability ranks a recipe higher. Class weighting balances the rare synthetic positive class. L2 regularization limits large coefficients; smaller C means stronger regularization. Phase 5 tests C = 0.1, 1, 10, maximum 1,000 iterations, and selects highest validation MRR. The selected C was 0.1 (the three validation MRR values tied; first maximum wins). The score is not a calibrated probability that a real user will like or cook a recipe.

**Random forest — supervised ensemble classifier.** A forest of 100 decision trees learns nonlinear combinations and thresholds of the same features. Trees use randomized samples and feature subsets; the model averages tree class probabilities. It can capture interactions that a linear model cannot, but needs more computation. Phase 4 uses minimum leaf size 5; Phase 5 tests 3, 10, 25 and selects highest validation MRR. Larger leaves reduce overly specific splits. Selected minimum leaf size was 10. Balanced class weights and `n_jobs=-1` are used.

**FROR (Fridge Rescue Optimized Ranker) — validation-optimized multi-objective score.** Five components are combined:

```text
S = wR*R + wJ*J + wC*C + wU*U - wM*min(M/5, 1)
R = logistic regression positive-class score
```

The 70 candidate weight vectors have nonnegative entries in increments of 0.25 summing to 1. Each is evaluated on the same validation inventories. The winning vector maximizes:

```text
Q = MRR@5 + 0.5*mean_utilization - 0.5*mean_missing/5 + 0.25*complete_rate
```

The utility is a research design choice, not a measured user preference. Note the distinction: the per-recipe missing penalty is capped at five; the aggregate utility's missing term is not capped. The final selected weights in component order (R,J,C,U,M) were `(0, 0.25, 0, 0.50, 0.25)`; thus the selected FROR uses Jaccard, utilization and missing penalty, not supervised predictions. The code still computes the logistic score even when its weight is zero. FROR is a tuned ranking algorithm, not itself a supervised classifier. The two supervised models remain separate research contributions.

All methods sort descending by score, then ascending by missing count, then by catalog row order for ties. The UI's separate complete and near-match categories are applied after ranking, whereas research metrics use the unfiltered top five.

### 6.3 Worked example

Inventory: egg, milk, butter, flour, onion (five types). Recipe A: egg, milk, flour, sugar (four types). Overlap = 3, coverage = 3/4 = 0.75, utilization = 3/5 = 0.60, Jaccard = 3/6 = 0.50, missing = 1 (sugar). Baseline score = 0.75. With the selected FROR weights the score is `0.25*0.50 + 0.50*0.60 - 0.25*(1/5) = 0.375`. TF-IDF and learned predictions depend on the fitted catalog and model. Scores across methods are not directly comparable as probabilities.

### 6.4 Other intelligent features

Staple suggestions greedily select an ingredient that completes the largest number of currently one-missing-ingredient recipes, breaking ties by catalog frequency and then deterministic name order. The chosen ingredient is added to a temporary inventory before selecting the next suggestion; the real inventory is not modified. The count is sequential, not an independent estimate for each item.

Association rules use recipe co-occurrence: support = count(A and B)/N; confidence = count(A and B)/count(A); lift = confidence / (count(B)/N). Lift above 1 indicates more co-occurrence than expected under independence. Default minimum support is 5%; rare pairs are omitted. This is association, not causation or a guarantee ingredients are interchangeable.

## 7. Training and reproducing experiments

### 7.1 What is a training example?

Food.com does not record actual household inventories. The scripts simulate one by choosing a target recipe, retaining ceil(40–90% of its ingredients) with at least one retained, then adding 2–8 randomly chosen catalog ingredient types not already retained (bounded by availability). These extras can include omitted target ingredients. Each target recipe is labeled positive. Up to 12 high-Jaccard non-target candidates and up to 24 randomly selected non-target candidates from the training partition are combined and labeled zero for training. These are proxy negatives, not confirmed human dislikes. The two supervised classifiers learn to distinguish synthetic target recipes from these alternatives.

### 7.2 Preliminary experiment (Phase 4)

```powershell
.venv\Scripts\python train_models.py --train-queries 1000 --test-queries 300 --seed 42 --output ml_results
```

The seed creates a 70/30 target partition (14,000/6,000 for the 20,000 catalog). Target sampling is with replacement. The whole catalog is searched at test time. This produces six-method metrics and paired bootstrap intervals against Baseline. Since results were inspected to develop FROR, this is preliminary evidence, not a fresh final test.

### 7.3 Optimization (Phase 5)

```powershell
.venv\Scripts\python optimize_models.py --train-queries 2000 --validation-queries 200 --test-queries 500 --seed 123 --previous-seed 42 --previous-train-queries 1000 --previous-test-queries 300 --previous-bundle ml_results/rankers.joblib --output optimized_results
```

The previous dataset fingerprint is checked and the preliminary RNG draws replayed. Previously tested targets are excluded from the final target partition, which is selected from the old test partition; new training targets come from the old training partition. The remaining rows form validation. For the reported run, this yields 12,000/4,000/4,000 target rows, and identifies 298 distinct previously tested targets. The model-bundle check verifies catalog bytes, not that the supplied earlier seed and query counts are correct; supply the actual original settings. Replay assumes identical catalog order, normalization, relevant code and compatible RNG behavior.

Training targets are drawn with replacement. Validation targets are drawn without replacement, up to the partition size. Six supervised hyperparameter configurations are trained and the best of each type selected by validation MRR@5. FROR weights are selected by validation utility. No final test metrics are used to select these parameters.

### 7.4 Held-out evaluation

```powershell
.venv\Scripts\python evaluate_final.py --dataset recipes_foodcom.csv --output optimized_results
```

This loads the frozen model, protocol and saved partitions, checks fingerprints and partition disjointness, and evaluates all seven methods on the same 500 synthetic inventories (distinct test targets). It saves per-query metrics, mean results, 2,000-resample paired bootstrap intervals for all method pairs and feature importance. The entire catalog remains searchable, so this is held-out target retrieval, not evaluation on a wholly unseen catalog. Do not tune using the final results and then report the same test as independent evidence.

## 8. Changing recipe amounts and experiment settings

### 8.1 Command reference

Run any script with `--help` to see supported flags (except `inspect_foodcom.py` and the legacy audit script, which have their own interfaces).

| Script | Flag | Default | Meaning |
|---|---|---|---|
| prepare_foodcom.py | --folder | data/FoodCom | Raw source directory |
| | --output | recipes_foodcom.csv | Output catalog path |
| | --size | 20000 | Maximum recipes sampled |
| | --seed | 42 | Sampling seed |
| train_models.py | --dataset | recipes_foodcom.csv | Input catalog |
| | --ratings | data/FoodCom/interactions_train.csv | Training ratings |
| | --output | ml_results | Preliminary output folder (may overwrite) |
| | --train-queries | 1000 | Number of synthetic training inventories |
| | --test-queries | 300 | Number of preliminary test inventories |
| | --seed | 42 | Random seed |
| optimize_models.py | --dataset / --ratings | same as above | Catalog and ratings |
| | --output | optimized_results | New output folder; must not exist |
| | --previous-bundle | ml_results/rankers.joblib | Matching preliminary bundle |
| | --previous-seed | 42 | Actual preliminary seed |
| | --previous-train-queries | 1000 | Actual preliminary training query count |
| | --previous-test-queries | 300 | Actual preliminary test query count |
| | --seed | 123 | Optimization seed |
| | --train-queries | 2000 | Training inventory count |
| | --validation-queries | 200 | Validation inventory count |
| | --test-queries | 500 | Saved final inventory count |
| evaluate_final.py | --dataset | recipes_foodcom.csv | Same catalog used for optimization |
| | --output | optimized_results | Existing optimization folder; final/ must not exist |
| evaluate.py | --dataset | recipes.csv | Legacy catalog |
| | --trials | 300 | Legacy simulated inventories |
| | --seed | 42 | Legacy seed |
| | --output | results | Legacy results folder (may overwrite) |

Keep query counts positive. Optimization caps validation and final test counts at their partition sizes; preliminary training/testing uses replacement. `--size` is not a training-query count. Increasing the catalog increases the number of candidate recipes per query and may increase RAM/time substantially.

### 8.2 Safe isolated experiment: 5,000 recipes

The following commands do not replace the official catalog or trained model folders. Use a new experiment directory name each time.

```powershell
New-Item -ItemType Directory -Path experiments\sample5000
.venv\Scripts\python prepare_foodcom.py --size 5000 --seed 42 --output experiments/sample5000/recipes_foodcom.csv
.venv\Scripts\python train_models.py --dataset experiments/sample5000/recipes_foodcom.csv --train-queries 200 --test-queries 50 --seed 42 --output experiments/sample5000/preliminary
.venv\Scripts\python optimize_models.py --dataset experiments/sample5000/recipes_foodcom.csv --previous-bundle experiments/sample5000/preliminary/rankers.joblib --previous-seed 42 --previous-train-queries 200 --previous-test-queries 50 --train-queries 400 --validation-queries 50 --test-queries 100 --seed 123 --output experiments/sample5000/optimized
.venv\Scripts\python evaluate_final.py --dataset experiments/sample5000/recipes_foodcom.csv --output experiments/sample5000/optimized
```

Each command above is one line. Wait for completion before the next. The UI does not automatically discover experimental folders; use the Python engine directly or copy the latest application code to a separate project root with that experiment's catalog at `recipes_foodcom.csv` and matching bundles in the expected locations. Do not replace the official experiment to preview an alternate catalog.

### 8.3 Other useful experiment sizes

| Purpose | Recipe count | Preliminary train/test | Optimized train/validation/test |
|---|---:|---:|---:|
| Quick pipeline check | 1,000 | 100/30 | 200/30/50 |
| Small experiment | 5,000 | 200/50 | 400/50/100 |
| Reported study | 20,000 | 1,000/300 | 2,000/200/500 |
| Scaling experiment | 50,000 | 1,000/300 | 2,000/200/500 |
| Full catalog exploration | 231,637 requested | Start small | Start small |

Use Section 8.2 as a template, changing all paths and matching previous settings. These alternative sizes are suggestions, not verified benchmarks. Actual catalog size can be smaller than requested. Very small catalogs may fail split replay because not enough unused targets remain. Recipe counts alone do not establish a fair model comparison: keep test generation and evaluation protocol consistent; compare on shared queries when possible. A different catalog requires new matching bundles. Do not reuse a model from the 20,000 catalog on the 5,000 catalog.

### 8.4 Advanced code experiments

| Change | Where | Effect |
|---|---|---|
| Hybrid weights | engine.py; matching score expressions in train_models.py and evaluate_final.py | Alters manual balance; keep all versions consistent |
| Logistic C values, forest leaf size/trees | optimize_models.py candidate definitions | Alters supervised search; train again |
| FROR weight grid | optimize_models.py `weights` expression | Alters candidate count and granularity |
| FROR objective | optimize_models.py `utility` expression and protocol string | Alters what is optimized; record before evaluating |
| Inventory noise / retained fraction | train_models.py `inventory()` | Changes task difficulty; affects replay; new experiment required |
| Top-k research metric | optimized_ranker.py `metrics()` and preliminary evaluation logic | Changes metric meaning; update names, outputs and documentation consistently |
| Ingredient aliases | ingredient_names.py | Changes matching; regenerate dataset and retrain |

Use version control or copies before changing code. For new hypotheses or weight tuning, use validation data; reserve a fresh evaluation set for any modified final algorithm. The current code has no dedicated ablation or end-to-end benchmark script yet.

## 9. Testing and interpreting results

### 9.1 Beginner functional test checklist

- [ ] Run `python -m pip check` with the virtual-environment interpreter; resolve missing dependencies.
- [ ] Launch the demo catalog, choose a few ingredients and verify all four basic methods return results.
- [ ] Launch the Food.com catalog and verify the expected recipe count and seven model options when the optimized bundle is present.
- [ ] Enter `eggs, garlic cloves, potatoes` and verify matching uses egg, garlic and potato; garlic powder must remain distinct.
- [ ] Change maximum missing ingredients to 0; no incomplete recipes should be displayed.
- [ ] Change the limit to 2; every recipe in Almost available must have one or two missing names.
- [ ] Inspect cooking steps and original names, and verify the source link refers to the correct recipe.
- [ ] Add missing ingredients to groceries twice; duplicate items should not appear. Add an item manually, restart, then check it to remove it.
- [ ] Compare the same inventory with Baseline, KNN and FROR; their ordering can differ, but each recipe's missing set should be consistent.
- [ ] Inspect staples and association rules. These can be empty or have zero gains depending on inventory.
- [ ] Run the isolated small experiment in Section 8 and check all expected output files.

### 9.2 Direct engine and normalization checks

These commands use the installed Python environment and can run without the browser:

```powershell
.venv\Scripts\python -c "from ingredient_names import normalize_items; assert normalize_items(['eggs','egg','garlic cloves','garlic powder']) == {'egg','garlic','garlic powder'}; print('Normalization OK')"
.venv\Scripts\python -c "from engine import Engine; e=Engine('recipes_foodcom.csv'); print(len(e.df),len(e.vocab)); print(e.rank(['egg','milk','butter'],'Baseline')[['name','coverage','missing_count']].head())"
.venv\Scripts\python -c "from engine import Engine; e=Engine('recipes_foodcom.csv'); e.load_learned('optimized_results/rankers.joblib'); print(e.rank(['egg','milk','butter'],'FROR (optimized)')[['name','score','missing_count']].head())"
```

For a quick legacy comparison on the demo catalog:

```powershell
.venv\Scripts\python evaluate.py --dataset recipes.csv --trials 30 --seed 42 --output demo_test_results
```

This legacy four-method experiment uses a different inventory generator (roughly 60% retained plus up to four distractors outside the target recipe) and must not be pooled with Phase 4/5 results.

### 9.3 Research metrics

| Metric | Interpretation |
|---|---|
| Hit@5 | Fraction of queries where the synthetic target appears in the first five; higher is better. |
| MRR@5 | Average reciprocal target rank (1/rank for ranks 1–5, zero otherwise); higher is better. |
| Missing | Mean number of missing ingredient types across top-five recipes, then across queries; lower is better. |
| Utilization | Mean fraction of available ingredient types used across top-five recipes and queries; higher is better. |
| Complete rate | Mean proportion of top-five recipes with zero missing listed ingredient types; higher is better. Not percentage of users who can cook a meal. |
| Ranking ms | Time for scoring and top-five metrics; excludes shared feature extraction and UI overhead; lower is better. |
| Utility Q | Composite of MRR, utilization, missing and complete rate (Section 6); higher is better under this chosen objective. |

In paired intervals, `difference = method - reference`. A negative missing difference favors the method, whereas a positive Hit@5 difference favors it. `low` and `high` are individual 95% percentile bootstrap bounds obtained by resampling paired queries. Intervals excluding zero suggest a difference under the simulated conditions, but multiple comparisons are not adjusted. Composite utility confidence intervals have not been generated by the current scripts; do not combine marginal confidence bounds to infer them.

Feature importance: random forest importance is impurity-based and normalized across features; logistic coefficients describe standardized feature effects conditional on other features. Correlated features can share or distort importance. Neither measure proves causation.

## 10. File reference

### 10.1 Active source files

| File | Responsibilities and dependencies |
|---|---|
| app.py | Streamlit entry point; imports Engine; caches catalog and models by file modification time; handles inventory, recipe categories, suggestions and SQLite grocery operations. |
| engine.py | Engine class; loads CSV, constructs sparse matrices, scores and sorts recipes, computes missing sets, staples and co-occurrences; imports ingredient_names and optionally learned/optimized modules. |
| ingredient_names.py | Conservative aliases, plural mapping, equipment exclusions and selected compound splitting; shared by preprocessing and engine. |
| inspect_foodcom.py | Prints first three examples and column names for six raw/preprocessed CSVs; records pickle presence without loading it. Writes foodcom_schema.json. |
| prepare_foodcom.py | Validates raw records, normalizes and samples, writes catalog/report/vocabulary. |
| audit_dataset.py | Legacy ingredient vocabulary audit; flags selected suspicious entries and produces review statistics. Candidates are not definitive errors. |
| evaluate.py | Earlier four-method simulated retrieval study, separate protocol. |
| learned_ranker.py | FEATURES order, shared feature computation, SHA-256 catalog fingerprint. |
| train_models.py | Inventory generator, historical rating features, preliminary split, supervised training, six-method evaluation and baseline intervals. |
| optimized_ranker.py | FROR components, score calculation, deterministic top-k sorting and common metrics. |
| optimize_models.py | Replays preliminary target sampling, creates disjoint partitions, trains candidate models, chooses parameters and weights on validation, saves bundle/protocol/partitions. |
| evaluate_final.py | Loads frozen optimized bundle, checks partition and dataset consistency, evaluates seven methods, saves timing, intervals and feature importance. |
| requirements.txt | Dependency version ranges. |
| README.md | Main installation, operation and research reference. |
| README_ML.md | Earlier supervised training notes; this main guide supersedes conflicting instructions. |
| README_OPTIMIZATION.md | Earlier optimization notes; this main guide supersedes conflicting instructions. |

### 10.2 Generated data and application state

| File | Contents / handling |
|---|---|
| recipes.csv | Small original demonstration catalog (32 recipes in the supplied version). Keep for quick UI checks. |
| recipes_foodcom.csv | Active normalized catalog. Preserve exact bytes for existing trained models. |
| recipes_foodcom.report.json | Raw conversion counts, seed, size and method. |
| recipes_foodcom.vocabulary.csv | Ingredient names and per-recipe frequencies. |
| foodcom_schema.json | Dataset schema and examples. |
| grocery.db | SQLite table `grocery(item TEXT PRIMARY KEY)`; local grocery state. Back up if needed; removing it resets the list when app next creates it. |
| .venv/ | Installed Python environment; recreate on another computer instead of copying it. |
| __pycache__/ | Automatically generated bytecode; safe to remove when program is stopped. |
| environment_versions.txt | Optional dependency snapshot produced in Section 3. |

### 10.3 Research output directories

`ml_results/` contains `rankers.joblib` (two trained classifiers, ratings array, fingerprint and feature order), `summary.csv` (six-method means), `trials.csv` (per-query metrics), `paired_intervals.csv` (comparisons to baseline), and `protocol.json` (settings and interpretation).

`optimized_results/` contains `rankers.joblib` (optimized classifiers plus FROR weights), `model_validation.csv` (six parameter configurations), `weight_validation.csv` (70 weight combinations with validation metrics and utility), `protocol.json` (seeds, actual partition/query counts, chosen weights/parameters and dataset hash), `partitions.npz` (train, validation, test and previously seen row indices), and `final/`:

| Final file | Purpose |
|---|---|
| summary.csv | Seven-method means including ranking-stage timing. |
| trials.csv | One row per query and method (3,500 in reported run). Query and target identify paired comparisons. |
| paired_intervals.csv | Pairwise bootstrap differences for the five quality metrics. |
| feature_importance.csv | Random forest feature importances and standardized logistic coefficients. |

Keep both research directories unchanged for the final presentation. Never load joblib bundles from unknown sources. Do not rename or reorder the catalog and expect arbitrary models to work; the app checks the hash.

### 10.4 Archived files (not required to run)

| File or directory | Historical purpose |
|---|---|
| full_dataset.csv | Original RecipeNLG source, over two million recipes. |
| prepare_dataset.py | Earlier source filtering, validation, deduplication and sample generation. |
| recipes_full.csv / recipes_full.report.json | Earlier 20,000-recipe sample and preparation report. |
| ingredient_parser.py | Experimental rule-based extraction from original ingredient descriptions. |
| rebuild_ingredients.py | Reconstructed earlier catalog and generated audit/annotation files. |
| test_ingredient_parser.py | Unit tests of retired parser. |
| recipes_nlp.csv | Retained reconstructed recipes (11,051 in prior run). |
| recipes_nlp.report.json | Reconstruction statistics. |
| recipes_nlp.review.csv | Ambiguous recipes held for review. |
| recipes_nlp.vocabulary.csv | Reconstructed ingredient vocabulary. |
| recipes_nlp.annotation.csv | Human annotation sample template. |
| smoke_results/ | Earlier demo evaluation output. |
| dataset_audit/ (if generated) | Vocabulary, flagged review candidates and audit summary. |

The current app/engine no longer imports the retired RecipeNLG modules. Archive them for research provenance, not execution. Old patch ZIPs are installation history, not runtime dependencies. Phase 6 scripts discussed previously (analyze_results.py, ablation_study.py, benchmark.py, test_application.py) are planned, not implemented in the documented release.

## 11. Troubleshooting and maintenance

| Problem | Resolution |
|---|---|
| Python or pip not recognized | Install Python, reopen terminal, create .venv; use explicit .venv interpreter path. |
| ModuleNotFoundError | Install requirements in the same environment used to launch the program. Check that latest patch modules exist. |
| PowerShell activation blocked | Use explicit `.venv\Scripts\python` commands instead of activating. |
| FileNotFoundError for Food.com | Run from project root; verify data/FoodCom/RAW_recipes.csv and other required CSV filenames; remove extra nesting. |
| Only demo appears | Generate recipes_foodcom.csv in the root; restart app. |
| Only four models appear | Check matching model bundle paths and selected dataset. Train if needed. Demo only offers four. |
| Model does not match dataset | Restore exact original CSV or train a new matching model. Do not bypass fingerprint checks. |
| Previous dataset differs | Prior bundle is for a different dataset; rerun the preliminary experiment on the new catalog in a new directory. |
| Not enough previously unused target recipes | Increase catalog size or reduce preliminary test queries in a new experiment; supply matching replay arguments. |
| FileExistsError during optimization/final evaluation | Output is protected; use a new experiment directory. If evaluation failed after creating final/, preserve/rename the incomplete folder before retrying the frozen run. |
| Training slow or high RAM use | Reduce catalog/query counts in an isolated experiment; monitor system resources. Forest prediction uses CPU threads; no GPU setting is required. |
| Joblib/scikit-learn compatibility warning | Recreate original package versions or retrain; do not assume old binary models are portable across versions. |
| Ingredients seem duplicated | Verify normalization update installed; do not blindly merge distinct food products. Document remaining variants. |
| No complete recipes | Add real available ingredients, try a different inventory, or increase missing limit for near matches. Zero matches is valid. |
| No ratings file | Models use neutral rating features; restore the training CSV for reproducing the reported experiment. |
| Browser does not open / port in use | Open URL printed by Streamlit; optionally use `--server.port 8502` after app.py. |
| Grocery items disappear | Check that the same project's grocery.db is used; the checkbox deletes the item. |

Before transferring the project: include active scripts, requirements, demo catalog, main README, authorized data or download instructions, exact catalog and matching trusted models if redistribution permits, and research output CSV/JSON files. Exclude `.venv` and caches; avoid sharing personal grocery data. Record package versions. Keep backups of original result directories before experiments. A new developer should be able to run the demo without the full dataset or models.

## 12. Research findings and limitations

### 12.1 Reported final experiment (user-provided Phase 5 results)

Catalog: 20,000 recipes; 2,000 training queries; 200 validation queries; 500 held-out test queries. Seeds 42 (preliminary), 123 (optimization). Selected logistic C=0.1 and forest minimum leaf size=10. FROR weights: [0, 0.25, 0, 0.5, 0.25].

| Method | Hit@5 | MRR@5 | Missing | Utilization | Complete rate | Ranking ms |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 82.40% | 0.6639 | 1.4044 | 24.56% | 7.08% | 1.102 |
| TF-IDF | 94.80% | 0.9046 | 4.2008 | 28.43% | 2.92% | 1.298 |
| KNN | 98.80% | 0.9763 | 3.8884 | 36.66% | 2.48% | 1.052 |
| Hybrid | 98.80% | 0.9532 | 2.7804 | 32.90% | 6.48% | 1.300 |
| Logistic regression | 98.40% | 0.9596 | 2.9592 | 32.51% | 5.44% | 3.002 |
| Random forest | 97.40% | 0.9523 | 2.2240 | 28.02% | 4.00% | 40.231 |
| FROR | 95.60% | 0.9314 | 1.9648 | 29.85% | 5.24% | 2.898 |

FROR reduced average missing ingredients by 29.33% versus Hybrid (paired absolute difference −0.8156, individual 95% interval −0.9000 to −0.7324) and 49.47% versus KNN. Against Baseline, FROR improved Hit@5 by 13.2 percentage points but required 0.5604 more missing ingredients on average. FROR achieved two of five proposed targets (MRR ≥0.93 and missing <2); it missed retrieval ≥97%, utilization ≥32% and complete rate ≥6.27%. Its selected formula does not use learned relevance. Applying the predefined composite utility to these means gives FROR approximately 0.8973, the highest observed utility; its statistical superiority on composite utility has not been established.

### 12.2 What can and cannot be concluded

The experiments demonstrate differences in synthetic target retrieval, ingredient-type utilization and missing ingredient counts. They do not establish reductions in actual household food waste, money spent, shopping time or meal-planning difficulty. Ratings are not household inventory observations. The simulated target construction favors overlap-based methods, and other recipes labeled negative may actually be suitable. The full catalog and its TF-IDF statistics are available across partitions; final targets are held out from supervised target training and earlier tested targets, not unseen documents. The recipe quantity/availability limitations and imperfect normalization remain. The UI's filtered categories are not the same evaluation task as unrestricted top-five retrieval. Timing is specific to the measured ranking stage on the original computer.

Potential future work: paired utility intervals using saved trials, validation-only ablation analysis, end-to-end latency measurement, and a controlled human evaluation with appropriate instructor/ethics approval and independent relevance labels. Do not treat these as completed work or change the final model using the existing final test scores without reserving new held-out cases.

### 12.3 Reproducibility and provenance

Dataset: Food.com Recipes and Interactions, Shuyang Li et al., Kaggle (link in Section 3). Related research: Li, S., Li, Y., and McAuley, J. (2019), *Generating Personalized Recipes from Historical User Preferences*. Software uses scikit-learn, NumPy, pandas and Streamlit. For exact reproduction preserve the dataset bytes, script versions, seeds, model bundles and dependency versions. The reported catalog SHA-256 is `841a46530f14f0c9d42b2c19b23428e8608030e4ea7df16689d32a4d3e0654d6`.

This guide was prepared from the supplied Phase 5 source code, prior project patches and reported experiment outputs. The current local Windows installation and full Food.com dataset were not available for a fresh end-to-end run during documentation. The test checklist and alternate experiment sizes are instructions, not claims that those tests have been executed. If your local code differs, inspect the actual script's `--help` and preserve the research version before editing.

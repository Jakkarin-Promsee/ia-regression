# IA Regression — extended methodology

Companion to [`README.md`](README.md): start there for **setup**, **repository layout**, and a tight summary of **Indicator-augmented regression (IA)**. This document is the **full** modeling walkthrough—filters, mask training, code sketches, baselines, and evaluation.

## Indicator-augmented regression (IA)

**IA** = **I**ndicator-**A**ugmented (the project nickname plays on “AI,” but the acronym is literal).

We predict the full **50-item** response vector from **partial** quizzes by regressing on **100 features**: **[masked values ∥ observation indicators]**. Each missing item is **zero-padded** in the value block; the matching **binary indicator** is 0 when unasked and 1 when answered. That separation is what lets linear Ridge distinguish **“not asked yet”** from **“asked and scored low.”** In missing-data terms, the indicators carry the **observation pattern**; in regression terms, they are **missingness / inclusion indicators** paired with the padded design vector—hence _indicator-augmented_ regression. Training uses **random-K masking** (see below) so one estimator is calibrated for every completion count _K_.

---

## Project context

A data-modeling project that turns the IPIP Big Five personality test into a meme-style
quiz where **users can quit at any time** and still get a meaningful result. The more
questions they answer, the more precise the result becomes — but even at 5 answers the
model produces a full personality readout for all 50 items.

Built on ~1M responses from Kaggle, with a Ridge regression trained via a _random-mask_
trick that lets a single linear model handle every possible "amount of information"
the user might give us.

---

## The Big Idea

Traditional personality quizzes force you through all 50 questions or give you nothing.
This one doesn't:

- **Quit anytime.** Answer 5, 13, 30, or all 50. You always get a full result page.
- **Precision grows live.** A precision bar shows the user how confident the model is.
- **Soft limit at 30.** Empirically, accuracy plateaus around 30 answers — so the UI
  prompts the user to stop there if they want, but they can keep going.
- **Randomized order each session.** Repeat users get a fresh experience.

### The result page — always shows all 50 items

| Tier                                | What                          | When shown |
| ----------------------------------- | ----------------------------- | ---------- |
| **Primary** (7 meme personalities)  | The branded headline labels   | Always     |
| **Secondary** (43 supporting items) | Less prominent flavor results | Always     |

Items the user actually answered are shown as known ("you said X"); the rest are filled
in by the model ("we think X"). Same layout for every K — no unlocks, no FOMO.

### The 7 meme personalities (Tier 1, the headline)

Each meme maps to a single Big-5 question. High agreement on that question triggers the
meme label.

| Question                                         | Meme                     |
| ------------------------------------------------ | ------------------------ |
| `CSN6` "I often forget to put things back"       | Frequently Forgets Items |
| `AGR9` "I feel others' emotions"                 | Empath                   |
| `AGR5` "I am not interested in others' problems" | Cold Heart               |
| `EXT4` "I keep in the background"                | Wallflower               |
| `EST7` "I change my mood a lot"                  | Mood Roller-Coaster      |
| `OPN3` "I have a vivid imagination"              | Daydreamer               |
| `OPN10` "I am full of ideas"                     | Dare to Dream            |

(Naming is illustrative — pick whatever fits your branding.)

---

## Why partial-answer prediction works

Big-5 questions are _highly redundant_ by design — each trait has 10 items measuring
roughly the same underlying signal. Cell 30–32 of the notebook confirms this on the
real data: questions within a trait are strongly correlated, and even across traits
there are weaker but real correlations.

So a user's full 50-dimensional answer vector is mostly determined by ~5 latent values
(the Big-5 traits) plus noise. Given any subset of K answers, those latent values are
already pinned down — you just need a model that knows how to read them out from
whichever subset the user happened to provide.

That's the slack the ML exploits.

---

## How the Model Works

### Step 0 — Data quality (the single biggest unlock)

Before any modeling, we drop respondents who almost certainly weren't paying attention.
Three rules applied jointly:

| Filter          | Rule                       | Why                                                                                    |
| --------------- | -------------------------- | -------------------------------------------------------------------------------------- |
| Unique IP       | `IPC == 1`                 | Drops university-batch / re-submitter spam (recommended in the dataset README itself). |
| Reasonable time | `60s ≤ testelapse ≤ 1800s` | Drops speed-clickers and people who left the tab open all afternoon.                   |
| Real variance   | `std(50 answers) > 0.3`    | Drops "all 3s", "all 5s", and other zero-effort patterns.                              |

This step happens **before** the column-drop, while the metadata is still available, and
keeps roughly 60-75% of the original ~1M rows. The kept rows are dramatically higher
quality — and even the best model can't denoise its own training signal, so this turns
out to be the single biggest precision improvement in the whole pipeline.

### Inputs and targets

For each user we build a 100-dimensional feature vector:

```
[ X_masked (50 values, 0 where missing) | mask (50 binary, 1 where answered) ]
```

The mask is the key trick: it tells the model "ignore the 0 at position 4, I made it up"
so missing values don't get confused with low real scores.

Target: the original 50 answers. So we're doing **matrix completion** — given partial
rows, reconstruct full rows.

### The rank-based `random_mask` algorithm (implementation in `main.ipynb`)

Training and evaluation need to mimic real users: each person answers **exactly K** of the 50 items, on a **random subset** (order does not enter the model—only which items are observed). The notebook implements this **without** nested Python loops over questions by using **per-row random ranks**:

1. Draw i.i.d. `scores ~ Uniform(0,1)` for each of the 50 columns and each row.
2. Convert to **ranks** `1 … 50` within each row (`argsort` twice gives stable ordinal ranks).
3. Draw one integer **K** per row: `K ~ UniformInteger(k_min, k_max)` (inclusive).
4. Set **`mask[i, j] = 1`** iff **`rank[i, j] ≤ K[i]`** — the **K smallest ranks** are the “answered” items. Multiply **`X_masked = X * mask`** so unasked positions are 0.

That procedure yields a **uniform random K-subset** of columns per row (each of the C(50,K) subsets equally likely for fixed K). When **`k_min < k_max`**, each training row gets its **own random K** between 5 and 50, so one Ridge fit sees every information level in one pass. When **`k_min == k_max == k`**, every row has **exactly k** observed items — used inside **k-iterator loops** for validation and test metrics.

```python
def random_mask(X, k_min=5, k_max=50, seed=None):
    rng = np.random.default_rng(seed)
    n, q = X.shape
    scores = rng.random((n, q))
    ranks = np.argsort(np.argsort(scores, axis=1), axis=1) + 1  # 1..q
    ks = rng.integers(k_min, k_max + 1, size=n).reshape(-1, 1)
    mask = (ranks <= ks).astype(np.float32)
    return X * mask, mask
```

### The random-K training trick

For **training**, the notebook calls `random_mask(X_train, k_min=5, k_max=50, …)` once: each row keeps a random count K and a random which-K set. The target **Y** is still the full 50 answers (`Ytr = X_train`). That prevents the model from only learning the identity map on fully observed rows and forces it to impute under missingness patterns that match deployment.

If we trained only on full inputs, the model would learn to copy the input perfectly
and fall apart at inference when entries are missing.

### Where the **k iterator** loops appear

**Hyperparameter search (`ridge_validation_mae`).** After fitting Ridge on the masked training matrix, the notebook averages validation MAE over **fixed-K** masks. For each **`k` in `(5, 10, 20, 30, 40, 50)`** it calls `random_mask(x_val, k_min=k, k_max=k, seed=k)`, predicts, clips, overwrites known answers, and accumulates MAE on **missing** cells only. The mean over that **k-loop** is the scalar score used to compare `alpha` values (the notebook tries e.g. `0.1, 1, 10, 100, 200, 400, 800, 1600`).

**Test evaluation (`evaluate_model`).** An outer loop runs **`for k in k_values`** (in the shipped notebook: `(5, 10, 15, 20, 25, 30, 35, 40, 43, 49)`). An inner loop runs **`for s in range(N_EVAL_SEEDS)`** (e.g. 10 seeds) calling `random_mask(..., k_min=k, k_max=k, seed=1000 + s)` so each **K** is reported with low variance. The same k-structured evaluation applies to Global Mean, Trait Mean, and Ridge.

### The model: multi-output Ridge regression

A single `Ridge(alpha=...)` predicts all 50 answers from the 100 features. For each
output question `q`:

```
pred[q]  ≈  w_q · [X_masked | mask] + b_q
```

`alpha` is selected on the validation fold by minimizing the **mean** of validation MAEs over the **k-iterator** grid `K ∈ {5, 10, 20, 30, 40, 50}` (each evaluated with `k_min = k_max = K`). The notebook’s search grid extends beyond `{0.1, 1, 10, 100}` to larger `alpha` values when needed (see `models/main.ipynb`).

At inference time we overwrite predictions with the user's actual answers (no point
guessing what they already told us) and clip to `[1, 5]` so we don't return Likert
values outside the scale.

### Why precision grows with K

Two compounding effects:

1. **More signal in the input.** With K=5, only 5/100 features are nonzero — predictions
   regress hard toward the population mean. With K=30, 30 features carry information.
2. **Trait-score averaging gets less noisy.** Trait score = mean of 10 predicted item
   values. The more of those 10 are _real_ answers, the smaller the averaging noise.

That's why the precision-bar curve drops sharply over the first ~13 answers and
flattens by ~30. By 30 you've covered ~6 questions per trait on average, which is
basically as good as 50.

---

## Models Compared

| Model                | What it does                                                   | Why we tried it                                |
| -------------------- | -------------------------------------------------------------- | ---------------------------------------------- |
| **Global Mean**      | Fill missing with `3` (neutral)                                | Sanity floor                                   |
| **Trait Mean**       | Missing question Q → mean of _answered_ questions in Q's trait | Strong baseline; what `df_scores` already does |
| **Ridge Regression** | Linear multi-output regression on `[values, mask]`             | Can use cross-trait correlations               |

Ridge is the only one that can use cross-trait info (e.g. high `OPN3` → expect high
`OPN10`), so we expect it to win — especially at low K where every bit of cross-question
signal matters.

Evaluation reports three MAEs per model per **K** (see the `k_values` tuple in the notebook; it includes fine steps such as 43 and 49), averaged over **`N_EVAL_SEEDS`** random masks per K (e.g. 10 seeds):

- MAE on missing answers (1–5 scale)
- MAE on the 5 trait scores (0–1) — drives the precision bar
- MAE on the 7 meme questions (1–5) — the headline of every result page

The notebook produces a 3-panel plot showing how each metric evolves with K, with a
dashed red line at K=30 to make the soft-limit decision empirically obvious.

---

## Project Layout

```
B5Personality/
├── README.md                        # setup & IA summary
├── Detail.md                        # this file — extended methodology
├── requirements.txt
├── client/                          # Streamlit app (app.py)
├── models/
│   ├── main.ipynb
│   ├── usage.ipynb
│   ├── main.pdf                     # optional notebook export
│   ├── data/questionDetial.md
│   ├── artifacts/big5_ridge.joblib
│   └── data/big-five-personality-test/.../data-final.csv
└── .venv/
```

---

## Notebook Walkthrough

`models/main.ipynb` is structured top-to-bottom as:

1. **Project Vision & Direction** — the UX described above.
2. **Set up & Load dataset** — pulls the dataset from KaggleHub on first run, caches
   it locally.
3. **Data Exploring** — answer-distribution sanity check, correlation matrices that
   prove the within-trait redundancy that makes partial prediction possible.
4. **Clean data**
   - **Quality filter** — drops bad respondents (`IPC == 1`, `60 ≤ testelapse ≤ 1800`,
     `std(answers) > 0.3`, no missing answers). Single biggest precision win in the project.
   - Drops the metadata columns, keeps only the 50 question columns.
   - Computes `df_scores` (5 trait scores in 0–1).
5. **Model And Algorithm**
   - Approach + design rationale
   - Train/Val/Test split (70/15/15)
   - `random_mask()` — **rank-based** vectorised mask: random K per row for training; `k_min=k_max=k` inside **k-loops** for validation and test
   - Trait-mean and global-mean baselines
   - Ridge training on masked features, **`alpha`** chosen via validation **k-iterator** MAE
   - Evaluation framework (outer **k** loop × inner **seed** loop) + precision plots
   - Best-model persistence + `predict_personality()` helper for the front-end

Run the cells top to bottom; total wall-clock is ~30 seconds on a laptop CPU.

---

## Using the Trained Model

After running the notebook end-to-end, `models/artifacts/big5_ridge.joblib` contains
the model bundle. From any other Python file:

```python
import joblib, numpy as np

bundle = joblib.load("models/artifacts/big5_ridge.joblib")
ridge   = bundle["ridge"]
all_q   = bundle["all_q_cols"]      # the 50 question codes in canonical order
memes   = bundle["meme_questions"]  # the 7 meme codes
col_idx = {c: i for i, c in enumerate(all_q)}

def predict(answers: dict[str, float]) -> np.ndarray:
    """answers = {'EXT1': 5, 'OPN3': 4, ...}  →  predicted (50,) vector of 1..5 floats."""
    x = np.zeros((1, 50), dtype=np.float32)
    m = np.zeros((1, 50), dtype=np.float32)
    for q, v in answers.items():
        if q in col_idx:
            x[0, col_idx[q]] = float(v)
            m[0, col_idx[q]] = 1.0
    pred = ridge.predict(np.hstack([x, m]))[0]
    pred = np.where(m[0] == 1, x[0], pred)   # keep real answers untouched
    return np.clip(pred, 1, 5)
```

The notebook ships a richer `predict_personality(answers)` helper that returns a
ready-to-render payload:

```jsonc
{
  "n_answered": 13,
  "trait_scores": {
    "EXT": 0.62,
    "EST": 0.55,
    "AGR": 0.61,
    "CSN": 0.58,
    "OPN": 0.7,
  },
  "primary": {
    "answered": { "CSN6": 4.0 },
    "predicted": {
      /* 6 meme items */
    },
  },
  "secondary": {
    "answered": { "EXT1": 5.0, "AGR2": 1.0 },
    "predicted": {
      /* 41 items */
    },
  },
}
```

The front-end uses `trait_scores` to drive the precision bar, and renders `primary`
items as the headline cards and `secondary` items as supporting detail.

---

## Running Locally

Requires Python 3.11.

```powershell
# 1. Activate the venv (already created in .venv/)
.\.venv\Scripts\Activate.ps1

# 2. Install deps if needed
pip install pandas numpy scikit-learn matplotlib seaborn kagglehub joblib jupyter

# 3. Open the notebook
jupyter notebook models/main.ipynb
# or just open it in VS Code / Cursor and run all cells
```

The Kaggle dataset is downloaded automatically on first run via `kagglehub`, so no
manual setup is needed.

---

## Tech Stack

| Layer         | Tool                                                                   |
| ------------- | ---------------------------------------------------------------------- |
| Data          | Kaggle "Big Five Personality Test" (~1M responses, IPIP-FFM, Nov 2018) |
| Wrangling     | pandas, numpy                                                          |
| Visualization | matplotlib, seaborn                                                    |
| Modeling      | scikit-learn (`Ridge`, `train_test_split`)                             |
| Serialization | joblib                                                                 |
| Notebook      | Jupyter / VS Code notebook                                             |

---

## What's Next

A few directions to explore after the class deadline:

- **Iterative refinement at inference.** After the first prediction, feed the predicted
  values back as soft inputs and predict again (Gibbs-style). Two or three iterations
  give a measurable bump at very low K (5, 10) for ~10 lines of code.
- **Per-meme specialized models.** Train a `HistGradientBoostingRegressor` per meme
  question to capture nonlinearities Ridge can't, then stack it with the unified Ridge
  via a tiny meta-Ridge fit on the val set.
- **Reverse-coded items.** Some questions are negatively worded (`EXT2 "I don't talk
a lot"`). Currently the model learns to handle the sign implicitly; flipping them
  before training would make `df_scores` mean psychologically meaningful Big-5 scores.
- **Per-K precision UI.** The notebook eval already gives MAE per K — we can show it
  in the UI as a real "we are 78% confident" indicator instead of an abstract bar.
- **Question-selection policy.** Instead of randomizing question order, ask the _most
  informative remaining question_ at each step (active learning / submodular pick).
  The user converges on a good result with even fewer answers.

---

## Credits

- Dataset: [Bojan Tunguz, "Big Five Personality Test" on Kaggle](https://www.kaggle.com/datasets/tunguz/big-five-personality-test).
- Item wording: IPIP (International Personality Item Pool) Big-Five Factor Markers.
- Built as a final project for a data-modeling course.

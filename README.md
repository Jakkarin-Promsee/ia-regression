# IA Regression (Machine learning)

### Indicator-augmented regression for partial Big Five answers · meme-style quiz

**IA** = **I**ndicator-**A**ugmented — _not_ “artificial intelligence,” though the pun is half the fun. The serious name matches what we actually do: stack **observed values** with **binary indicators** so a linear model knows what was measured versus padding.

A data-modeling project that turns the IPIP Big Five personality test into a meme-style quiz where **users can quit at any time** and still get a meaningful result. The more questions they answer, the more precise the result becomes — but even with only a few answers the model produces a full readout for all 50 items.

- **Technique (full write-up):** [`Detail.md`](Detail.md) — random-K training, multi-mask augmentation, baselines, evaluation.
- **Live app:** [`client/`](client/) — Streamlit UI that loads the trained Ridge bundle and runs the quiz.
- **Modeling:** [`models/main.ipynb`](models/main.ipynb) — data prep, training, evaluation, and artifact export.
- **Engineering write-up (PDF):** [`models/main.pdf`](models/main.pdf) — notebook export for readers who do not use Jupyter (regenerate after changes; see [Exporting `main.pdf`](#exporting-mainpdf)).

Built on ~1M responses from Kaggle, with multi-output **Ridge** on indicator-augmented features and a **random-mask** training scheme so one model serves every completion rate _K_.

---

## Repository layout

```
B5Personality/
├── README.md                          # this file — setup, overview, IA technique summary
├── Detail.md                          # extended methodology (companion to README)
├── requirements.txt                   # pinned dependency ranges for pip
├── client/                            # Streamlit quiz app
│   ├── app.py                         # entry point (run with Streamlit)
│   └── app.html                       # earlier static prototype (reference)
├── models/
│   ├── main.ipynb                     # exploration, cleaning, Ridge training, evaluation
│   ├── main.pdf                       # exported notebook for engineers / readers (optional; regenerate from main.ipynb)
│   ├── usage.ipynb                    # short notebook: load artifact & sanity-check predictions
│   ├── artifacts/
│   │   └── big5_ridge.joblib          # trained model bundle (created by main.ipynb)
│   └── data/
│       ├── questionDetial.md          # the 50 IPIP item codes & texts
│       └── big-five-personality-test/ # created after first Kaggle download (kagglehub cache under models/)
│           └── IPIP-FFM-data-8Nov2018/
│               └── data-final.csv     # raw ~1M-row dataset
└── .venv/                             # recommended local Python 3.11 virtualenv (create yourself; not always committed)
```

Paths under `models/data/big-five-personality-test/` are populated when you run `main.ipynb` and `kagglehub` downloads [the Kaggle dataset](https://www.kaggle.com/datasets/tunguz/big-five-personality-test).

---

## Prerequisites

- **Python 3.11** (matches the project’s tooling and notebook).
- **Git** (to clone this repository).

Optional:

- **Jupyter** or **VS Code / Cursor** with the Jupyter extension — to run `main.ipynb` interactively.
- **Google Chrome** — only needed if you [export `main.pdf`](#exporting-mainpdf) using the HTML → headless-Chrome method below.

---

## Installation

### 1. Clone and enter the repository

```powershell
git clone <your-repo-url>
cd B5Personality
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

From the repository root, install everything needed for **notebooks** (training), **Streamlit**, and **notebook → PDF** export:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

See [`requirements.txt`](requirements.txt) for package names and compatible version ranges (notebooks, Streamlit, and `nbconvert` for exporting [`models/main.pdf`](#exporting-mainpdf)).

The Kaggle dataset is downloaded automatically on first notebook run via `kagglehub`; no manual CSV download is required.

---

## Running the Streamlit app

The UI expects the trained artifact at `models/artifacts/big5_ridge.joblib`. If that file is missing, run `models/main.ipynb` end-to-end first (see [Training pipeline](#training-pipeline)).

From the **repository root**:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run client/app.py
```

Or from the `client` folder:

```powershell
cd client
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Open it in your browser to use the quiz.

---

## Training pipeline

1. Activate the venv and ensure dependencies are installed (see [Installation](#installation)).
2. Open `models/main.ipynb` in Jupyter or VS Code / Cursor.
3. Run all cells **top to bottom**.
   - First run downloads and caches the dataset under `models/data/`.
   - Wall-clock is on the order of **~30 seconds** on a typical laptop CPU after data is cached.
4. Confirm `models/artifacts/big5_ridge.joblib` exists.

Quick checks without rerunning the full notebook: open `models/usage.ipynb` with working directory `models/` so `./artifacts/big5_ridge.joblib` resolves.

---

## Exporting `main.pdf`

`models/main.pdf` is a **static export** of `models/main.ipynb` for anyone who wants the full methodology, plots, and outputs **without** opening Jupyter. It is **not** auto-updated when you edit the notebook — regenerate it after substantive changes.

**Typical workflow (Windows, with Chrome installed):**

```powershell
cd models
# nbconvert is included in requirements.txt; skip this if you already ran pip install -r ../requirements.txt
python -m nbconvert --to html main.ipynb --output main.html
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD\main.pdf" "file:///$($PWD -replace '\\','/')/main.html"
Remove-Item main.html
```

Adjust the Chrome path if your browser is installed elsewhere. Other platforms can use the same idea: export HTML with `nbconvert`, then print to PDF from a browser or use `jupyter nbconvert --to webpdf` if Playwright is working on your machine.

---

## Indicator-augmented regression (IA)

The prediction problem is **matrix completion on Likert items**: from a **partial** answer vector, predict all 50 IPIP scores. We use a single multi-output regressor whose inputs are **not** raw incomplete vectors alone — they are **augmented with indicators**.

### Feature construction

For each user we form a **100-dimensional** input by concatenating:

| Block                      | Role                                                                                                           |
| -------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Masked values** (50)     | Real answers where the user responded; **0** elsewhere as _padding_ (not a claim that the true score is zero). |
| **Indicators / mask** (50) | **1** if that item was answered, **0** if it was missing for this session.                                     |

So the mask does the bookkeeping: it marks **which entries are genuine observations versus padding**. In missing-data language, this is the usual **observation pattern** (which variables were measured); in regression, it is an **indicator augmentation** so the model can assign **different weights** to “missing + padded zero” than to a hypothetical low score that was actually observed.

### Estimator and training

We fit **one multi-output Ridge** model on these 100 features to predict all 50 items. **Random-K masking** during training (together with **multi-mask augmentation**) exposes the same underlying respondents under many different subsets _K_, so the same linear map works whether the user stopped after 5 questions or 50. At inference, known answers overwrite predictions and outputs are clipped to the Likert range.

For filters, baselines, the `random_mask` code sketch, and evaluation curves, see **[`Detail.md`](Detail.md)** — it is the long-form appendix; this README stays the **quick path** for GitHub visitors.

---

## The Big Idea

Traditional personality quizzes force you through all 50 questions or give you nothing. This one does not:

- **Quit anytime.** Answer a few questions or all 50 — you always get a full result page.
- **Precision grows live.** A precision bar reflects how much the model can infer from what you answered.
- **Soft limit around 30.** Empirically, accuracy tends to plateau near 30 answers — the UI can nudge users there without blocking further questions.
- **Randomized order each session.** Repeat users get a fresh question order.

### Result page — always all 50 items

| Tier                               | What                    | When shown |
| ---------------------------------- | ----------------------- | ---------- |
| **Primary** (7 meme personalities) | Branded headline labels | Always     |
| **Secondary** (43 items)           | Supporting detail       | Always     |

Items the user answered show as known; the rest are filled by the model. Same layout for every K — no unlocks.

### The 7 meme personalities (Tier 1)

Each meme maps to one Big Five item; high agreement triggers the label (wording in the app may differ from this table).

| Question | Meme (example label)     |
| -------- | ------------------------ |
| `CSN6`   | Frequently Forgets Items |
| `AGR9`   | Empath                   |
| `AGR5`   | Cold Heart               |
| `EXT4`   | Wallflower               |
| `EST7`   | Mood Roller-Coaster      |
| `OPN3`   | Daydreamer               |
| `OPN10`  | Dare to Dream            |

---

## Why partial-answer prediction works

Big Five items are **redundant by design** — each trait has 10 items measuring related signal. A full 50-dimensional answer vector is largely determined by a few latent trait scores plus noise. Given any subset of K answers, those latents are partially pinned; **indicator-augmented regression** learns to map from “which items were observed + what they were” back to the full vector.

---

## Pipeline summary (after IA setup)

### Data quality (before modeling)

Respondents are filtered for plausible engagement (e.g. unique IP where applicable, reasonable completion time, non-flat answer patterns). This step is applied **before** dropping metadata columns and has a large impact on training quality. Details: [Data quality and full pipeline](Detail.md#how-the-model-works) in `Detail.md`.

### Training tricks (same 100-dim IA features)

1. **Random K per row:** each training row uses a random subset of answered questions; the rest are masked.
2. **Multi-mask augmentation:** each row is repeated with different random masks so the model sees many “partial views” of the same person.

### Estimator

**Multi-output Ridge regression** on **masked values stacked with indicators**, predicting all 50 items. Validation selects `alpha` across a grid; evaluation reports MAE at several K values. At inference, known answers overwrite predictions and outputs are clipped to `[1, 5]`.

---

## Models compared (notebook)

| Model           | Role                                                    |
| --------------- | ------------------------------------------------------- |
| **Global mean** | Sanity floor                                            |
| **Trait mean**  | Strong baseline using answered items within each trait  |
| **Ridge**       | Uses cross-item / cross-trait structure; best for low K |

---

## Using the trained artifact in Python

After training, load `models/artifacts/big5_ridge.joblib`:

```python
import joblib
import numpy as np

bundle = joblib.load("models/artifacts/big5_ridge.joblib")
ridge   = bundle["ridge"]
all_q   = bundle["all_q_cols"]
memes   = bundle["meme_questions"]
col_idx = {c: i for i, c in enumerate(all_q)}

def predict(answers: dict[str, float]) -> np.ndarray:
    """answers = {'EXT1': 5, 'OPN3': 4, ...}  ->  predicted (50,) on 1..5 scale."""
    x = np.zeros((1, 50), dtype=np.float32)
    m = np.zeros((1, 50), dtype=np.float32)
    for q, v in answers.items():
        if q in col_idx:
            x[0, col_idx[q]] = float(v)
            m[0, col_idx[q]] = 1.0
    pred = ridge.predict(np.hstack([x, m]))[0]
    pred = np.where(m[0] == 1, x[0], pred)
    return np.clip(pred, 1, 5)
```

The notebook defines a richer `predict_personality(answers)` helper returning trait scores, primary/secondary splits, and counts for the UI.

---

## Notebook walkthrough (`main.ipynb`)

1. **Project vision** — UX goals.
2. **Setup & load** — KaggleHub download and paths.
3. **Exploration** — distributions and correlation structure.
4. **Clean data** — quality filters, trait scores (`df_scores`).
5. **Modeling** — train/val/test split, `random_mask`, baselines, Ridge + augmentation, evaluation plots, persistence and prediction helpers.

---

## Tech stack

| Layer         | Tools                                                   |
| ------------- | ------------------------------------------------------- |
| Data          | Kaggle “Big Five Personality Test” (IPIP-FFM, Nov 2018) |
| Wrangling     | pandas, numpy                                           |
| Visualization | matplotlib, seaborn                                     |
| Modeling      | scikit-learn (`Ridge`, etc.)                            |
| Serialization | joblib                                                  |
| App           | Streamlit (`client/app.py`)                             |
| Notebooks     | Jupyter / VS Code / Cursor                              |

---

## What’s next

Ideas for future work: iterative refinement at inference, stronger nonlinear heads for meme items, explicit reverse-coding of negatively keyed items, richer precision UI from per-K MAEs, and smarter question ordering (e.g. active learning).

---

## Credits

- Dataset: [Bojan Tunguz, “Big Five Personality Test” on Kaggle](https://www.kaggle.com/datasets/tunguz/big-five-personality-test).
- Item wording: IPIP (International Personality Item Pool) Big-Five Factor Markers.
- Course context: data-modeling final project.

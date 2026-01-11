# ⚽ football-analytics-ml

**football-analytics-ml** is a spatial football feature engineering framework built on StatsBomb event data.

It produces reusable **player-level behavioural datasets** that describe how footballers occupy, move through and make directional decisions in space — not just what actions they perform.

Expensive data ingestion and feature computation are run **once**, written to disk as parquet artifacts, and reused for:

- modelling pipelines  
- research notebooks  
- blog and article analysis  

without ever reprocessing raw StatsBomb data.

---

## 🧠 Spatial Philosophy

StatsBomb uses a canonical **120 × 80 pitch coordinate system**:

| Axis | Meaning |
|-----|--------|
| `x` | Own goal → opponent goal |
| `y` | Left → right touchline |
| Direction | Both teams attack left → right |
| Half-time | Coordinates never flip |

This allows globally consistent spatial logic without mirroring.

The framework layers multiple spatial coordinate languages on top of this base pitch:

| Layer | Purpose |
|-----|--------|
| Raw metric | Motion, speed, angles |
| Thirds | Territorial responsibility |
| Channels | Width usage |
| 3×3 grid | Positional identity |
| Half-spaces | Interior creativity |
| Depth bands | Phase-of-play involvement |
| Directional vectors | Decision-making style |

---

## 🏗 Repository Structure

```text
football-analytics-ml/
├─ src/football_analytics/
│  ├─ data/          StatsBomb ingestion & minutes logic
│  ├─ geometry/      pitch geometry & coordinate systems
│  ├─ analyses/      action-family feature builders
│  ├─ features/      player-level aggregation
│  └─ dqa/           feature validation utilities
├─ scripts/          command-line pipelines
├─ notebooks/        research & blog analysis
├─ artifacts/        parquet datasets & outputs
└─ blog_notes/       Substack drafts
```

---

## 📦 Pipeline Output

The pipeline produces one canonical parquet file per competition & season, e.g.

```text
artifacts/features/player_level_wc2018.parquet
```

This file is the only dataset used by all notebooks and modelling pipelines.

---

## 🛠 Environment Setup

```bash
conda create -n football-analytics-ml python=3.11 -y
conda activate football-analytics-ml
pip install -U pip
pip install -e .
conda install -c conda-forge pyarrow
```

---

## 🏗 Build a Dataset

```bash
python scripts/build_player_features.py \
  --competition-id 43 \
  --season-id 3 \
  --out artifacts/features/player_level_wc2018.parquet
```

---

## 📊 Load the Dataset

```python
import pandas as pd
df = pd.read_parquet("artifacts/features/player_level_wc2018.parquet")
```

---

# 📊 Data Quality Assurance (DQA)

The pipeline includes a built-in **Data Quality Assurance (DQA) engine** that validates every feature table before modelling.

It protects the project from:

- broken feature engineering  
- discretisation failures  
- duplicated or redundant features  
- player identity leakage  
- dead / noisy columns  
- impossible football states  
- mathematically invalid ratios  

---

## Generate a DQA Report

```python
import football_analytics.dqa as dqa

report = dqa.feature_quality_report(
    passing_df,
    group_col="player",
    impossible_rules=[
        ("pass_completed", "pass_attempted"),
        ("forward_passes", "passes"),
    ],
)
```

---

## Automatic Feature Pruning

```python
drop = set()

drop |= set(report.health.query("`null_%` > 40 or constant").index)
drop |= set(report.bucket_collapsed[report.bucket_collapsed].index)
drop |= set(report.sparse.index)
drop |= set(report.low_entropy[report.low_entropy < 0.1].index)
drop |= set(report.leakage[report.leakage > 0.95].index)

sorted(drop)
```

---

## Philosophy

> If the DQA report is clean, your model performance is **real**.  
> If the DQA report is dirty, your model is **lying**.

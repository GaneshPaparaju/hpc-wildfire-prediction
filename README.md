# 🔥 HPC Wildfire Ignition Prediction

A High-Performance Computing (HPC) project demonstrating **parallel feature engineering, scalable model training, and performance analysis** on large-scale spatiotemporal wildfire data.

This project focuses on **when parallelization works, when it doesn’t, and why** — using real measurements instead of toy examples.

---

## 📌 Project Overview

Wildfire ignition prediction requires processing massive temporal weather datasets.  
This project applies **HPC techniques** to accelerate a full machine-learning pipeline built on:

- **9.5 million daily meteorological observations**
- **37,000+ spatial locations across the US**
- **12 years of data (2013–2025)**

The primary goal is **computational performance analysis**, not just predictive accuracy.

---

## 🎯 Objectives

- Parallelize **60-day rolling feature engineering**
- Benchmark **Joblib, Dask, and built-in multithreading**
- Measure **speedup, efficiency, and overhead**
- Validate **Amdahl’s Law** empirically
- Compare **XGBoost vs Random Forest** from an HPC perspective

---

## 🧠 Key HPC Concepts Demonstrated

- Task granularity and load balancing
- CPU core scaling and saturation
- Speedup vs efficiency trade-offs
- Parallel overhead analysis
- Framework–algorithm compatibility

---

## 🏗️ Pipeline Summary

1. Parallel CSV loading (Dask)
2. Data preprocessing and balancing
3. 60-day rolling feature engineering (Joblib)
4. Hyperparameter search (Joblib + scikit-learn)
5. Model training (XGBoost & Random Forest)
6. HPC benchmarking across multiple CPU counts

---

## ⚙️ Technologies Used

- **Python** (Pandas, NumPy, Scikit-learn)
- **Joblib** (process-based parallelism)
- **Dask** (distributed execution)
- **XGBoost**
- **Random Forest**
- **Slurm-based HPC cluster (Explorer OOD)**

---

## 📊 Parallel Feature Engineering Results

### Rolling Feature Engineering (Optimized Joblib)

| CPU Cores | Time (s) | Speedup | Efficiency |
|----------|----------|---------|------------|
| 1        | 417.3    | 1.00×   | 100% |
| 8        | 150.8    | 2.77×   | 34.6% |
| 16       | 93.1     | 4.48×   | 28.0% |
| 32       | 59.3     | 7.04×   | 22.0% |
| 50       | 42.9     | **9.72×** | **19.4%** |

**Key Insight:**  
Initial naïve parallelization achieved only **~1.1× speedup**.  
After workload-balanced chunking, speedup improved to **9.72×**.

---

## 📐 Amdahl’s Law Validation

- Parallel fraction: **89.9%**
- Sequential overhead: **10.1%**
- Theoretical maximum speedup: **9.9×**
- Observed speedup: **9.72×**

➡️ Achieved **~98% of the theoretical maximum**, clearly validating Amdahl’s Law.

---

## 🔍 Hyperparameter Search Scaling

| CPU Cores | Speedup | Efficiency |
|----------|---------|------------|
| 4        | 3.84×   | 96% |
| 8        | 5.76×   | 72% |
| 16       | 9.39×   | 59% |

**Lesson:**  
Coarse-grained tasks (full model training runs) parallelize extremely well.

---

## 🤖 Model Training: XGBoost vs Random Forest

### Best Observed Configurations

| Model | Parallel Method | CPUs | Wall Time (s) | Speedup | Efficiency |
|-----|----------------|------|---------------|--------|------------|
| XGBoost | Dask Distributed | 8 | **42.5** | 2.0× | 25% |
| Random Forest | Built-in Threads | 28 | 210.2 | **16.4×** | 59% |

### Interpretation

- **XGBoost**
  - Fastest absolute runtime
  - Limited scaling due to boosting’s sequential nature
- **Random Forest**
  - Near-linear scaling
  - Excellent example of embarrassingly parallel workloads

---

## 📈 Visual Results

Key performance plots are available in the `/figures` directory:
- Rolling feature speedup & efficiency
- Hyperparameter search scaling
- XGBoost vs Random Forest scaling comparison

---

## 🧪 Machine Learning Performance (Secondary Focus)

| Model | Accuracy | ROC-AUC |
|-----|---------|---------|
| XGBoost | ~0.62 | ~0.58 |
| Random Forest | ~0.66 | ~0.55 |

Predictive accuracy was intentionally treated as **secondary** to HPC analysis.

---

## 🔑 Key Takeaways

- More CPU cores do **not** guarantee better performance
- Task design matters more than framework choice
- Joblib excels for coarse-grained CPU tasks
- Dask introduces overhead on single-node workloads
- Built-in threading is often optimal for tree-based models

---

## 📂 Repository Structure

```text
.
├── notebooks/
│   ├── HPC_TEAM10.ipynb        # Main HPC experiments & benchmarks
│   └── HPC_TEAM10_EDA.ipynb    # Exploratory data analysis
├── figures/                   # Saved benchmark plots
├── report/
│   └── TEAM_10_Report_HPC.pdf  # Full technical report
└── README.md

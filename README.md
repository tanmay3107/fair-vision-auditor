# FairVision-Auditor ⚖️👁️

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/Library-PyTorch-EE4C2C)
![Focus](https://img.shields.io/badge/Focus-Algorithmic_Fairness-purple)
![Status](https://img.shields.io/badge/Status-Evaluation_Ready-green)

An automated evaluation and mitigation toolkit designed to audit PyTorch image classifiers for hidden biases and actively repair their data pipelines using dynamic augmentation strategies. 

Raw accuracy is a deceptive metric in high-stakes domains like medical diagnostics. A model can easily hit high overall accuracy while silently failing on edge cases, varying imaging conditions, or specific demographic slices. FairVision-Auditor exposes these failures and forces the model to learn robust, invariant features.

**Author:** Tanmay Janak

---

## 🚀 Key Engineering Features

* **Statistical Auditing Engine:** Calculates industry-standard algorithmic fairness metrics from raw inference data:
  * **Demographic Parity Difference (DPD)**
  * **Disparate Impact (DI)**
  * **Equal Opportunity Difference (EOD)**
* **Slicing Evaluator:** A memory-optimized inference loop (`torch.no_grad()`) that safely aggregates predictions and sensitive metadata across large datasets without GPU memory leaks.
* **Hybrid Augmentation Pipeline:** A custom PyTorch `Dataset` wrapper that intercepts data loading in real-time. It applies standard transforms to privileged data and aggressive, domain-specific augmentations (like `RandomErasing` and `ColorJitter`) to underperforming unprivileged slices.
* **Automated PDF Reporting:** Generates professional, compliance-ready visual audits comparing baseline models against mitigated models using `matplotlib`.

---

## 🎯 The Benchmarking Objective

The primary goal of this mitigation toolkit is strictly comparative. The objective is **not** to force the hybrid-augmented model to reach a predefined absolute accuracy threshold (e.g., 80%). 

Instead, the objective is to mathematically prove that the hybrid model's accuracy on the difficult, unprivileged data slices strictly **outperforms the baseline model's accuracy**, effectively closing the fairness gap and preventing disparate impact.

---

## 📂 Repository Structure

```text
fair-vision-auditor/
├── benchmark.py             # The core script to train and compare baseline vs. hybrid models
├── evaluator.py             # Memory-safe PyTorch inference loop for metadata slicing
├── hybrid_augmentation.py   # Dynamic PyTorch Dataset wrapper for targeted augmentation
├── metrics.py               # Pure mathematical implementations of DPD, DI, and EOD
└── report_generator.py      # Automated PDF generation for visual compliance audits
```

---

## 💻 Local Setup & Execution

### Prerequisites
* Python 3.10+
* PyTorch
* Matplotlib

### 1. Install Dependencies
```bash
pip install torch torchvision matplotlib
```

### 2. Run the Full Benchmark Audit
Execute the primary benchmarking script. This will simulate a dataset, train a baseline CNN, train a mitigated Hybrid CNN, and output the comparative delta.
```bash
python benchmark.py
```

### 3. Generate the Visual Report
To output the final PDF report highlighting the metric improvements:
```bash
python report_generator.py
```
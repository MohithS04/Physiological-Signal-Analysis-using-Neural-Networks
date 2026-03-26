# Physiological Signal Analysis Using Neural Networks

Neural network classifiers for cardiac arrhythmia and sleep apnea detection, evaluated across three public ECG databases under clean and noisy conditions.

Built at Kent State University as part of a comparative study on ECG signal classification with single-layer and multi-layer neural networks.

---

## Overview

The project runs the same general pipeline — filter, segment, extract features, classify — across three datasets with meaningfully different tasks:

- **MIT-BIH Arrhythmia**: Detect irregular heartbeats from clinical ECG recordings
- **Apnea-ECG**: Identify sleep apnea episodes using heart rate variability signatures
- **MIT-BIH Noise Stress Test**: Test how well the models hold up when the signal is intentionally degraded

Each dataset has its own quirks (class imbalance, noise levels, subject variability), so the results tell different stories.

---

## Datasets

All three datasets are publicly available on [PhysioNet](https://physionet.org/).

| Dataset | Source | Subjects Used | Sampling Rate | Task |
|---|---|---|---|---|
| MIT-BIH Arrhythmia | [physionet.org/content/mitdb](https://physionet.org/content/mitdb/1.0.0/) | 114, 124, 202, 221, 102 | 360 Hz | Arrhythmia detection |
| Apnea-ECG | [physionet.org/content/apnea-ecg](https://physionet.org/content/apnea-ecg/1.0.0/) | a04, a06, b05, a08, c10 | 100 Hz | Sleep apnea detection |
| MIT-BIH Noise Stress Test | [physionet.org/content/nstdb](https://physionet.org/content/nstdb/1.0.0/) | 118e_6, 118e12, 118e18, 119e_6, 119e24 | 360 Hz | Noise-robust classification |

---

## Pipeline

```
Raw ECG signal
    ↓
Butterworth band-pass filter (noise removal)
    ↓
Windowed segmentation (with overlap)
    ↓
Feature extraction (time-domain, frequency-domain, PQRST morphology)
    ↓
SMOTE (class balancing)
    ↓
Normalization
    ↓
Neural network classification (SLNN / MLP / ensemble)
    ↓
Evaluation (accuracy, precision, recall, F1, AUC)
```

---

## Preprocessing

### Dataset 1 — MIT-BIH Arrhythmia
- Filter: Butterworth band-pass, 0.5–50 Hz
- Segmentation: 3-second windows, 0.5-second overlap
- Labels (AAMI standard): Normal (N, L, R) → 0 | Abnormal (A, V, F, E) → 1
- Each window labeled 1 if any abnormal beat appears in it

### Dataset 2 — Apnea-ECG
- Filter: Butterworth band-pass, 0.5–40 Hz
- Segmentation: 10-second windows, 2-second overlap (to capture full respiratory cycles)
- Labels: per-minute annotations, Apnea (A) or Normal (N)

### Dataset 3 — MIT-BIH Noise Stress Test
- Filter: Butterworth band-pass, 0.5–50 Hz
- Segmentation: 3-second windows, 0.5-second overlap
- Noise types: baseline wander, electrode motion, muscle artifacts

---

## Features

### Time-domain
Min, max, amplitude, mean, standard deviation, median, skewness, kurtosis

### Morphological
Peak count, trough count, mean RR interval (time between successive R-peaks)

### Frequency-domain
FFT-derived mean frequency, total spectral power, spectral entropy

### PQRST intervals
P, Q, R, S, T amplitudes; PR interval; QRS duration; QT interval

Features per window: **15** (Arrhythmia/Apnea), **12** (Noise Stress Test)

---

## Models

**Single-Layer Neural Network (SLNN)** — baseline; tests whether the features are linearly separable

**Multi-Layer Perceptron (MLP)** — two hidden layers with ReLU activation; picks up nonlinear relationships between amplitude, interval, and frequency features

**Ensemble (Apnea-ECG only)** — soft-voting combination of Random Forest, Gradient Boosting, and MLP. Random Forest handles noisy high-dimensional inputs; Gradient Boosting corrects for prior prediction errors sequentially. Together they handle class imbalance better than either network alone.

All models use SMOTE for class balancing and standard normalization before training. Evaluation uses 80/20 train/test split.

---

## Results

### Dataset 1 — MIT-BIH Arrhythmia

| Model | Accuracy | Macro F1 | AUC |
|---|---|---|---|
| Single-Layer NN | 97.43% | 0.75 | 0.9925 |
| Multi-Layer NN | 99.32% | 0.83 | 0.9983 |

Per-subject (MLP):

| Subject | Accuracy | Recall | AUC | Notes |
|---|---|---|---|---|
| 114 | 99.86% | 1.00 | 1.00 | |
| 124 | 99.86% | 1.00 | 0.9997 | |
| 202 | 100% | 1.00 | 1.00 | |
| 221 | 99.86% | 1.00 | 1.00 | |
| 102 | 97.44% | 0.67 | 1.00 | Lower F1 due to class imbalance (99 normal vs. 4 abnormal) |

Recall stayed at 1.0 across all subjects — no missed arrhythmic beats. Subject 102's lower F1 is a class imbalance artifact, not a detection failure.

---

### Dataset 2 — Apnea-ECG

| Model | Accuracy | AUC | Notes |
|---|---|---|---|
| Single-Layer NN | 92.37% | 0.75 | F1 = 0.0 — only predicting "normal" |
| Multi-Layer NN | 92.34% | 0.77 | Same issue |
| **Ensemble (RF + GB + MLP)** | **85%** | **0.92** | Actually detects apnea |

The 92% accuracy figure for the standalone networks is misleading — they learned to predict "normal" for everything. The ensemble trades raw accuracy for actual apnea detection, which is the whole point.

Per-subject (ensemble):

| Subject | Accuracy | F1 | AUC | Notes |
|---|---|---|---|---|
| a04 | 55.1% | 0.26 | 0.63 | Model biased toward apnea class |
| a06 | 71.9% | 0.08 | 0.65 | Predictions skewed toward normal |
| b05 | 82.5% | 0.27 | 0.63 | Best overall; most balanced |
| a08 | 28.5% | 0.10 | 0.78 | Heavy waveform noise |
| c10 | 2.7% | 0.00 | 0.84 | Extreme imbalance; nearly all predicted as apnea |

Subject-level variability is large here. c10's AUC of 0.84 alongside 2.7% accuracy is a good example of why accuracy alone tells you nothing on heavily imbalanced data.

---

### Dataset 3 — MIT-BIH Noise Stress Test

| Model | Accuracy | Macro F1 | AUC |
|---|---|---|---|
| Single-Layer NN | 69.1% | 0.678 | 0.745 |
| Multi-Layer NN | 67.0% | 0.649 | 0.725 |

Both models stayed above 0.70 AUC under deliberately degraded signals. The single-layer model generalized slightly better overall; the MLP was more adaptable to subjects with heavier noise. Subjects with higher noise levels tended to show higher recall but slightly lower precision.

---

## Honest Takeaways

**The arrhythmia results are strong.** Near-perfect detection across five subjects, with the PQRST morphology and RR interval features carrying real diagnostic weight. The main caveat is that the selected records have fairly clean signals.

**The apnea results need careful reading.** The standalone networks hit 92% by ignoring the minority class, which is a common failure mode on imbalanced ECG data. The ensemble's 85% with AUC 0.92 is the more meaningful number. Per-subject variability suggests subject-specific calibration would likely help.

**The noise stress results land in a reasonable range.** ~69% accuracy and ~0.74 AUC on intentionally corrupted signals suggests the preprocessing holds up, though abnormal beat detection under heavy noise still needs work.

---

## Per-Subject Summary (All Datasets)

| Dataset | Subject | Accuracy | F1 | AUC | Notes |
|---|---|---|---|---|---|
| MIT-BIH Arrhythmia | 114 | 99.8% | 1.00 | ~1.00 | |
| | 124 | 99.8% | 0.98 | ~1.00 | |
| | 202 | 100% | 0.99 | 1.00 | |
| | 221 | 99.8% | 0.97 | ~1.00 | |
| | 102 | 97.0% | 0.66 | ~1.00 | Class imbalance |
| Apnea-ECG | a04 | 55.1% | 0.26 | 0.63 | |
| | a06 | 71.9% | 0.08 | 0.65 | |
| | b05 | 82.5% | 0.27 | 0.63 | Best subject |
| | a08 | 28.5% | 0.10 | 0.78 | Signal noise |
| | c10 | 2.7% | 0.00 | 0.84 | Extreme imbalance |
| Noise Stress Test | 118e_6 | 69.0% | 0.70 | 0.74 | |
| | 118e12 | 67.0% | 0.69 | 0.73 | |
| | 118e18 | 68.0% | 0.72 | 0.72 | |
| | 119e_6 | 69.0% | 0.70 | 0.74 | |
| | 119e24 | 70.0% | 0.71 | 0.75 | Best in noisy set |

---

## Authors

**Mohith Reddy Seelam** — Department of Computer Science & Mathematical Sciences, Kent State University
mseelam1@kent.edu

**JungYoon Kim** — Department of Computer Science, Kent State University
jkim78@kent.edu

---

## References

1. [Psychiatric disorders from EEG signals through deep learning](https://www.sciencedirect.com/science/article/pii/S266724212400082)
2. [Physiological signal processing via machine learning for stress detection](https://openscholar.dut.ac.za/server/api/core/bitstreams/a623ab64-deeb-4b87-ba69-90a56ac0d39e/content)
3. [Hybrid deep learning for emotion recognition using physiological signals](https://www.researchgate.net/publication/384165067)
4. [ETD: Extended time delay algorithm for ventricular fibrillation detection](https://www.researchgate.net/publication/270658168)
5. [Machine learning approaches to recognize human emotions](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2023.1333794/full)
6. [ECG signal feature extraction: trends in methods and applications](https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-023-01075-1)

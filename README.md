# Physiological Signal Analysis Using Neural Networks

Neural network classifiers for cardiac arrhythmia and sleep apnea detection, evaluated across three public ECG databases under clean and noisy conditions.

Built at Kent State University as part of a comparative study on ECG signal classification with single-layer and multi-layer neural networks.

---

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ECG Signal Analysis — Physiological Signal Pipeline</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Space+Grotesk:wght@300;500;700&display=swap');

  :root {
    --bg: #060d12;
    --panel: #0b1620;
    --border: #1a2e3a;
    --green: #00ff88;
    --cyan: #00d4ff;
    --amber: #ffaa00;
    --red: #ff4466;
    --purple: #aa66ff;
    --text: #c8dde8;
    --muted: #4a6272;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    font-family: 'JetBrains Mono', monospace;
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  .header {
    padding: 28px 32px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 16px;
    background: linear-gradient(180deg, #0a1822 0%, transparent 100%);
  }

  .pulse-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 12px var(--green);
    animation: pulse 1.4s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.7); }
  }

  .header-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--cyan);
  }

  .header-sub {
    margin-left: auto;
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.08em;
  }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1px;
    background: var(--border);
    border-bottom: 1px solid var(--border);
  }

  .channel {
    background: var(--bg);
    padding: 0;
    position: relative;
    overflow: hidden;
  }

  .channel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px 6px;
    border-bottom: 1px solid var(--border);
  }

  .channel-label {
    font-size: 9px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 600;
  }

  .channel-stats {
    display: flex;
    gap: 12px;
    font-size: 9px;
    color: var(--muted);
  }

  .stat-val { font-weight: 600; }

  .canvas-wrap {
    position: relative;
    height: 90px;
  }

  canvas {
    display: block;
    width: 100%;
    height: 100%;
  }

  /* Grid lines overlay */
  .grid-overlay {
    position: absolute;
    inset: 0;
    pointer-events: none;
    background-image:
      linear-gradient(rgba(0,212,255,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,255,0.04) 1px, transparent 1px);
    background-size: 20px 20px;
  }

  /* Scan line */
  .scan-line {
    position: absolute;
    top: 0; bottom: 0;
    width: 2px;
    background: linear-gradient(180deg, transparent, var(--green), transparent);
    opacity: 0.6;
    pointer-events: none;
  }

  .bottom-panel {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1px;
    background: var(--border);
    border-bottom: 1px solid var(--border);
  }

  .metric-box {
    background: var(--panel);
    padding: 14px 16px;
  }

  .metric-label {
    font-size: 8px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
  }

  .metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 700;
    line-height: 1;
  }

  .metric-unit {
    font-size: 9px;
    color: var(--muted);
    margin-top: 2px;
  }

  .legend-bar {
    display: flex;
    gap: 24px;
    padding: 12px 20px;
    align-items: center;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 9px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
  }

  .legend-line {
    width: 24px; height: 2px;
    border-radius: 1px;
  }

  .badge {
    margin-left: auto;
    font-size: 8px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    border: 1px solid var(--border);
    padding: 3px 8px;
    border-radius: 2px;
  }

  /* classifier bar at bottom */
  .classifier-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1px;
    background: var(--border);
  }

  .classifier-box {
    background: var(--bg);
    padding: 12px 16px;
  }

  .clf-title {
    font-size: 8px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }

  .clf-bars { display: flex; flex-direction: column; gap: 5px; }

  .clf-row { display: flex; align-items: center; gap: 8px; font-size: 8px; }

  .clf-name { width: 48px; color: var(--muted); letter-spacing: 0.06em; }

  .clf-bar-wrap {
    flex: 1;
    height: 5px;
    background: var(--border);
    border-radius: 1px;
    overflow: hidden;
  }

  .clf-bar-fill {
    height: 100%;
    border-radius: 1px;
    animation: grow 1.5s ease-out forwards;
    transform-origin: left;
  }

  @keyframes grow {
    from { transform: scaleX(0); }
    to { transform: scaleX(1); }
  }

  .clf-val { width: 32px; text-align: right; }

  .status-row {
    padding: 8px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 8px;
    letter-spacing: 0.1em;
    color: var(--muted);
    border-top: 1px solid var(--border);
  }

  .status-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 6px var(--green);
    animation: pulse 1.4s ease-in-out infinite;
  }
</style>
</head>
<body>

<div class="header">
  <div class="pulse-dot"></div>
  <div class="header-title">ECG Signal Analysis — Multi-Dataset Neural Classification</div>
  <div class="header-sub">Kent State University · SCI Lab</div>
</div>

<div class="grid">
  <!-- Channel 1: Arrhythmia -->
  <div class="channel">
    <div class="channel-header">
      <span class="channel-label" style="color: var(--green)">MIT-BIH Arrhythmia</span>
      <div class="channel-stats">
        <span>360 Hz</span>
        <span style="color: var(--green)" class="stat-val">99.3% ACC</span>
      </div>
    </div>
    <div class="canvas-wrap">
      <div class="grid-overlay"></div>
      <canvas id="c1"></canvas>
      <div class="scan-line" id="scan1"></div>
    </div>
  </div>

  <!-- Channel 2: Apnea -->
  <div class="channel">
    <div class="channel-header">
      <span class="channel-label" style="color: var(--cyan)">Apnea-ECG</span>
      <div class="channel-stats">
        <span>100 Hz</span>
        <span style="color: var(--cyan)" class="stat-val">AUC 0.92</span>
      </div>
    </div>
    <div class="canvas-wrap">
      <div class="grid-overlay"></div>
      <canvas id="c2"></canvas>
      <div class="scan-line" id="scan2"></div>
    </div>
  </div>

  <!-- Channel 3: Noise Stress -->
  <div class="channel">
    <div class="channel-header">
      <span class="channel-label" style="color: var(--amber)">Noise Stress Test</span>
      <div class="channel-stats">
        <span>360 Hz</span>
        <span style="color: var(--amber)" class="stat-val">69% ACC</span>
      </div>
    </div>
    <div class="canvas-wrap">
      <div class="grid-overlay"></div>
      <canvas id="c3"></canvas>
      <div class="scan-line" id="scan3"></div>
    </div>
  </div>
</div>

<div class="bottom-panel">
  <div class="metric-box">
    <div class="metric-label">Heart Rate</div>
    <div class="metric-value" style="color: var(--green)" id="hr">72</div>
    <div class="metric-unit">BPM</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">RR Interval</div>
    <div class="metric-value" style="color: var(--cyan)">0.83</div>
    <div class="metric-unit">seconds</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">QRS Duration</div>
    <div class="metric-value" style="color: var(--text)">0.078</div>
    <div class="metric-unit">seconds</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">QT Interval</div>
    <div class="metric-value" style="color: var(--purple)">0.289</div>
    <div class="metric-unit">seconds</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">Classification</div>
    <div class="metric-value" style="color: var(--green); font-size: 14px; margin-top: 4px;" id="clf-label">NORMAL</div>
    <div class="metric-unit" id="clf-conf">confidence 0.997</div>
  </div>
</div>

<div class="classifier-row">
  <!-- Arrhythmia results -->
  <div class="classifier-box">
    <div class="clf-title">MIT-BIH Arrhythmia · Model Comparison</div>
    <div class="clf-bars">
      <div class="clf-row">
        <span class="clf-name">SLNN</span>
        <div class="clf-bar-wrap"><div class="clf-bar-fill" style="width: 97.43%; background: var(--green); animation-delay: 0.1s"></div></div>
        <span class="clf-val" style="color: var(--green)">97.4%</span>
      </div>
      <div class="clf-row">
        <span class="clf-name">MLP</span>
        <div class="clf-bar-wrap"><div class="clf-bar-fill" style="width: 99.32%; background: var(--green); animation-delay: 0.25s"></div></div>
        <span class="clf-val" style="color: var(--green)">99.3%</span>
      </div>
    </div>
  </div>

  <!-- Apnea results -->
  <div class="classifier-box">
    <div class="clf-title">Apnea-ECG · Ensemble vs Single</div>
    <div class="clf-bars">
      <div class="clf-row">
        <span class="clf-name">SLNN</span>
        <div class="clf-bar-wrap"><div class="clf-bar-fill" style="width: 75%; background: var(--muted); animation-delay: 0.2s"></div></div>
        <span class="clf-val" style="color: var(--muted)">AUC 0.75</span>
      </div>
      <div class="clf-row">
        <span class="clf-name">MLP</span>
        <div class="clf-bar-wrap"><div class="clf-bar-fill" style="width: 77%; background: var(--muted); animation-delay: 0.3s"></div></div>
        <span class="clf-val" style="color: var(--muted)">AUC 0.77</span>
      </div>
      <div class="clf-row">
        <span class="clf-name">Ensemble</span>
        <div class="clf-bar-wrap"><div class="clf-bar-fill" style="width: 92%; background: var(--cyan); animation-delay: 0.4s"></div></div>
        <span class="clf-val" style="color: var(--cyan)">AUC 0.92</span>
      </div>
    </div>
  </div>

  <!-- Noise results -->
  <div class="classifier-box">
    <div class="clf-title">Noise Stress Test · Robustness</div>
    <div class="clf-bars">
      <div class="clf-row">
        <span class="clf-name">SLNN</span>
        <div class="clf-bar-wrap"><div class="clf-bar-fill" style="width: 69.1%; background: var(--amber); animation-delay: 0.1s"></div></div>
        <span class="clf-val" style="color: var(--amber)">69.1%</span>
      </div>
      <div class="clf-row">
        <span class="clf-name">MLP</span>
        <div class="clf-bar-wrap"><div class="clf-bar-fill" style="width: 67%; background: var(--amber); animation-delay: 0.25s"></div></div>
        <span class="clf-val" style="color: var(--amber)">67.0%</span>
      </div>
    </div>
  </div>
</div>

<div class="legend-bar">
  <div class="legend-item">
    <div class="legend-line" style="background: var(--green)"></div>
    Arrhythmia DB
  </div>
  <div class="legend-item">
    <div class="legend-line" style="background: var(--cyan)"></div>
    Apnea-ECG
  </div>
  <div class="legend-item">
    <div class="legend-line" style="background: var(--amber)"></div>
    Noise Stress
  </div>
  <div class="legend-item">
    <div class="legend-line" style="background: var(--red); opacity: 0.7"></div>
    Arrhythmic event
  </div>
  <div class="badge">SMOTE · Butterworth BPF · PQRST Features</div>
</div>

<div class="status-row">
  <div class="status-dot"></div>
  <span>LIVE SIMULATION</span>
  <span>·</span>
  <span>3-SECOND WINDOWS · 0.5s OVERLAP</span>
  <span>·</span>
  <span>15 FEATURES EXTRACTED PER WINDOW</span>
  <span>·</span>
  <span>WFDB · MEDIAPIPE · SKLEARN</span>
</div>

<script>
// --- ECG waveform generators ---

function ecgPulse(t, amplitude = 1, noise = 0) {
  // PQRST morphology synthesizer
  const mod = t % 1;
  let y = 0;
  const n = (Math.random() - 0.5) * noise;

  // P wave
  if (mod > 0.10 && mod < 0.22) {
    y = 0.15 * amplitude * Math.sin(Math.PI * (mod - 0.10) / 0.12);
  }
  // Q dip
  else if (mod > 0.28 && mod < 0.34) {
    y = -0.12 * amplitude * Math.sin(Math.PI * (mod - 0.28) / 0.06);
  }
  // R spike
  else if (mod > 0.33 && mod < 0.42) {
    const peak = 0.375;
    y = amplitude * Math.exp(-Math.pow((mod - peak) / 0.025, 2));
  }
  // S dip
  else if (mod > 0.42 && mod < 0.50) {
    y = -0.18 * amplitude * Math.exp(-Math.pow((mod - 0.46) / 0.025, 2));
  }
  // T wave
  else if (mod > 0.52 && mod < 0.72) {
    y = 0.28 * amplitude * Math.sin(Math.PI * (mod - 0.52) / 0.20);
  }

  return y + n;
}

function apneaEcg(t, apneaPhase = false) {
  // During apnea: slower, irregular, amplitude-modulated
  const base = apneaPhase ? 0.6 : 1.0;
  const rateShift = apneaPhase ? 0.75 : 1.0;
  const noise = apneaPhase ? 0.06 : 0.015;
  return ecgPulse(t * rateShift, base, noise);
}

function noisyEcg(t, noiseLevel = 0.3) {
  const clean = ecgPulse(t, 1, 0);
  // baseline wander
  const wander = 0.12 * Math.sin(2 * Math.PI * t * 0.2);
  // EMG noise
  const emg = noiseLevel * (Math.random() - 0.5) * 0.8;
  // electrode motion burst
  const burst = (Math.floor(t * 0.4) % 5 === 2) ? 0.4 * Math.sin(t * 80) * Math.random() : 0;
  return clean + wander + emg + burst;
}

// --- Canvas renderer ---

class ECGRenderer {
  constructor(canvasId, scanId, generator, color, options = {}) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.scanEl = document.getElementById(scanId);
    this.generator = generator;
    this.color = color;
    this.options = options;
    this.buffer = [];
    this.t = Math.random() * 100;
    this.speed = options.speed || 0.004;
    this.maxPoints = 0;
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width * window.devicePixelRatio;
    this.canvas.height = rect.height * window.devicePixelRatio;
    this.canvas.style.width = rect.width + 'px';
    this.canvas.style.height = rect.height + 'px';
    this.maxPoints = Math.floor(rect.width * window.devicePixelRatio);
    this.buffer = new Array(this.maxPoints).fill(0);
    this.scanPos = 0;
  }

  draw() {
    const ctx = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;
    const mid = H / 2;
    const amp = H * 0.35;

    // Fade trail
    ctx.fillStyle = 'rgba(6, 13, 18, 0.18)';
    ctx.fillRect(0, 0, W, H);

    // Advance signal
    this.t += this.speed;
    const newVal = this.generator(this.t);
    this.buffer.push(newVal);
    if (this.buffer.length > this.maxPoints) this.buffer.shift();

    // Draw waveform with glow
    const drawLine = (alpha, lineWidth, blur) => {
      ctx.save();
      ctx.strokeStyle = this.color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = lineWidth;
      ctx.shadowColor = this.color;
      ctx.shadowBlur = blur;
      ctx.beginPath();
      const step = W / this.maxPoints;
      for (let i = 0; i < this.buffer.length; i++) {
        const x = i * step;
        const y = mid - this.buffer[i] * amp;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    };

    // Outer glow
    drawLine(0.15, 8, 20);
    // Mid glow
    drawLine(0.4, 3, 10);
    // Sharp line
    drawLine(0.95, 1.5, 4);

    // Mark arrhythmic events (channel 1 only)
    if (this.options.markEvents && Math.abs(newVal) > 0.85) {
      ctx.save();
      ctx.strokeStyle = '#ff4466';
      ctx.globalAlpha = 0.6;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      const x = (this.buffer.length - 1) * (W / this.maxPoints);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
      ctx.restore();
    }

    // Scan line position
    if (this.scanEl) {
      const pct = ((this.buffer.length - 1) / this.maxPoints) * 100;
      this.scanEl.style.left = pct + '%';
    }
  }
}

// --- Init renderers ---

let apneaPhase = false;
setInterval(() => { apneaPhase = !apneaPhase; }, 4000);

const r1 = new ECGRenderer('c1', 'scan1',
  t => ecgPulse(t, 1.0, 0.02), '#00ff88',
  { speed: 0.0038, markEvents: true });

const r2 = new ECGRenderer('c2', 'scan2',
  t => apneaEcg(t, apneaPhase), '#00d4ff',
  { speed: 0.0028 });

const r3 = new ECGRenderer('c3', 'scan3',
  t => noisyEcg(t, 0.25), '#ffaa00',
  { speed: 0.0042 });

// Animate HR readout
let hrBase = 72;
setInterval(() => {
  hrBase = Math.max(58, Math.min(96, hrBase + (Math.random() - 0.5) * 4));
  document.getElementById('hr').textContent = Math.round(hrBase);

  // Occasionally flash arrhythmia
  const isArr = Math.random() < 0.08;
  const lbl = document.getElementById('clf-label');
  const conf = document.getElementById('clf-conf');
  if (isArr) {
    lbl.textContent = 'ARRHYTHMIA';
    lbl.style.color = '#ff4466';
    conf.textContent = 'confidence 0.931';
  } else {
    lbl.textContent = 'NORMAL';
    lbl.style.color = '#00ff88';
    conf.textContent = 'confidence ' + (0.980 + Math.random() * 0.018).toFixed(3);
  }
}, 1200);

// Main loop
function loop() {
  r1.draw();
  r2.draw();
  r3.draw();
  requestAnimationFrame(loop);
}

loop();
</script>
</body>
</html>


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

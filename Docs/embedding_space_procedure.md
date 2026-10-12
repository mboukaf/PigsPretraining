# Building the embedding space — encoder-only procedure (6 cardiac signals)

**Project:** representation learning for blood pressure / blood flow signals
**Scope of this document:** everything needed to go from the raw porcine recordings to a trained encoder that maps a window of heartbeats to an embedding vector `e`. Flow prediction, identifiability and uncertainty quantification are deliberately **out of scope** and will be built on top of this encoder later.
**Data:** porcine dataset, DOI 10.17605/OSF.IO/9YWJU (Liège / Canterbury), described in Cushway et al., *PLOS ONE* 19(10): e0306925, 2024.

---

## Table of contents

0. Notation
1. Goal and design principles
2. Signals used
3. Step 1 — Data inventory and sanity checks
4. Step 2 — Artifact detection and cleaning
5. Step 3 — Filtering
6. Step 4 — Beat grid (segmentation)
7. Step 5 — Beat-level quality control
8. Step 6 — Resampling beats to fixed length
9. Step 7 — Normalization
10. Step 8 — Windowing
11. Step 9 — Data splits
12. Step 10 — The encoder architecture (shape by shape)
13. Step 11 — How the encoder is trained (self-supervision)
14. Step 12 — Optimization and monitoring
15. Step 13 — Evaluating the embedding space
16. Step 14 — Using the trained encoder (inference)
17. Summary tables (shapes, parameter counts, hyperparameters)
18. Open decisions and risks

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| `fs` | Sampling rate of the raw recordings = 250 Hz |
| `M` | Number of signals (modalities) = 6 |
| `m` | Index of a modality, `m ∈ {1,…,6}` |
| `N` | Number of raw samples in a recording |
| `K` | Number of heartbeats detected in a recording |
| `i` | Beat index |
| `L` | Number of points each beat is resampled to = 64 |
| `W` | Number of beats per window = 32 |
| `n` | Window index |
| `B` | Batch size (number of windows processed together) = 128 |
| `D` | Width of the transformer tokens = 128 |
| `d_e` | Dimension of the embedding = 32 |
| `x` | Raw signals of one recording, shape `(M, N)` |
| `X` | One window of resampled beats, shape `(W, M, L)` |
| `r` | Beat-duration feature of each beat in a window, shape `(W,)` |
| `a` | Availability mask (is beat `i` of modality `m` usable?), shape `(W, M)` |
| `e` | The embedding of a window, shape `(d_e,)` = `(32,)` |
| view | A randomly modified copy of a window used during training (Section 13) |

A **token** is one element of the sequence the transformer reads. Here, **one token = one heartbeat** (all 6 signals of that beat combined).

---

## 1. Goal and design principles

**Goal.** Learn a function (the *encoder*)

```
f : window of heartbeats (any subset of the 6 signals)  →  e ∈ R^32
```

such that windows recorded in similar physiological states map to nearby points, and windows in different states map to distant points.

**What "meaningful" means — four checkable properties** (tested in Section 15):

1. **Contains physiology.** Known quantities (heart rate, mean pressure, pulse pressure, stroke volume, LV contractility indices, Windkessel parameters fitted from pressure + flow, …) can be read out of `e` with a simple linear model.
2. **Tracks state changes.** Within one animal, `e` moves at interventions (PEEP steps, endotoxin, fluid boluses) and stays stable when nothing happens.
3. **Is consistent across sensor subsets.** The same window seen through different subsets of signals gives nearby embeddings.
4. **Does not memorize the animal.** Physiology must be much easier to decode from `e` than the identity of the pig.

**Design principles.**

- **The heartbeat is the unit.** All 6 signals are cardiac-synchronous, so they are cut on a common beat grid. Comparing beat to beat is far easier for a network than comparing arbitrary chunks of signal.
- **Amplitude is information.** Normalization is global per signal, never per beat or per window.
- **Missing signals are normal.** The encoder must accept any subset of the 6 signals, both because recordings have gaps and because, at the bedside, most of these signals will not exist.
- **No labels are used for training.** The encoder is trained by self-supervision only. Labels (experimental phase, physiological quantities) are used exclusively to *evaluate* the embedding.
- **Animals are never mixed between training and testing.** All evaluation is leave-one-pig-out.

---

## 2. Signals used

| # | Symbol | Signal | Sensor (per the paper) | Expected units | Typical pig range | Notes |
|---|---|---|---|---|---|---|
| 1 | `P_ao` | Proximal aortic pressure | Pressure catheter | mmHg | 40–130 | Reference signal for the beat grid |
| 2 | `Q_ao` | Aortic flow | To be verified (likely perivascular flow probe) | mL/s (verify; could be L/min) | −50 to 400 | Zero offset may drift; check sign convention |
| 3 | `P_fem` | Femoral arterial pressure | Pressure catheter | mmHg | 40–140 | Delayed vs `P_ao` by the pulse transit time (≈ 30–70 ms) |
| 4 | `P_lv` | Left-ventricular pressure | Admittance PV catheter | mmHg | −5 to 130 | Rises *before* `P_ao` (isovolumic contraction) |
| 5 | `V_lv` | Left-ventricular volume | Admittance PV catheter | mL | 10–80 (uncalibrated offset possible) | Reported as noisy in the paper; absolute level may be unreliable |
| 6 | `P_vc` | Vena cava pressure | Pressure catheter | mmHg | 0–20 | Small pulsatility, strong respiratory modulation |

The two airway signals (`P_aw`, `Q_aw`) are **not** fed to the encoder in this version. `P_aw` is still useful for *evaluation*, because the PEEP level can be read from it (Section 15.1).

---

## 3. Step 1 — Data inventory and sanity checks

Before any processing, for every recording file:

1. **List channels, units, sampling rate and duration.** Confirm that all 6 channels share the same time base (same `fs`, same number of samples, same start time).
2. **Record available metadata:** pig identifier, body weight if available, timestamps of interventions (endotoxin start/end, saline boluses, PEEP steps). If intervention timestamps are not provided, list which ones can be reconstructed from the signals (PEEP from `P_aw`) and which need the original protocol or the authors.
3. **Plot every channel over the whole recording** at low resolution, plus 10-second zooms at several time points. Look for gaps, flat lines, saturation, flushes, sign inversions, unit errors.
4. **Check inter-channel timing.** Within a beat, the expected order is: `P_lv` upstroke → `P_ao` upstroke ≈ `Q_ao` upstroke → `P_fem` upstroke. A violation suggests a channel delay introduced by the acquisition chain (filter lag), which must be corrected or at least documented.
5. **Check units against physiology:** mean `P_ao` ≈ 60–100 mmHg; stroke volume from `∫Q_ao dt` over a beat ≈ 15–40 mL for a 20–30 kg pig; `P_vc` ≈ 2–15 mmHg.

**Output of this step:** an inventory table (one row per pig: duration, channels present, known problems, intervention times) and a decision on which recordings and segments enter the pipeline.

Shape bookkeeping for one recording:

```
x : (M=6, N)      N = 250 · T_seconds        e.g. 1 h → N = 900 000
```

---

## 4. Step 2 — Artifact detection and cleaning

Artifacts are marked, not repaired. Each channel gets a boolean validity mask at the raw sampling rate:

```
valid : (M=6, N)   True where the sample is usable
```

A sample is marked invalid if any of the following holds (thresholds to tune after looking at the data):

| Artifact | Detection rule (per channel) |
|---|---|
| Missing data | NaN, or explicit gap in the time stamps |
| Flat line / disconnection | Standard deviation over a 1 s sliding window below a small threshold |
| Saturation | Value stuck at the min or max of the acquisition range for > 20 ms |
| Flush (pressure lines) | Pressure above ~200 mmHg sustained for > 0.5 s, square-wave shape |
| Out of physiological range | Outside the per-channel bounds of Section 2 (with margin) |
| Damping (pressure lines) | Pulse pressure dropping by > 50 % over a few beats without a matching change in the other pressures |

Each invalid run is extended by a small margin (e.g. 0.5 s on each side) to avoid keeping partial artifacts.

**No interpolation over artifacts.** Invalid stretches become *missing* beats for that signal (Section 7). The encoder is designed to handle missing data; inventing data would teach it wrong patterns.

---

## 5. Step 3 — Filtering

- **Low-pass:** zero-phase (forward–backward) 4th-order Butterworth, cutoff **25 Hz**, applied to all 6 signals.
  - Physiologically relevant content of pressure and flow is below ~20 Hz (about the first 10–15 harmonics of a ~1.7 Hz heart rate).
  - Zero-phase filtering preserves timing, which matters for transit delays and the order of upstrokes.
  - This also acts as the anti-aliasing filter for the resampling of Section 8.
- **No high-pass.** The DC level (mean pressure, mean flow, volume level) is information and must be kept.
- Filtering is applied on valid stretches only (or the whole signal, then invalid samples are re-masked), so that artifacts do not leak through the filter into neighboring valid data.

Shapes are unchanged: `x : (6, N)`.

---

## 6. Step 4 — Beat grid (segmentation)

All 6 signals are cut on **one common set of beat boundaries** per recording.

### 6.1 Detecting the pressure foot on `P_ao`

1. Compute the time derivative `dP_ao/dt` (on the filtered signal).
2. Find the peaks of `dP_ao/dt` (maximum upstroke slope), with a minimum distance of 0.3 s between peaks and a prominence threshold relative to the local signal amplitude. Each peak marks one beat.
3. For each peak, find the preceding diastolic minimum of `P_ao` within a 250 ms search window.
4. **Foot** = intersection of (a) the horizontal line through the diastolic minimum and (b) the tangent to `P_ao` at the point of maximum slope (*intersecting-tangent method*). This is more robust to noise than taking the minimum directly.

Result: foot times `f_1 < f_2 < … < f_{K+1}` (in samples).

### 6.2 Shifting boundaries to include the whole cardiac cycle

The LV contracts *before* the aortic valve opens (isovolumic contraction, roughly 30–60 ms in a pig). If beats were cut exactly at the aortic foot, the LV pressure upstroke would fall at the end of the *previous* beat. Boundaries are therefore shifted earlier by a fixed offset:

```
b_i = f_i − Δ_shift          Δ_shift ≈ 80 ms = 20 samples
```

**Check:** in recordings where `P_lv` is present, verify on a sample of beats that the `P_lv` upstroke (maximum `dP_lv/dt`) lies after `b_i`, i.e. inside the same beat as the corresponding aortic upstroke. Adjust `Δ_shift` if not.

### 6.3 When `P_ao` is missing or invalid

If `P_ao` is invalid over a stretch, the grid is built from another pressure, in this order of preference:

- `P_lv`: maximum `dP_lv/dt` minus a smaller offset (≈ 20 ms);
- `P_fem`: foot minus (transit delay + `Δ_shift`), where the transit delay is estimated over neighbouring stretches where both `P_ao` and `P_fem` are valid.

The reference used is stored per beat, so that the impact of the grid source can be analysed later.

### 6.4 Output

```
boundaries : (K+1,)     sample indices b_1 < … < b_{K+1}
rr         : (K,)       beat durations in seconds, rr_i = (b_{i+1} − b_i) / fs
t_beat     : (K,)       start time of each beat (seconds since start of recording)
grid_src   : (K,)       which signal defined the boundary (P_ao / P_lv / P_fem)
```

A pig at ~100 bpm gives ~6 000 beats per hour.

---

## 7. Step 5 — Beat-level quality control

Two levels of quality control.

### 7.1 Beat-level (the boundary itself)

A beat is **rejected entirely** (all signals) if:

- `rr_i` is outside [0.3 s, 1.2 s] (adjust to the observed heart-rate range), or
- `rr_i` differs from the median of the 10 surrounding beats by more than 25 % (ectopic beat or missed detection).

### 7.2 Signal-level (each signal within each beat)

Beat `i` of signal `m` is marked **unavailable** if:

- more than 5 % of its raw samples are invalid (Section 4), or
- a signal-specific plausibility check fails, for example:
  - `P_ao`, `P_fem`: pulse pressure < 5 mmHg or > 100 mmHg;
  - `Q_ao`: beat integral (stroke volume) ≤ 0 or far outside the recording's typical range;
  - `P_lv`: peak systolic pressure below diastolic `P_ao` (no ejection possible);
  - `V_lv`: beat-to-beat change in end-diastolic volume above a threshold (catheter jump).

### 7.3 Output

```
beat_ok : (K,)        beat kept at all
avail   : (K, 6)      signal m usable in beat i   (False if beat_ok is False)
```

Unavailable entries are not deleted: they become *missing* tokens parts that the encoder handles explicitly (Section 12.3).

---

## 8. Step 6 — Resampling beats to fixed length

Each beat of each signal, i.e. the raw segment `x[m, b_i : b_{i+1}]` (variable length, ≈ 125–200 samples), is resampled to **L = 64 points** on a uniform grid spanning `[b_i, b_{i+1})`, by linear interpolation.

- **Why 64:** a 0.6 s beat represented with 64 points is effectively sampled at ~107 Hz, i.e. content up to ~50 Hz is preserved, far above the 25 Hz of the low-pass filter. Aliasing is therefore not a concern. 128 points would double the cost without information gain.
- **What is lost:** absolute time inside the beat (a long and a short beat both become 64 points). This is restored by giving the beat duration `rr_i` to the network as a separate input (Section 12.4).

```
beats : (K, 6, 64)      float32, unavailable entries filled with 0 (never used, masked)
rr    : (K,)
avail : (K, 6)
```

Storage: 6 000 beats × 6 × 64 × 4 bytes ≈ 9 MB per hour of recording. The whole dataset fits in memory; windows (Section 10) are built on the fly by indexing.

---

## 9. Step 7 — Normalization

### 9.1 Per-signal global affine scaling

For each signal `m`, one center `c_m` and one scale `s_m` are computed **on the training pigs only** (recomputed for each cross-validation fold):

```
beats[:, m, :] ← (beats[:, m, :] − c_m) / s_m
```

Use robust statistics: `c_m` = median over all valid samples of all training beats, `s_m` = interquartile range / 1.35 (≈ standard deviation for Gaussian data, insensitive to residual artifacts).

**Never normalize per beat or per window.** Mean pressure, pulse pressure, flow level and LV volume are among the most important pieces of information; per-beat normalization would erase them.

### 9.2 Beat duration

```
r_i = (log rr_i − c_rr) / s_rr        c_rr, s_rr from training pigs
```

The logarithm makes heart-rate changes symmetric (a 10 % increase and a 10 % decrease have the same magnitude).

### 9.3 Signal-specific decisions (to settle after inspecting the data)

| Signal | Issue | Options |
|---|---|---|
| `Q_ao` | Zero offset of the flow probe may drift | Subtract, per recording, the median flow in late diastole (where true aortic flow ≈ 0). Recommended. |
| `V_lv` | Absolute level depends on catheter calibration, may differ between pigs | (a) keep absolute level (risk: encodes pig identity); (b) subtract a per-recording reference (e.g. median end-diastolic volume over the first 5 min), keeping changes only. Start with (b) and compare. |
| `P_vc` | Very small pulsatile amplitude compared with its scale | Fine as is with robust scaling; check that it is not dominated by noise. |

---

## 10. Step 8 — Windowing

A **window** is `W = 32` consecutive beats (≈ 19 s at 100 bpm, i.e. several respiratory cycles, so respiratory modulation of the cardiac signals is visible inside a window).

- **Stride:** 8 beats between consecutive windows (75 % overlap). Overlap gives more training examples at no cost but they are strongly correlated; this is accounted for in batching and evaluation.
- **Keeping a window:** at least 28 of its 32 beats must have `beat_ok = True`.
- **Signal availability at window level:** signal `m` is considered *present in the window* if it is available in at least 90 % of the beats of the window. Otherwise it is treated as absent for the whole window (all its entries set to unavailable). Signals present in the window can still have a few unavailable beats.

One window:

```
X : (W=32, M=6, L=64)     normalized beats            → 12 288 numbers
r : (32,)                 normalized beat durations
a : (32, 6)               availability mask
meta: pig id, start time, experimental phase (evaluation only)
```

Number of windows: (K − 32)/8 + 1 ≈ 750 per hour per pig. With 8 pigs and several hours each, roughly 20 000–40 000 windows, but the real diversity is closer to "8 animals × a handful of physiological states". This is the main limitation of the dataset and drives several choices below.

---

## 11. Step 9 — Data splits

- **Outer loop: leave-one-pig-out.** For each fold, one pig is the test pig; the encoder is trained on the others. With 8 pigs (or fewer after quality control), this gives 8 folds. Results are reported per fold and averaged.
- **Inner validation:** within the training pigs of a fold, one pig is held out for monitoring and hyperparameter choices (Section 14).
- **Everything computed from data is computed inside the fold:** normalization statistics (Section 9), probe fits (Section 15), thresholds tuned on data.
- **Never split windows of the same pig between training and test.** Overlapping windows from the same animal are nearly identical; random window splits would grossly overestimate performance.

---

## 12. Step 10 — The encoder architecture (shape by shape)

Overview (training-time shapes, batch of `B = 128` windows):

```
X (128, 32, 6, 64) , r (128, 32) , a (128, 32, 6)
   │
   │  12.1  per-signal beat CNNs            → (128, 32, 6, 64)
   │  12.2  missing-signal replacement      → (128, 32, 6, 64)
   │  12.3  fusion across signals           → (128, 32, 128)       one token per beat
   │  12.4  + position + duration           → (128, 32, 128)
   │  12.5  + CLS token                     → (128, 33, 128)
   │  12.6  transformer encoder (4 layers)  → (128, 33, 128)
   │  12.7  embedding head on CLS           → (128, 32)
   ▼
e (128, 32)
```

### 12.1 Per-signal beat CNN (one per signal, 6 in total)

Each signal has its own small 1D convolutional network, because a pressure beat, a flow beat and a volume beat have very different shapes. It turns one beat (64 points) into a vector of 64 numbers.

For one signal `m`, all beats of the batch are processed together, `n = B · W = 4 096` beats:

| Layer | Output shape | Parameters |
|---|---|---|
| input | `(n, 1, 64)` | – |
| Conv1d 1→16, kernel 7, stride 1, padding 3 + GELU | `(n, 16, 64)` | 128 |
| Conv1d 16→32, kernel 5, stride 2, padding 2 + GELU | `(n, 32, 32)` | 2 592 |
| Conv1d 32→32, kernel 5, stride 2, padding 2 + GELU | `(n, 32, 16)` | 5 152 |
| Flatten | `(n, 512)` | – |
| Linear 512→64 | `(n, 64)` | 32 832 |

≈ 40.7 k parameters per signal, ≈ 244 k for the 6 signals.

**Why flatten instead of pooling:** the position of features inside the beat (time of systolic peak, dicrotic notch, end of ejection) is information. Global average pooling would discard it.

Output for all signals: `U : (128, 32, 6, 64)`.

### 12.2 Missing-signal replacement

For every entry where `a[b, i, m] = False` (signal absent from the window, or unavailable in that beat), the CNN output is replaced by a **learned vector** `μ_m^miss ∈ R^64`, one per signal:

```
U[b, i, m, :] ← μ_m^miss     wherever a[b, i, m] = False
```

6 × 64 = 384 parameters. The network thus *knows* a signal is missing (rather than seeing zeros it might confuse with a real flat signal) and learns how to behave in that case.

### 12.3 Fusion across signals → one token per beat

The 6 vectors of each beat are concatenated and mapped to one token of width `D = 128`:

```
concat over signals : (128, 32, 6·64 = 384)
Linear 384→128      : (128, 32, 128)        49 280 parameters
LayerNorm           : (128, 32, 128)        256 parameters
```

This is the step where signals of the *same beat* are combined (e.g. LV pressure and aortic pressure during ejection, pressure and flow, aortic and femoral pressure with their delay). Relations *across beats* are handled by the transformer.

### 12.4 Position and duration information

Two learned additions, both of width 128:

- **Beat position embedding** `E_pos ∈ R^{32×128}` (4 096 parameters): tells the transformer that this is beat 1, 2, …, 32 of the window. Attention by itself has no notion of order; this is the only source of order information.
- **Duration embedding** `E_dur(r_i) = Linear 1→128 (r_i)` (256 parameters): restores the absolute beat duration lost by resampling.

```
token[b, i] = fused[b, i] + E_pos[i] + E_dur(r[b, i])        (128, 32, 128)
```

### 12.5 Summary token (CLS)

A learned vector `CLS ∈ R^128` (128 parameters) is placed in front of the sequence. It is not a beat; its role is to gather information from all beats through attention and become the summary of the window.

```
sequence : (128, 1 + 32 = 33, 128)
```

During training, some beats are hidden (Section 13.1). Hidden beats are removed from the sequence, so the sequence is shorter and padded to a common length; a padding mask `(128, 33)` tells the transformer which positions to ignore.

### 12.6 Transformer encoder

Standard bidirectional transformer encoder (each token can attend to all others, past and future, like BERT — not causal like GPT):

| Setting | Value |
|---|---|
| Number of layers | 4 |
| Token width `D` | 128 |
| Attention heads | 4 (32 dimensions each) |
| Feed-forward block | 128 → 512 → 128, GELU |
| Normalization | Pre-LayerNorm |
| Dropout | 0.1 |

Per layer: attention ≈ 66 k parameters, feed-forward ≈ 132 k → ≈ 198 k; 4 layers ≈ 0.79 M parameters. Attention over 33 tokens is computationally negligible.

```
H : (128, 33, 128)         H[:, 0] is the CLS output, H[:, 1:] are the beat outputs
```

### 12.7 Embedding head

```
e = Linear 128→32 ( LayerNorm( H[:, 0] ) )       e : (128, 32)
```

≈ 4.4 k parameters. **`e` is the embedding.** Everything after this point (Section 13) exists only to train the encoder and is discarded afterwards.

**Encoder total: ≈ 1.1 M parameters.**

---

## 13. Step 11 — How the encoder is trained (self-supervision)

No labels are used. The training signal comes from three objectives, each implemented by a small *training-only* component attached to the encoder. These components are **not prediction heads for flow**: they exist only to shape the embedding space and are thrown away after training.

| Objective | What it forces | Training-only component |
|---|---|---|
| A. Masked beat reconstruction | The encoder captures how beats and signals relate to each other (rich token features) | Small transformer decoder |
| B. Window template prediction | The 32 numbers of `e` themselves summarize the typical waveform of every signal | Small MLP from `e` |
| C. Two-view agreement (VICReg) | Same physiological state → same `e`, whatever the sensors, noise or exact beats seen | MLP projector |

### 13.1 Building two views of each window

For each window in the batch, two modified copies (views) are built. Each view is the result of the following random operations:

1. **Time shift (view b only).** View b is taken from the same recording, shifted by δ beats, δ uniform in {0, …, 4}. The physiological state barely changes over 4 beats, so both views should have the same embedding. This encourages smoothness of `e` over time.
2. **Signal subset (sensor dropout).** Each signal present in the window is kept with probability 0.7, with at least one kept. Dropped signals are set to unavailable for the whole window (`a[:, m] = False`). Each view draws its own subset independently. This is what teaches consistency across sensor subsets (property 3).
3. **Beat masking.** A fraction of beats (≈ 50 %) is hidden from the encoder. Hidden beats are chosen as **contiguous spans of 3–4 beats**, not isolated beats: neighbouring beats are almost identical, so hiding isolated beats would let the network solve reconstruction by copying a neighbour, which teaches nothing.
4. **Small signal augmentations** (mild, so as not to destroy information):
   - additive Gaussian noise (≈ 1–2 % of each signal's scale);
   - small constant offset on pressures (± 1 mmHg, mimicking catheter zeroing errors);
   - beat-grid jitter: before resampling, each boundary is shifted by ± up to 10 samples, so the encoder becomes robust to grids defined from different reference signals (Section 6.3).
   - **Not applied:** amplitude scaling, time stretching, or anything else that changes physiologically meaningful features. Every augmentation declares "this change does not matter"; scaling pressure would declare that blood pressure level does not matter, which is false.

Per view, the bookkeeping is:

```
present : (B, 32, 6)   entry exists in the data (before any dropout)
input   : (B, 32, 6)   entry is given to the encoder (after sensor dropout and beat masking)
target  : present AND NOT input      → entries to reconstruct in Objective A
```

### 13.2 Objective A — Masked beat reconstruction

A lightweight transformer decoder tries to rebuild the beats the encoder did not see, including the signals that were dropped in the view.

Decoder input, full length (all 32 beats + the embedding):

- For beats seen by the encoder: their encoder outputs `H[:, 1:]`, projected 128 → 64.
- For hidden beats: a learned **mask vector** `∈ R^64`.
- Added to every position: decoder position embedding `(32, 64)` and duration embedding `Linear 1→64 (r_i)`.
- One extra token: the embedding `e` projected `Linear 32→64`.

```
decoder input  : (B, 1 + 32, 64)
2 transformer layers, width 64, 4 heads, feed-forward 64→256→64
output head    : Linear 64 → 6·64 = 384, reshaped to (B, 32, 6, 64)
```

Loss: mean squared error between reconstruction and true normalized beats, computed **only on `target` entries** (beats hidden from the encoder, and signals dropped from the view, wherever the true data exist). Entries that are unavailable in the raw data are never used as targets.

Two parts of this loss do different jobs:

- reconstructing hidden beats of signals that *were* given → learning temporal patterns (respiratory modulation, beat-to-beat variability);
- reconstructing signals that were *dropped* → learning cross-signal relations (e.g. LV volume from pressures, flow from pressures). This is the most informative part.

≈ 0.13 M parameters (training only).

### 13.3 Objective B — Window template prediction

The decoder of Objective A can rebuild beats mostly from the visible beats and partly ignore `e`. Objective B closes that loophole by asking `e` alone to summarize the window.

From `e` only, predict for each of the 6 signals:

- the **mean beat** over the 32 beats of the window, and
- the **standard deviation** across beats at each of the 64 points (captures beat-to-beat variability, notably respiratory modulation such as pulse pressure variation).

```
e (B, 32) → Linear 32→256 + GELU → Linear 256 → 2·6·64 = 768 → reshape (B, 2, 6, 64)
```

Targets are computed from the **full** window (all signals present in the data, before sensor dropout and masking), so `e` must also carry information about signals that were not in the view. Loss: mean squared error over available signals.

≈ 0.2 M parameters (training only).

### 13.4 Objective C — Two-view agreement (VICReg)

The embeddings of the two views, `e_a` and `e_b`, are passed through a projector (training only):

```
e (B, 32) → Linear 32→128 + BatchNorm + ReLU
          → Linear 128→128 + BatchNorm + ReLU
          → Linear 128→128                          z : (B, 128)
```

≈ 37 k parameters. The projector is standard practice: the agreement constraints are applied on `z`, leaving `e` free to keep information that `z` discards.

VICReg loss on `z_a, z_b` (each `(B, 128)`):

1. **Invariance:** mean squared difference between `z_a` and `z_b` → both views must agree.
2. **Variance:** for each of the 128 dimensions, the standard deviation over the batch must stay above 1 (hinge penalty `max(0, 1 − std)`), for each view → prevents *collapse*, i.e. the trivial solution where every window gets the same embedding.
3. **Covariance:** sum of squared off-diagonal entries of the 128×128 covariance matrix of `z` (divided by 128), for each view → different dimensions must carry different information.

Standard weights: 25 (invariance), 25 (variance), 1 (covariance).

The projector width (128) is kept at most equal to the batch size (128); otherwise the covariance matrix estimated from the batch is degenerate.

**Batch composition matters for VICReg.** If a batch contains several overlapping windows from the same pig and time, the variance term pushes apart windows that are physiologically identical. Batches are therefore built by sampling windows across pigs and across time, avoiding windows closer than ~1 minute from the same recording in the same batch.

### 13.5 Total training loss

```
L = L_A(view a) + L_A(view b)                      masked reconstruction
  + β · [ L_B(view a) + L_B(view b) ]              template prediction
  + γ · L_C(view a, view b)                        VICReg
```

- Start with β = 1.
- Choose γ so that the VICReg term has roughly the same magnitude as the reconstruction terms at the start of training (VICReg values are typically tens, reconstruction errors on normalized data are typically below 1, so γ ≈ 0.02–0.05 as a first guess).
- After a first run, check that no term dominates the gradients; rebalance if one loss stays flat while another collapses.

---

## 14. Step 12 — Optimization and monitoring

### 14.1 Optimization

| Setting | Value (starting point) |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5·10⁻⁴, linear warm-up over the first 5 % of steps, then cosine decay to 10⁻⁵ |
| Weight decay | 0.05 (not applied to biases, LayerNorm, embeddings) |
| Batch size | 128 windows (× 2 views) |
| Gradient clipping | global norm 1.0 |
| Training length | ~100 passes over the training windows, or until the monitoring criteria below stop improving |
| Precision | float32 (the model is small; mixed precision optional on GPU) |

### 14.2 What to monitor during training

The training loss alone does not say whether the embedding is meaningful. Monitor, every few epochs:

1. **Each loss term separately** (A, B, C, and the three VICReg components).
2. **Collapse indicators on `e`** (on the validation pig):
   - standard deviation of each of the 32 dimensions across windows — any dimension near 0 is dead;
   - **effective rank** of the covariance of `e`: exp(entropy of the normalized eigenvalues). Close to 32 = all dimensions used; close to 1 = collapsed.
3. **A quick probe** on the validation pig (Section 15.1, a few targets such as heart rate, mean `P_ao`, stroke volume): if probe quality stops improving or degrades while the loss keeps decreasing, training is overfitting to the training animals.
4. **Reconstruction examples:** plots of a few windows with true vs reconstructed beats, including dropped signals.

The checkpoint kept is chosen on the validation pig's probe quality, not on the training loss.

---

## 15. Step 13 — Evaluating the embedding space

All evaluations are done on the **test pig of each fold** (never seen by the encoder), with the encoder frozen. The four properties of Section 1 each have a test.

### 15.1 Property 1 — Contains physiology (linear probes)

**Targets computed per window from the raw signals** (these are evaluation labels, never used in training):

| Group | Targets |
|---|---|
| Heart rate / timing | heart rate; pulse transit time `P_ao` → `P_fem` |
| Arterial pressure | mean, systolic, diastolic and pulse pressure of `P_ao` and `P_fem`; pulse pressure variation over the window |
| Flow | stroke volume (∫`Q_ao` over a beat), cardiac output |
| Arterial load | Windkessel `R`, `C`, `Z_c` fitted per window from `P_ao` and `Q_ao` |
| Left ventricle | end-diastolic and end-systolic volume (relative), ejection fraction (relative), max `dP_lv/dt` (contractility index), end-diastolic `P_lv` (preload index) |
| Venous | mean `P_vc` |
| Experimental context | phase label (baseline, PEEP level, post-endotoxin, post-fluid); PEEP level read from `P_aw` |

**Probe:** ridge regression (continuous targets) or logistic regression (phase labels) from `e` to the target, fitted on the training pigs' embeddings with the regularization strength chosen by cross-validation across training pigs, evaluated on the test pig. Report R² (or balanced accuracy).

**Two probing regimes**, both important:

- **Full input:** embedding computed with all available signals.
- **Reduced input:** embedding computed from a subset that *excludes the signal the target is derived from*, e.g. stroke volume probed from an embedding computed on pressures only. Good performance here means the encoder has learned cross-signal relations, not just copied input statistics. This is the most informative test for the final pressure → flow goal.

### 15.2 Property 2 — Tracks state changes

- **Trajectories:** for the test pig, project the sequence of window embeddings over time to 2D (PCA fitted on the training pigs) and plot it, coloured by experimental phase and annotated with intervention times.
- **Quantitatively:**
  - *within-phase stability*: average distance between consecutive embeddings inside a stable phase;
  - *between-phase separation*: distance between the mean embedding of consecutive phases, divided by the within-phase spread;
  - *change detection*: does a simple change-point detector on `e` over time find the intervention times (and with what delay)?

### 15.3 Property 3 — Consistency across sensor subsets

For the same test windows, compute embeddings from several subsets, for example:

```
S_full = all 6 signals
{P_ao}, {P_fem}, {P_ao, P_fem}, {P_ao, Q_ao}, {P_lv, V_lv}, {P_fem, P_vc}
```

Report:

- cosine similarity (or Euclidean distance relative to the typical spread of `e`) between `e(S)` and `e(S_full)` for the same window;
- **probe transfer:** fit the probe on full-input embeddings, apply it unchanged to subset embeddings. If the subset embeddings live in the same space, the probe still works (with some loss).

### 15.4 Property 4 — No memorization of the animal

- Within the training pigs, fit a classifier predicting the pig identity from `e`, using a time-based split (first part of each recording for fitting, last part for testing, so that overlapping windows do not inflate the score).
- Compare with how well physiology is predicted. If pig identity is nearly perfectly predictable while physiology probes are mediocre, the encoder is memorizing animals: increase augmentation, reduce capacity, reconsider absolute-level signals such as `V_lv` (Section 9.3), or add simulated data.

### 15.5 Baselines (needed to interpret every number above)

1. **Handcrafted features:** the vector of standard per-window features (means, pulse pressures, heart rate, max derivatives, decay time constant, …) fed to the same probes. The learned embedding should do at least as well with full input, and should clearly do better in the reduced-input regime.
2. **Untrained encoder:** same architecture, random weights. Shows what training actually adds.
3. **Raw-template PCA:** mean beat per signal over the window (6 × 64 numbers), reduced to 32 dimensions by PCA fitted on training pigs. A strong, simple, linear baseline.

### 15.6 What to report

One table per property, rows = methods (learned encoder + baselines), columns = targets or subsets, values averaged over leave-one-pig-out folds with the spread across folds. Plus trajectory figures for each test pig.

---

## 16. Step 14 — Using the trained encoder (inference)

For a new recording with any subset `S` of the 6 signals:

1. Steps 1–7 (inventory, artifact masking, filtering, beat grid from the best available pressure, beat QC, resampling, normalization with the **training** statistics).
2. Sliding windows of 32 beats, stride 8 (or stride 1 for a beat-by-beat updated embedding).
3. For each window: signals not in `S` are marked unavailable; no beat masking, no augmentation; one forward pass through the encoder (Section 12).
4. Output: a time series of embeddings

```
E : (number of windows, 32)       one 32-dimensional point every 8 beats (≈ 5 s)
```

Only the components of Section 12 are needed (≈ 1.1 M parameters). The training-only components of Section 13 are discarded.

Latency note: the encoder is bidirectional over a 32-beat window, so an embedding describes the last ~19 s and is available at the end of the window. A causal variant (each beat only attending to previous beats) can be considered later for real-time use.

---

## 17. Summary tables

### 17.1 Shapes along the pipeline

| Stage | Object | Shape |
|---|---|---|
| Raw recording | `x` | `(6, 250·T)` |
| Validity mask | `valid` | `(6, 250·T)` |
| Beat grid | `boundaries`, `rr` | `(K+1,)`, `(K,)` |
| Resampled beats | `beats`, `avail` | `(K, 6, 64)`, `(K, 6)` |
| One window | `X`, `r`, `a` | `(32, 6, 64)`, `(32,)`, `(32, 6)` |
| Batch | `X`, `r`, `a` | `(128, 32, 6, 64)`, `(128, 32)`, `(128, 32, 6)` |
| After per-signal CNNs | `U` | `(128, 32, 6, 64)` |
| After fusion | beat tokens | `(128, 32, 128)` |
| With CLS | sequence | `(128, 33, 128)` |
| Transformer output | `H` | `(128, 33, 128)` |
| **Embedding** | **`e`** | **`(128, 32)`** |
| *Training only:* projector | `z` | `(128, 128)` |
| *Training only:* decoder output | reconstructed beats | `(128, 32, 6, 64)` |
| *Training only:* template output | mean and std beats | `(128, 2, 6, 64)` |

### 17.2 Parameter counts

| Component | Parameters | Kept after training |
|---|---|---|
| 6 per-signal beat CNNs | ≈ 244 k | yes |
| Missing-signal vectors | 384 | yes |
| Fusion (Linear 384→128 + LayerNorm) | ≈ 49.5 k | yes |
| Position + duration embeddings + CLS | ≈ 4.5 k | yes |
| Transformer encoder (4 layers) | ≈ 0.79 M | yes |
| Embedding head | ≈ 4.4 k | yes |
| **Encoder total** | **≈ 1.1 M** | |
| Decoder (Objective A) | ≈ 0.13 M | no |
| Template head (Objective B) | ≈ 0.2 M | no |
| VICReg projector (Objective C) | ≈ 37 k | no |

### 17.3 Hyperparameters (starting values)

| Category | Hyperparameter | Value |
|---|---|---|
| Preprocessing | Low-pass cutoff | 25 Hz, zero-phase, 4th order |
| | Beat boundary shift `Δ_shift` | 80 ms before the aortic foot |
| | Points per beat `L` | 64 |
| | Normalization | Global per signal, robust (median, IQR), training pigs only |
| Windowing | Beats per window `W` | 32 |
| | Stride | 8 beats |
| | Window kept if | ≥ 28 valid beats |
| | Signal present in window if | available in ≥ 90 % of beats |
| Architecture | Per-signal CNN output | 64 |
| | Token width `D` | 128 |
| | Transformer | 4 layers, 4 heads, feed-forward 512, dropout 0.1 |
| | Embedding size `d_e` | 32 |
| Views | Time shift δ | 0–4 beats |
| | Signal keep probability | 0.7 (≥ 1 signal kept) |
| | Beat masking | ≈ 50 %, spans of 3–4 beats |
| | Noise / offset / grid jitter | 1–2 % / ± 1 mmHg / ± 10 samples |
| Losses | β (template) | 1 |
| | γ (VICReg) | ≈ 0.02–0.05, tuned to balance magnitudes |
| | VICReg internal weights | 25 / 25 / 1 |
| Optimization | Optimizer, learning rate | AdamW, 5·10⁻⁴, warm-up + cosine |
| | Weight decay | 0.05 |
| | Batch size | 128 windows |
| Evaluation | Protocol | Leave-one-pig-out, inner validation pig |

---

## 18. Open decisions and risks

### 18.1 Decisions to make after inspecting the data

1. **Units and sign of `Q_ao`**, and how aortic flow was measured (check the earlier papers on the same experiments: Murphy et al. 2020, *Ann. Biomed. Eng.*; Cushway et al. 2022, *Comput. Biol. Med.* and *Comput. Methods Programs Biomed.*).
2. **Treatment of the absolute level of `V_lv`** (Section 9.3).
3. **Availability of intervention timestamps**, needed for the evaluation of Section 15.2.
4. **Value of `Δ_shift`**, checked against the `P_lv` upstroke.
5. **Which pigs and segments are usable** after quality control (the paper excluded some pigs for airway or interrupted data; the cardiac channels of those pigs may still be usable here).

### 18.2 Design choices worth testing later

- Window length 16 vs 32 vs 64 beats (reactivity vs stability).
- Embedding size 16 vs 32 vs 64.
- One token per beat (this document) vs one token per beat *per signal* (6 × 32 = 192 tokens): the latter handles missing signals more naturally and lets attention link signals explicitly, at a higher cost.
- Positive pairs for VICReg: same window only vs shifted windows (current) vs also windows from the same stable experimental phase.
- Adding the airway signals (`P_aw`, `Q_aw`) as per-beat scalars or as separate breath tokens.

### 18.3 Main risks

| Risk | Symptom | Mitigation |
|---|---|---|
| Too few animals | Good training loss, poor test-pig probes; pig identity easily decodable | Strong augmentation, small model, leave-one-pig-out monitoring, pretraining on simulated data and on large human pressure datasets |
| Collapse | Effective rank of `e` drops, all embeddings similar | VICReg variance term, monitor effective rank, check γ |
| Shortcut learning in reconstruction | Reconstruction is good but `e` is uninformative | Span masking, Objective B, probes on reduced inputs |
| Shortcut through absolute calibration offsets | Embedding encodes catheter calibration rather than physiology | Per-recording referencing of unreliable absolute levels (`V_lv`, flow zero) |
| Beat-grid mismatch at inference | Embeddings shift when the grid comes from `P_fem` or `P_lv` | Grid jitter augmentation; train some views with grids from other reference signals |

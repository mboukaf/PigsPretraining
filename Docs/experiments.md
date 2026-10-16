# Encoder training experiments

Log of the self-supervised training runs of the encoder described in
`embedding_space_procedure.md`: what was changed, why, and what happened.
Code: `src/model/train.py`. Each run writes `runs/<run>/fold<k>/log.json`;
`python -m model.compare <runs…>` (from `src/`) prints the summary tables used below.

> **TL;DR (2026-10-13):** on this dataset the self-supervised encoder does not beat classical
> waveform features on any test (physiology probes, cross-signal targets, missing signals,
> detection of PEEP manoeuvres); its only unique property is a sensor-invariant representation.
> Diagnosis: cross-animal generalization with 2–6 training animals. See **Overall conclusion** at
> the end; recommended next step: more / simulated data (option C).

## How runs are judged

Selection uses the **validation pig**, never the training loss (spec §14.2).

- **Probe R²**: ridge regression from the embedding `e` to four per-window targets, fitted on
  the training pigs' windows and scored on the validation pig:
  heart period, mean P_ao, P_ao pulse pressure, mean Q_ao.
  The run's score is the mean of the four R², each clipped at −1 so that one target that
  fails badly cannot dominate.
- **Effective rank of `e`** on the validation pig: exp(entropy of the covariance eigenvalues).
  32 means all embedding dimensions are used; 1 means collapse.
- **Validation reconstruction loss (A)**: if it rises while the training loss falls, the
  encoder is fitting the training animals rather than learning transferable structure.
- **Epoch 0** is the untrained encoder (spec §15.5 baseline "untrained encoder").

Common settings unless stated: fold 7 (test pig 7, validation pig 8, training pigs 1–6),
10 epochs, batch 128, AdamW 5e-4 with warm-up and cosine decay, β = 1, MPS (~25 s/epoch).

---

## E0 — first end-to-end run (spec defaults, γ = 0.1)

Setup: Q_ao zero-referenced, V_lv level-referenced (`qao_mode=zero`, `vlv_mode=relative`).

| | untrained | epoch 2 (best) | epoch 10 |
|---|---|---|---|
| train A | 3.06 | 0.71 | 0.19 |
| **val A** | – | **1.01** | **1.42** |
| probe heart period | 0.91 | 0.88 | 0.48 |
| probe mean P_ao | 0.70 | 0.83 | 0.84 |
| probe pulse pressure | −0.35 | 0.58 | 0.50 |
| probe mean Q_ao | −23 | −12 | −29 |
| effective rank (val / train) | 3.3 / – | 2.9 / 5.2 | 3.2 / 7.4 |

Findings:
1. **Overfits the training animals after ~2 epochs.** Training reconstruction keeps falling,
   validation reconstruction rises from epoch 2 on (spec risk "too few animals").
2. **Mean Q_ao is not transferable between pigs** (R² −12 to −29). Typical peak flow differs
   5.7× between pigs of similar size (0.77 L/min mean flow in pig 8 vs 5 L/min in pig 3), i.e.
   flow-probe calibration, not physiology. Besides breaking the probe, the absolute flow level
   lets the encoder identify the animal (spec §18.3, "shortcut through absolute calibration
   offsets"), which plausibly drives finding 1. V_lv amplitude shows the same issue (3.3×).
3. **Low effective rank** (≈3 of 32 on the validation pig, VICReg covariance term ≈9 and not
   decreasing): the embedding dimensions stay strongly correlated at γ = 0.1.

## E1 — remove flow / volume calibration (scaled Q_ao and V_lv)

Change: `qao_mode=scaled`, `vlv_mode=scaled` (`src/data/normalize.py`). After the existing
referencing, each recording's Q_ao is divided by its median beat peak flow and V_lv by its median
beat amplitude. Shape and within-recording relative changes are kept (a 30 % drop of stroke volume
still reads as 0.7); the calibration factor that identifies the pig is removed. The Q_ao probe
target becomes "mean flow relative to the pig's typical flow".

Results (10 epochs, evaluated every epoch):

| epoch | val A | probe heart period | mean P_ao | pulse pressure | mean Q_ao (relative) |
|---|---|---|---|---|---|
| 0 (untrained) | 2.73 | 0.92 | 0.85 | 0.20 | **0.82** |
| 2 | 0.75 | 0.87 | 0.85 | −0.09 | 0.58 |
| 5 (saved) | 0.69 | 0.82 | 0.73 | 0.47 | 0.73 |
| 10 | 0.74 | 0.74 | 0.69 | 0.48 | 0.70 |

Findings:
1. **The calibration fix works**: the untrained encoder's mean-Q_ao probe goes from −23 (E0) to
   +0.82, and **validation reconstruction no longer rises** (0.68–0.75 after epoch 2, vs
   1.01 → 1.42 in E0). The overfitting of E0 was largely the calibration shortcut.
2. **The full-input probes cannot show what training adds.** The untrained encoder is already
   at 0.92 / 0.85 on heart period / mean P_ao: these targets are averages of the encoder's own
   inputs, and a random 32-d projection keeps that linear information (this is why the spec
   lists "untrained encoder" as a baseline, §15.5). Training slightly lowers them.
3. **Reduced-input probes do show it** (`python -m model.evaluate`, new, spec §15.1):

   | probe | untrained | trained (ep 5) |
   |---|---|---|
   | relative flow from **P_ao only** | −0.49 | **+0.25** |
   | relative flow from all signals but Q_ao | 0.48 | 0.45 |
   | pulse pressure from P_ao only | 0.57 | 0.60 |

   Predicting flow from a single aortic pressure line, which a random encoder cannot do, is the
   cross-signal learning the spec aims for and the closest test to the final pressure → flow goal.

## E2, E3 — stronger decorrelation (γ = 0.3, 0.5), scaled modes

Change: VICReg weight γ, to raise the effective rank (finding 3 of E0).

Results: no improvement. Effective rank on the validation pig stays ≈ 3, VICReg covariance ≈ 9,
validation reconstruction 0.68–0.71, full-input probes no better than E1. Flow from P_ao only
reaches 0.62–0.64 at epoch 1 (checkpoints chosen by the old full-input score).

Interpretation (revised): the low effective rank is measured within **one** pig's 4-hour
recording, which plausibly contains only a few independent physiological trends; across the six
training pigs it is ≈ 7. VICReg decorrelates the projector output z, not e (the spec keeps e free).
So rank ≈ 3 on a single validation pig is not by itself a defect. **Decision: keep γ = 0.1.**

## Changes after E3

- **Checkpoint selection now includes the reduced-input probes** (`model/train.py`,
  `selection_score`): mean of the clipped R² of the 4 full-input probes and of the 2
  relative-flow probes (from P_ao only, and from everything but Q_ao). The old full-input score
  rewarded copying input statistics, which the untrained encoder already does, and picked
  epoch 1 for E2 / E3.
- **Sensor dropout strength is a parameter** (`--keep-prob`): dropping signals more often should
  force more cross-signal inference, which the reduced-input probes reward. A 1-epoch smoke test
  at keep 0.5 already gave flow-from-P_ao R² = +0.54.
- `model/compare.py` reports the best **trained** epoch (= the saved checkpoint) separately from
  the untrained epoch 0.

## E4, E5, E6 — sensor-dropout strength (keep probability 0.7 / 0.5 / 0.3)

Setup: scaled Q_ao / V_lv, γ = 0.1, new selection score, evaluation every 2 epochs.
Untrained score on this fold: 0.463 (flow from P_ao only −0.49, flow without Q_ao 0.48).

| run | keep prob | best score (epoch) | heart period | mean P_ao | pulse pressure | mean Q_ao | flow from P_ao only | flow without Q_ao |
|---|---|---|---|---|---|---|---|---|
| E4 | 0.7 (spec) | 0.577 (10) | 0.74 | 0.69 | 0.48 | 0.70 | 0.42 | 0.43 |
| E5 | 0.5 | 0.563 (10) | 0.94 | 0.79 | 0.34 | 0.63 | 0.41 | 0.27 |
| E6 | 0.3 | **0.639** (4) | 0.95 | 0.93 | 0.35 | 0.77 | 0.27 | **0.57** |

Findings:
1. **All three beat the untrained encoder** on the selection score (0.56–0.64 vs 0.46), mainly
   through the reduced-input flow probes.
2. **The differences between them are within the noise.** The same probe moves by ±0.2 between
   consecutive evaluations of one run (E4 flow-from-P_ao: 0.35, 0.10, 0.19, 0.37, 0.42). With one
   validation pig and one seed, these three settings cannot be ranked.
3. Stronger dropout (0.3) keeps the full-input probes closest to the untrained encoder
   (0.93–0.95 for heart period / mean P_ao) while still adding cross-signal information.

Decision: evaluate the spec setting (0.7) and the strongest dropout (0.3) on more folds before
choosing (E4 and E6 extended to folds 2, 4, 6).

## E4 / E6 on folds 2, 4, 6 — the probe metric breaks on other validation pigs

| fold (val pig) | untrained score | E4 (keep 0.7) | E6 (keep 0.3) | flow from P_ao only, untrained → E4 |
|---|---|---|---|---|
| 2 (pig 3) | −0.29 | −0.10 | −0.27 | −39.7 → −61.1 |
| 4 (pig 5) | −0.42 | 0.29 | 0.14 | −4.3 → 0.04 |
| 6 (pig 7) | 0.13 | 0.03 | −0.07 | −1.4 → −1.3 |
| 7 (pig 8) | 0.46 | 0.58 | 0.64 | −0.5 → 0.42 |

**Even the untrained encoder** gets catastrophic flow R² on some validation pigs (−39.7 on pig 3).
Cause: R² is computed **within one validation pig**, where these per-window targets vary little,
so any constant offset between pigs dominates it. After per-recording scaling, "mean flow
relative to the pig's peak flow" still differs between pigs through waveform shape (mean/peak
ratio; pig 3's beats differ), and pressures differ in level between animals. The absolute level
of the relative-flow target is neither transferable nor meaningful (it was removed on purpose by
the scaling); what matters is whether the embedding **follows changes within the new animal**
(spec properties 1–2).

## Changes after E6

- **Within-pig R²** (`model/monitor.py`): predictions and targets are centred on the validation
  pig before computing R². Both scores are logged (`probe_<target>` and `probe_<target>@within`);
  checkpoint selection uses the within-pig version.
- **An encoder checkpoint is saved at every evaluation** (`encoder_ep<k>.pt`), so other selection
  rules can be applied afterwards without retraining.

## E7, E8 — keep probability 0.7 vs 0.3 on all 8 folds (within-pig selection)

Setup: scaled Q_ao / V_lv, γ = 0.1, 10 epochs, evaluation every 2 epochs, validation pig = next
pig after the test pig. 16 runs, all completed.

**Validation pigs** (within-pig R², mean over 8 folds; flow probes NaN for pig 1, no Q_ao):

| | untrained | **E7 keep 0.7 (spec)** | E8 keep 0.3 |
|---|---|---|---|
| selection score | 0.30 | **0.49** (> untrained in **7/8** folds) | 0.40 (4/8) |
| mean P_ao | 0.54 | **0.80** | 0.81 |
| pulse pressure | 0.11 | **0.59** | 0.48 |
| heart period | 0.97 | 0.91 | 0.94 |
| mean Q_ao (full input) | 0.63 | 0.63 | −0.24 |
| flow from P_ao only (median) | −1.61 | −0.13 | – |

Decision: **E7 (the spec's keep probability 0.7) is the chosen configuration**; it is more stable
across animals than stronger dropout. Its settings are now the defaults of `model/train.py`
(scaled Q_ao / V_lv, γ 0.1, keep 0.7).

## Final check — E7 on the held-out test pigs

`python -m model.evaluate E7 --split test`: probes fitted on the training pigs of each fold,
scored on that fold's test pig (never used for training or checkpoint selection).

| within-pig R², 8 test pigs | untrained mean / median | trained mean / median | folds improved |
|---|---|---|---|
| mean P_ao | 0.59 / 0.75 | **0.79 / 0.84** | **6/8** |
| pulse pressure | 0.07 / 0.47 | **0.51** / 0.47 | 5/8 |
| heart period | 0.96 / 0.98 | 0.90 / 0.93 | 1/8 |
| mean Q_ao (full input) | 0.36 / 0.72 | 0.55 / 0.61 | 2/8 |
| relative flow from P_ao only | −2.43 / −1.04 | −6.69 / −0.24 | 4/8 |
| relative flow without Q_ao | −1.50 / 0.32 | −0.95 / −0.40 | 2/8 |
| pulse pressure from P_ao only | 0.74 / 0.83 | 0.52 / 0.65 | 0/8 |

## Status and conclusions (after E8)

What works:
- The whole pipeline runs end to end and reproducibly: preprocessing → windows → two views →
  encoder + three objectives → per-fold training (≈ 25 s / epoch on MPS) → validation-based
  checkpoint selection → test-pig evaluation.
- Removing flow / volume calibration (scaled modes) fixed the overfitting seen in E0.
- The trained encoder captures **arterial pressure state** better than an untrained encoder on
  unseen animals: mean pressure improves on 6/8 test pigs (median R² 0.75 → 0.84) and pulse
  pressure on 5/8.

What does not work yet:
- **Cross-signal flow from pressure does not transfer reliably to new animals.** On test pigs the
  relative-flow probes improve on only 2–4 of 8 folds and stay negative in median. This is the
  spec's most informative test (§15.1) and the one that matters for the final pressure → flow goal.
- Heart period and single-line pulse pressure are slightly *worse* than the untrained encoder,
  which already represents them almost perfectly.
- Results vary strongly between animals: with 8 pigs, one unusual animal (e.g. pig 3: largest
  stroke volume, ringing-free but differently shaped flow) moves the averages a lot.

Likely reasons, in order of plausibility:
1. **Too few animals for the flow relationship.** The pressure → flow mapping depends on each
   animal's arterial properties and on the flow probe's placement; 6 training pigs per fold do not
   cover that variability (spec §18.3, "too few animals").
2. **Flow data quality differs by pig** (ringing in pig 5, different waveform shape in pig 3,
   pig 1 unusable), so the reconstruction target itself is inconsistent across animals.
3. The probes are linear on a 32-d embedding learned without labels; within one pig the target
   variance is small, so within-pig R² is a demanding metric.

Options from here (not done; they change the research plan):
- Pretraining on simulated cardiovascular data and/or large human arterial-pressure datasets,
  then fine-tuning on the pigs (spec §18.3 mitigation).
- Report **within-pig correlation** alongside R² (scale-free; R² also penalises gain errors).
- Longer training (spec suggests ~100 epochs) — not expected to help much: validation
  reconstruction plateaus by epoch 2–6 in every run.
- Beat-grid jitter augmentation (deferred, §13.1) and grids from P_lv / P_fem at training time.

Reproduce the chosen configuration on all folds (from `src/`):

    for k in 1 2 3 4 5 6 7 8; do python -m model.train --test-pig $k --epochs 10 --eval-every 2 --run-name E7; done
    python -m model.compare E7
    python -m model.evaluate E7 --split test

## Which pigs? Paper exclusions vs our quality control

The paper (Cushway et al. 2024) used only pigs 4, 6, 7, 8: pigs 1 and 5 for airway flow (not an
encoder input), pigs 2 and 3 for "multiple interruptions in the data stream". All experiments
above use all 8 pigs, with interruptions masked by Steps 2 and 5 (spec §18.1.5).

Splitting E7's test-pig results by group (within-pig R², flow from P_ao only):

| test pig | untrained | trained |
|---|---|---|
| 2 (paper: interruptions) | 0.05 | **−15.95** |
| 3 (paper: interruptions) | −8.85 | **−30.29** |
| 5 | −3.24 | 0.11 |
| paper pigs 4, 6, 7, 8 (median) | −0.54 | −0.21 (pig 8: 0.73) |

The catastrophic flow failures are exactly pigs 2 and 3; their flow (and pig 3's waveform shape,
see Step 6) seems inconsistent with the other animals even after masking. Pressure gains, on the
other hand, come mostly from the excluded pigs. Hypothesis: pigs 2 / 3 (and possibly 5) hurt the
flow relationship, while their pressures help.

Code: `--pigs` and `--no-flow-pigs` options (`data/store.py:select_recordings`, used by both
`model/train.py` and `model/evaluate.py`, so evaluation rebuilds exactly the run's data).

## E9 — paper pigs only (4, 6, 7, 8)

Same as E7 but `--pigs 4 6 7 8`: 4 folds, 2 training pigs per fold (2.1k–4.2k training windows
instead of ~9.7k).

Results: see the comparison below.

## E10 — all pigs, flow only from trusted pigs

Same as E7 but `--no-flow-pigs 2 3 5`: Q_ao unavailable for pigs 2, 3, 5 (pig 1 already had no
usable Q_ao), so flow comes only from pigs 4, 6, 7, 8 while all pressures are kept. 8 folds.

## E7 vs E9 vs E10 on the same test pigs (4, 6, 7, 8)

Test-pig within-pig R², median over the 4 pigs (`python -m model.evaluate E9 E10 --split test`):

| | E7 all pigs | **E9 paper pigs only** | E10 all pigs, flow from 4/6/7/8 only |
|---|---|---|---|
| **relative flow from P_ao only** | −0.21 | **+0.54** (> 0 on 3/4: 0.72, 0.74, 0.36; pig 6 −0.93) | −2.19 |
| mean Q_ao (full input) | 0.50 | **0.61** | 0.31 |
| relative flow without Q_ao | −0.24 | 0.06 | 0.28 |
| **mean P_ao** | **0.82** | 0.60 | **0.82** |
| pulse pressure | **0.62** | 0.53 | 0.42 |
| heart period | 0.91 | 0.92 | 0.90 |

Findings:
1. **Training on the paper's pigs only gives the best flow-from-pressure** — the first configuration
   where this probe is positive on most test pigs. The paper's exclusions matter for flow.
2. **But pressure generalizes worse** with only 2 training pigs per fold (mean P_ao 0.82 → 0.60):
   the excluded pigs' pressures carry useful variety.
3. **E10 (keep their pressures, drop their flow) does not get the best of both**: flow from P_ao
   is worst of the three. So the problem with pigs 2 / 3 / 5 is not only their flow target;
   plausibly their pressure–flow *relationship* (through the other signals and the decoder's
   cross-signal reconstruction) or their pressure waveforms themselves differ.
4. Caveats: one seed per configuration, 4 test pigs, and part of E9's flow advantage is already
   present in its untrained baseline (0.30 vs −0.54 for E7): with consistent training pigs the
   normalization and the probe itself transfer better. Differences of this size between single
   runs need seed replicates before they are trusted.

Decision for now: **E9 (paper pigs) for anything flow-related, E7 (all pigs) for pressure**;
confirm with 2–3 seeds per configuration before choosing a single default.

## Seed replicates — E7 (all pigs) vs E9 (paper pigs), seeds 0, 1, 2, test pigs 4, 6, 7, 8

Runs `E7`, `E7s1`, `E7s2` and `E9`, `E9s1`, `E9s2` (`--seed 1/2`), evaluated with
`python -m model.evaluate … --split test`. For each seed: median over the 4 test pigs; table shows
mean ± sd over the 3 seeds (within-pig R²). The untrained column differs per configuration because
the normalization and the probe are fitted on that configuration's training pigs.

| | all pigs, trained | all pigs, untrained | paper pigs, trained | paper pigs, untrained |
|---|---|---|---|---|
| flow from P_ao only | +0.08 ± 0.30 | +0.04 ± 0.50 | **+0.29 ± 0.25** | +0.23 ± 0.13 |
| mean Q_ao (full input) | +0.58 ± 0.12 | **+0.83 ± 0.09** | +0.68 ± 0.06 | +0.68 ± 0.17 |
| flow without Q_ao | +0.16 ± 0.35 | +0.31 ± 0.12 | +0.09 ± 0.08 | +0.15 ± 0.25 |
| mean P_ao | +0.80 ± 0.02 | **+0.88 ± 0.04** | +0.71 ± 0.12 | +0.79 ± 0.07 |
| pulse pressure | +0.65 ± 0.10 | +0.67 ± 0.08 | +0.49 ± 0.12 | +0.06 ± 0.23 |
| heart period | +0.90 ± 0.01 | **+0.97 ± 0.00** | +0.92 ± 0.00 | +0.95 ± 0.01 |

Fraction of (seed, test pig) pairs where the trained encoder beats the untrained one:
flow from P_ao only 0.50 (all pigs) / 0.75 (paper pigs); pulse pressure 0.50 / 0.67;
mean P_ao 0.42 / 0.33; heart period 0.00 / 0.33.

Flow from P_ao only varies enormously between seeds for the same test pig (pig 4, all pigs:
−6.83 to −0.19; pig 8, paper pigs: −1.44 to 0.36).

### Findings — and corrections to earlier conclusions

1. **On held-out test pigs, the trained encoder does not reliably beat an untrained encoder.**
   Apart from pulse pressure in the paper-pigs setup (+0.49 vs +0.06), every difference between
   trained and untrained is within the seed-to-seed spread, and for several targets the untrained
   encoder is *better* (mean P_ao, mean Q_ao, heart period in the all-pigs setup).
2. **Correction to "Status and conclusions (after E8)"**: the claim that training improves mean
   pressure on unseen animals (6/8 test pigs, seed 0) came mostly from test pigs 1, 2, 3, 5, where
   the untrained encoder did poorly. On test pigs 4, 6, 7, 8 with 3 seeds it does not hold.
3. **Correction to "E7 vs E9 vs E10"**: E9's flow-from-P_ao of +0.54 was a favourable seed. Over 3
   seeds it is +0.29 ± 0.25, against +0.23 ± 0.13 untrained. Training on the paper's pigs gives a
   better *level* for flow from pressure than training on all pigs (+0.29 vs +0.08), but most of
   that is already present without training: consistent training pigs make the normalization and
   the probe transfer better.
4. Validation-pig selection (E1–E8) was optimistic: checkpoint choice on one validation pig with
   noisy probes overfits that pig.

### What this means

With 8 animals and 10 epochs, the self-supervised objectives do not yet produce an embedding that
is more *linearly* informative about these window-level quantities than random features of the
same architecture. Plausible reasons:

- **The probe targets are too easy**: window averages of the encoder's own inputs (mean pressure,
  heart period) are preserved by random projections; training can only lose some of it. The spec's
  harder targets (stroke volume, Windkessel R / C / Z_c, LV indices, intervention phases, §15.1)
  are not computed yet and are where a trained encoder should matter.
- **Too few animals**: what the encoder learns is partly specific to the 2–6 training pigs; seed
  noise is as large as the effects.
- **Objective balance**: reconstruction is solved within 2–6 epochs on training pigs; nothing in
  the objectives directly rewards transfer across animals.

Recommended next steps:
1. Add the spec's harder probe targets (§15.1) and the baselines (handcrafted features,
   raw-template PCA, §15.5) before further tuning — without them, the evaluation cannot show
   what training adds.
2. Always report ≥ 3 seeds; select checkpoints on more than one validation pig (or fix the
   epoch budget) to avoid overfitting the selection.
3. For the flow goal specifically: more animals or pretraining on simulated / human
   pressure data (spec §18.3).

## Benchmark — harder targets and baselines (spec §15.1, §15.5)

New code (from `src/`): `python -m model.benchmark <runs> --folds 4 6 7 8` → `runs/benchmark.json`.

- **Targets** (`model/targets.py`, 21 per window, never used in training): heart rate, pulse
  transit time P_ao→P_fem; mean / systolic / diastolic / pulse pressure and pulse-pressure
  variation of P_ao; mean and pulse pressure of P_fem; relative stroke volume and cardiac output;
  2-element Windkessel R, τ (diastolic decay), C = τ/R and characteristic impedance Z_c; relative
  EDV and LV stroke amplitude; max dP_lv/dt, LV end-diastolic pressure; mean P_vc; **PEEP** (from
  airway pressure, which the encoder never sees). Pressures in mmHg; Q_ao / V_lv relative to
  each recording's typical beat (calibration not comparable between pigs). Sanity check on pig 7:
  HR 63 bpm, PTT 95 ms, MAP 42 mmHg, τ 1.4 s, dP/dt_max 484 mmHg/s, EDP 9 mmHg, PEEP 4.5 cmH₂O.
- **Methods**, all on the same normalized windows restricted to the regime's input signals:
  trained encoder (e, 32-d), untrained encoder (same init), **handcrafted** features (per signal:
  mean / min / max / peak-to-peak / max slope averaged over the window, beat-to-beat std of the
  peak-to-peak, mean and std of beat duration; 38 features), **template PCA** (mean beat per
  signal → PCA 32 on the training pigs). Diagnostics: `trained_H` (the encoder's 128-d beat
  tokens averaged — is information lost at the embedding head?) and `handcrafted+trained`
  (does e add anything?).
- **Regimes**: full (6 signals), pressures (P_ao, P_fem, P_lv, P_vc), P_ao only. A target whose
  source is not in the regime is **cross-signal**.
- **Score**: within-pig Pearson r on the test pig (scale-free), ridge probe fitted on training
  pigs. 24 checkpoints: E7 and E9 × seeds 0–2 × test pigs 4, 6, 7, 8.

Median within-pig r (over targets, seeds and test pigs):

| | trained | trained_H | untrained | **handcrafted** | handcrafted + trained | template PCA |
|---|---|---|---|---|---|---|
| all pigs, full | 0.79 | 0.83 | 0.82 | **0.99** | 1.00 | 0.96 |
| all pigs, pressures | 0.71 | 0.75 | 0.75 | **0.90** | 0.89 | 0.82 |
| all pigs, P_ao only | 0.62 | 0.56 | 0.62 | **0.72** | 0.73 | 0.66 |
| paper pigs, full | 0.76 | 0.79 | 0.74 | **0.99** | 1.00 | 0.94 |
| paper pigs, pressures | 0.69 | 0.66 | 0.66 | **0.88** | 0.87 | 0.77 |
| paper pigs, P_ao only | 0.70 | 0.62 | 0.72 | **0.77** | 0.77 | 0.60 |
| **cross-signal**, paper pigs, P_ao only | 0.64 | 0.52 | 0.67 | **0.73** | 0.67 | 0.55 |
| **cross-signal**, paper pigs, pressures | 0.53 | 0.42 | 0.42 | 0.50 | **0.60** | 0.58 |

Trained beats handcrafted in only 11–40 % of (checkpoint, target) comparisons; adding e to the
handcrafted features improves the probe (> 0.01 in r) in only 22–40 % of cases.

Relative stroke volume / cardiac output from **P_ao only**, paper pigs: r ≈ 0.81–0.85 for trained,
untrained and handcrafted alike — within a new animal, changes of flow are trackable from aortic
pressure (consistent with pulse-contour methods), but the learned embedding is not better at it
than simple beat statistics.

### Findings

1. **The handcrafted baseline is the bar, and the current encoder does not reach it**, in any
   regime, including the cross-signal targets the spec considers most informative. The one
   exception is the pressures-only cross-signal median for paper pigs, where combining
   handcrafted features with e helps (0.50 → 0.60) — weak, single summary.
2. **Trained ≈ untrained** on these targets as well (fraction trained > untrained 0.32–0.50).
3. **The information is not lost at the embedding head**: the 128-d token outputs (`trained_H`)
   are no better than e. The encoder itself does not extract it.
4. **e is mostly redundant with simple beat statistics**: it rarely adds to them.
5. PEEP (context) is poorly predicted by every method (r 0.15–0.45): little of the ventilator
   setting is visible in the cardiac windows, or PEEP varies too little within most test pigs.

### Interpretation

The self-supervised objectives, as implemented, make the encoder good at its own tasks
(reconstruction of hidden / dropped beats on the training pigs) without making the embedding a
better linear description of the physiology than the waveform statistics it is computed from.
With 2–6 training animals per fold, that is not surprising; the spec lists this risk (§18.3).
Before more tuning, a decision is needed about the direction (see below).

### Options

- **A. Reframe**: report the handcrafted / template baselines as the reference, and use the
  encoder only where it can add something baselines cannot (e.g. missing-signal robustness at
  inference, sequence-level change detection, §15.2–15.3).
- **B. Fix the learning signal** before scaling: e.g. make the template objective (B) predict
  physiological summaries directly (handcrafted-like targets as auxiliary self-supervised targets,
  computed from the full window), so e must carry them; or probe-guided selection on several
  validation pigs.
- **C. More data**: pretraining on simulated cardiovascular data and/or large human pressure
  datasets, then the pigs for adaptation (spec §18.3).

## Option B — summary objective (E11, E12)

Change (not in the spec): a fourth training-only head `SummaryHead` (e → 128 → 38) predicts, from
the embedding alone, the window's summary statistics (`model/features.py`: per signal mean / min /
max / peak-to-peak / max slope averaged over beats + beat-to-beat std of the peak-to-peak; mean and
std of r). Targets are computed from the **full clean window** (before sensor dropout and beat
masking), standardized with the training set's mean / std, and the loss is an MSE over the
signals present in the window:

    L = L_A + β L_B + γ L_C + σ [L_S(a) + L_S(b)]

It is still self-supervised (no labels, only the data). Because of sensor dropout, e must carry the
statistics of signals the view did not see — explicit pressure toward cross-signal information,
and toward exactly the quantities the handcrafted baseline exploits. The handcrafted baseline of
the benchmark now uses the same function, so definitions match. σ = 0 reproduces the earlier runs.

Smoke test (1 epoch, paper pigs, σ = 1, fold 7): L_S ≈ 1.7 at start (same scale as L_A ≈ 1.6).

Runs: E11 (σ = 1) and E12 (σ = 5), paper pigs, seed 0, test pigs 4, 6, 7, 8; benchmarked against
E9 (same setup, σ = 0).

Benchmark, median within-pig r on the test pigs:

| | E9 (σ = 0) | **E11 (σ = 1)** | E12 (σ = 5) | untrained | handcrafted |
|---|---|---|---|---|---|
| full | 0.74 | 0.77 | 0.75 | 0.70 | **0.99** |
| pressures | 0.61 | 0.62 | 0.67 | 0.62 | **0.88** |
| P_ao only | 0.67 | 0.66 | 0.58 | 0.69 | **0.77** |
| cross-signal, P_ao only | 0.53 | 0.62 | 0.47 | 0.66 | **0.73** |

E11 vs E9 is within single-seed noise; E12 hurts the LV targets (full input: 0.77 → 0.38).
Trained beats handcrafted in 12–30 % of comparisons (E9: 17–39 %).

### Diagnostic: does the summary skill transfer to a new pig?

The summary head learns its task: training loss ≈ 0.2 per view (standardized MSE). Running the
last-epoch model's summary head on each window and correlating predicted with actual summaries
(within-pig r, full input):

| | training pigs | **test pig** |
|---|---|---|
| all statistics (median) | 0.91–0.93 | **0.58** |
| beat mean / max / peak-to-peak | 0.94–0.97 | 0.65–0.76 |
| beat min | 0.85–0.88 | 0.44–0.47 |
| beat-duration mean | 0.95 | 0.92–0.94 |

**The encoder learns the summaries on its training animals but does not transfer them to a new
animal** — not even "mean of the input beats", which it sees directly (0.65–0.69 on test pigs).
Only beat duration, which enters through a separate linear path, transfers. With 2 training pigs,
a new pig's waveforms (pressure levels, beat shapes) lie outside the training distribution and the
network does not extrapolate; handcrafted statistics compute the same quantities exactly for any
animal. This is a cross-animal generalization limit, not a readout or objective problem.

### Conclusion of option B

Changing the objective does not fix it: the bottleneck is how far a new animal is from the few
training animals. Options that address that directly:

- **B2 — per-recording normalization of all signals** (as already done for Q_ao / V_lv): every pig's
  signals centred / scaled by its own robust statistics, so a new pig is in-distribution by
  construction. The absolute levels (e.g. mean pressure) are lost from the waveform input and
  would have to be passed separately (e.g. as per-recording scalars) if needed. Cheap to test.
- **C — more animals / simulated or human data** (spec §18.3): widen the training distribution.
- **A — reframe** around what the baselines cannot do (missing-signal robustness, temporal
  change detection).

## Option B2 — per-recording normalization (E13, E14, E15)

Change (not in the spec): `--per-recording` (`data/normalize.py`, `Normalizer.per_recording`).
Each recording's signals are centred and scaled by that recording's own median and IQR/1.35
instead of the training pigs' statistics. Beat durations stay globally normalized; evaluation
targets are unchanged (computed in physical units by `model/targets.py`). Check on fold 7 (paper
pigs): with global normalization the test pig's medians sit far from the training pigs (P_fem
−1.33, P_vc +2.35 normalized units); with per-recording normalization every pig is centred at 0.

Trade-off: between-animal level differences (e.g. one pig's higher mean pressure) are removed from
the waveform input; within-recording changes are kept.

Runs (seed 0, test pigs 4, 6, 7, 8): E13 paper pigs; E14 all pigs; E15 paper pigs + summary
objective σ = 1 (to repeat the transfer diagnostic). All 12 runs completed.

**Transfer diagnostic** (summary head, predicted vs actual window summaries, within-pig r):

| | E11 global, test pig | **E15 per-recording, test pig** | training pigs (both) |
|---|---|---|---|
| all statistics (median) | 0.58 | **0.70** | 0.91–0.92 |
| beat mean | 0.66 | **0.86** | 0.95–0.96 |
| beat max | 0.72 | **0.88** | 0.96 |
| peak-to-peak | 0.69 | **0.80** | 0.94 |
| beat min | 0.48 | 0.47 | 0.87–0.89 |

Per-recording normalization **does reduce the cross-animal gap** for what the encoder learns
(0.58 → 0.70 overall, mean / max / peak-to-peak +0.11 to +0.20), confirming the diagnosis: part
of the problem was the new animal being out of distribution.

**Benchmark** (median within-pig r on the test pigs, seed 0):

| | paper global (E9) | paper per-rec (E13) | paper per-rec + S (E15) | all global (E7) | all per-rec (E14) | handcrafted (range) |
|---|---|---|---|---|---|---|
| full | 0.74 | 0.74 | 0.76 | 0.78 | 0.69 | 0.90–0.99 |
| pressures | 0.61 | 0.58 | 0.65 | 0.71 | 0.60 | 0.83–0.90 |
| P_ao only | 0.67 | 0.61 | 0.65 | 0.54 | 0.64 | 0.72–0.77 |
| cross-signal, P_ao only | 0.53 | 0.53 | 0.58 | 0.40 | 0.61 | 0.62–0.73 |

Trained > handcrafted in 11–35 % of comparisons; trained > untrained in 42–57 %. All differences
between the trained configurations are within single-seed noise.

### Conclusion of option B2

Per-recording normalization makes the learned summaries transfer better, but **the probed
embedding still does not beat simple beat statistics**, and still does not clearly beat an
untrained encoder. With 8 animals, neither the objective (B) nor the input distribution (B2)
is enough for the learned representation to outperform explicit features on these targets.
Remaining directions are the ones that change the setting rather than the training details:
**C** (more / simulated / human data) or **A** (use the encoder where baselines cannot compete).

## Option A1 — robustness to missing signals (spec §15.3)

`python -m model.robustness <runs> --folds 4 6 7 8` (`model/robustness.py`) → `runs/robustness.json`.
Scenario: a probe fitted once on full-input training windows is applied, without refitting, to a
new pig where only a subset of signals exists. 24 checkpoints (E7 / E9 × seeds 0–2 × test pigs
4, 6, 7, 8); within-pig r, median over pigs × targets, mean ± sd over seeds. Subsets from the spec.

**Consistency** (r between full-input and subset representation of the same window):

| | trained | untrained | handcrafted (median-filled) | handcrafted + regression imputation |
|---|---|---|---|---|
| 6 subsets, both pig sets | **0.88–1.00** | 0.13–0.85 | −0.01–0.39 | 0.41–0.85 |

**Transfer** (all pigs; paper pigs similar):

| subset | trained | untrained | handcrafted, median-filled | **handcrafted, regression-imputed** |
|---|---|---|---|---|
| P_fem + P_vc | 0.59 ± 0.01 | 0.28 | 0.20 | **0.68** |
| P_fem only | 0.49 ± 0.06 | 0.27 | 0.18 | **0.74** |
| P_lv + V_lv | 0.66 ± 0.03 | 0.38 | 0.53 | **0.77** |
| P_ao only | 0.51 ± 0.20 | 0.55 | 0.52 | **0.71** |
| pressures | 0.71 ± 0.02 | 0.69 | 0.88 | **0.91** |
| P_ao + Q_ao | 0.68 ± 0.03 | 0.69 | 0.78 | **0.82** |

Findings:
1. **Training produces a genuinely sensor-invariant representation** (consistency 0.88–1.00,
   spec property 3), which neither the untrained encoder nor the baselines have, and it loses
   less when signals are removed (median drop full → subset 0.07–0.11 vs 0.21–0.22 for
   median-filled handcrafted features). Against naive filling of missing features it wins clearly
   when P_ao is absent.
2. **But the classical pipeline — predict the missing handcrafted features from the available ones
   (ridge regression on the training pigs), then apply the full probe — beats it on every subset**
   (paired difference −0.08 to −0.19; trained better in only 16–33 % of comparisons).

## Option A2 — detecting a change of state (spec §15.2)

`python -m model.tracking <runs>` (`model/tracking.py`) → `runs/tracking.json`. The only
intervention with recoverable timing is the PEEP recruitment manoeuvre (from airway pressure,
never an encoder input): 4–8 % of windows per pig have PEEP ≥ 7 cmH₂O (baseline ≈ 4). Class-balanced
logistic regression fitted on the training pigs, ROC AUC on the test pig; 24 checkpoints.

| AUC (median over pigs, mean ± sd over seeds) | trained | untrained | **handcrafted** | template PCA |
|---|---|---|---|---|
| all pigs, full | 0.69 ± 0.13 | 0.86 ± 0.01 | 0.85 | **0.91** |
| all pigs, P_ao only | 0.63 ± 0.26 | 0.69 ± 0.10 | **0.95** | 0.83 |
| paper pigs, full | 0.65 ± 0.13 | 0.77 ± 0.09 | 0.84 | **0.93** |
| paper pigs, P_ao only | 0.77 ± 0.04 | 0.68 ± 0.05 | **0.87** | 0.54 |

Findings: the manoeuvres are clearly visible in the cardiac waveforms (handcrafted / template PCA
AUC 0.83–0.95), but **the trained encoder is worse than the untrained one** (e.g. 0.69 vs 0.86, all
pigs, full input): training *removes* this information. Plausible cause: the two-view objective
asks windows shifted by up to 4 beats (≈ one breath) to have the same embedding, which teaches
invariance to respiratory-cycle changes — precisely what a PEEP manoeuvre alters.

---

# Overall conclusion (as of 2026-10-13)

Question: with the 8 porcine recordings, does the self-supervised encoder of the spec produce a
representation that is more useful than classical waveform features?

**Answer: no, on every test we could construct.** Summary of the evidence (all on held-out test
pigs, within-pig scores, ≥ 3 seeds where it matters):

| test | best method | trained encoder vs it |
|---|---|---|
| 21 physiological targets, any input subset (§15.1) | handcrafted beat statistics | below (r 0.62–0.79 vs 0.72–0.99) |
| cross-signal targets (§15.1, reduced input) | handcrafted / template PCA | below |
| trained vs untrained encoder | — | no consistent difference (seed noise) |
| missing signals without refitting (§15.3) | handcrafted + regression imputation | below (−0.08 to −0.19 r) |
| sensor-invariance of the representation (§15.3) | **trained encoder** | **best** (0.88–1.00) |
| detecting PEEP manoeuvres (§15.2) | handcrafted / template PCA | below, and below untrained |

What was tried to fix it, and why it did not: data cleaning and calibration removal (E1, fixed
overfitting), objective weights (E2–E8), pig selection (E9–E10), seed replication, a summary
objective forcing the embedding to carry the statistics (E11–E12: learned on training pigs,
r 0.92, not transferred to new pigs, r 0.58), per-recording normalization to make new pigs
in-distribution (E13–E15: transfer 0.70, still below baselines). The consistent diagnosis is a
**cross-animal generalization limit**: with 2–6 training animals per fold, what the encoder learns
does not extrapolate to a new animal as well as explicitly computed features do, and its invariance
objectives remove some state information.

Solid by-products, usable regardless of the encoder:
- the preprocessing pipeline (Steps 1–9, `src/preprocessing`, `src/data`) and its quality control;
- the evaluation suite: targets (`model/targets.py`), benchmark, robustness and tracking scripts;
- the finding that **within a new animal, relative stroke volume and cardiac output are tracked
  from aortic pressure alone at r ≈ 0.8** with simple features, and that recruitment manoeuvres
  are detectable from aortic pressure alone (AUC up to 0.95);
- the evidence that the paper's excluded pigs 2 / 3 behave differently for flow.

Recommended next step: **option C (more / different data)** is the only lever left that addresses
the diagnosis — e.g. pretraining on simulated beats from a lumped cardiovascular model spanning
many virtual "animals", or on large human arterial-pressure datasets, then evaluating with the
suite above, with the handcrafted / regression-imputation baselines as the bar to beat. If the
thesis must stay within this dataset, the defensible framing is A: classical features as the
method, and the encoder study as a documented negative result explaining why (small-n
cross-animal generalization, invariance objectives removing state information).

## Within-pig split — probe fitted and tested on the same pig

`python -m model.withinpig E7` (`model/withinpig.py`) → `runs/withinpig.json`. For each of the 8
pigs, the probe is fitted on part of that pig's recording and tested on another part of the same
pig; encoders are not retrained (the trained encoder is E7's fold where the pig was the test pig,
so it never saw that pig). Splits: **blocked** (5 contiguous blocks, each tested once, with a
gap of one window length) and **chronological** (first 60 % → last 40 %; the later phases are
unseen). Compared with the cross-pig setting (probe fitted on the other pigs; E7 benchmark on all
8 folds). Seed 0.

Median Pearson r over the 21 targets and 8 pigs:

| | cross-pig | within-pig, blocked | within-pig, chronological |
|---|---|---|---|
| handcrafted, P_ao only | 0.71 | 0.82 | **0.84** |
| template PCA, P_ao only | 0.63 | 0.76 | 0.72 |
| trained encoder, P_ao only | 0.52 | 0.54 | 0.40 |
| untrained encoder, P_ao only | 0.55 | 0.61 | 0.48 |
| handcrafted, full input | 0.99 | 1.00 | 1.00 |
| trained encoder, full input | 0.76 | 0.69 | 0.58 |

Within-pig absolute R² (median): handcrafted 0.99 (full) / 0.59 blocked, 0.18 chronological
(P_ao only); trained encoder negative everywhere (−0.6 to −4.3).

Key targets from P_ao only, Pearson r (cross-pig → blocked → chronological):
- handcrafted: relative CO 0.68 → 0.78 → 0.84; relative SV 0.69 → 0.68 → 0.79; max dP_lv/dt
  0.84 → 0.85 → 0.87; Windkessel τ 0.69 → 0.80 → 0.86; PEEP ≈ 0.3 throughout;
- trained encoder: relative CO 0.23 → 0.59 → 0.51; pulse pressure 0.82 → 0.78 → 0.22.

Findings:
1. **Calibrating on the same animal helps the simple features**: from aortic pressure alone, the
   median r rises from 0.71 (other pigs) to 0.82–0.84 (same pig), and relative cardiac output
   reaches r = 0.84 even when predicting the later, unseen phases of the experiment. The
   correlation transfers to new phases better than the absolute level (R² 0.18 chronological):
   the trend is right, the offset drifts.
2. **The encoder does not benefit**: within-pig r is no better than cross-pig (0.54 vs 0.52) and
   worse chronologically (0.40), with negative R². Its embedding is neither linearly informative
   nor stable over time within one animal — consistent with the invariance objectives removing
   slow state changes (A2).
3. So the cross-animal gap identified earlier is not the only problem of the encoder; even with a
   per-animal calibration, the learned embedding stays below simple beat statistics.

## Calibrate then track — relative cardiac output from P_ao after calibrating on the same pig

`python -m model.calibrate` (`model/calibrate.py`) → `runs/calibrate.json`,
`figures/results/fig5_calibrate_track.png`. Pigs 2–8 (pig 1 has no usable flow); handcrafted P_ao
beat statistics with a ridge probe; calibration on the first D minutes, tracking on the rest
(after a one-window gap). Methods: **population** (probe from the other pigs, no calibration),
**population + offset** (one-point calibration: mean error on the calibration period removed),
**population + slope & offset** (linear rescaling on the calibration period), **calibrated**
(probe fitted on the calibration period only). Medians over the 7 pigs:

| D [min] | 5 | 10 | 20 | 30 | 45 | 60 |
|---|---|---|---|---|---|---|
| r, population (= population + offset) | **0.84** | **0.85** | **0.86** | **0.86** | **0.86** | **0.85** |
| r, population + slope & offset | 0.68 | 0.15 | 0.70 | 0.71 | 0.69 | 0.74 |
| r, calibrated | 0.34 | 0.42 | 0.46 | 0.64 | 0.76 | 0.80 |
| trend concordance %, population | 88.9 | 88.9 | 88.2 | 88.2 | 88.2 | 88.2 |
| trend concordance %, calibrated | 94.4 | 94.4 | 90.0 | 90.9 | 90.0 | 87.5 |
| RMSE % of mean CO, population | 26.5 | 26.3 | 26.4 | 26.6 | 25.3 | 26.5 |
| RMSE % of mean CO, population + offset | 32.1 | 32.1 | 31.8 | 30.3 | 33.9 | 37.6 |

Trend concordance: % of consecutive 5-min changes with the same sign in prediction and
reference, excluding changes < 5 % (median 16 scored changes per pig).

Findings:
1. **The population probe tracks relative cardiac output from aortic pressure alone without any
   per-pig calibration**: r ≈ 0.85 and ≈ 88 % trend concordance over the tracking period, stable
   for every D. This is the main positive result of the project.
2. **Calibrating on the start of the recording does not help, and fitting a slope there hurts**:
   the first minutes are a stable baseline where cardiac output barely varies, so a slope (or a
   whole probe) fitted there is noise — some pigs even get a negative slope (r 0.15 at D = 10).
   A pig-specific probe needs ≥ 45–60 min of calibration data (covering at least one recruitment
   manoeuvre) to approach the population probe (r 0.76–0.80).
3. **Offset calibration does not improve the level either**, because "relative CO" is defined
   relative to the whole recording's typical flow (to remove flow-probe calibration), so the
   population probe is already near the right level, while an offset from the baseline carries the
   baseline level into the later, lower-output phases. Absolute agreement in L/min cannot be
   evaluated across pigs with these flow probes; the meaningful metrics here are trend metrics
   (r, concordance).
4. **The prediction compresses large changes** (bottom panels): the direction and timing of every
   change are right, but deep drops (terminal decline of pigs 4 and 6, actual → ~0) are
   under-predicted (prediction ~10–12). Plausible causes: ridge shrinkage and the compliant,
   non-linear pressure–flow relation at very low output.

Interpretation for the thesis: a **population pulse-contour-like model of aortic pressure features
tracks within-animal cardiac-output trends in unseen pigs** (r ≈ 0.85, concordance ≈ 88 %), with
no calibration needed for trending; short calibration periods are counter-productive unless they
contain haemodynamic variation.

# Data scope (decision 2026-10-16): pigs 4, 6, 7, 8 only

From now on the project uses only the four pigs of Cushway et al. 2024. Pigs 1, 2, 3, 5 are
excluded for good, from preprocessing onwards (`preprocessing/config.py`: `PIGS`,
`EXCLUDED_PIGS` with the reasons; raw `.mat` files kept on disk but unused). `pigs.h5` was rebuilt
with the four pigs and `data.store.load_recordings()` defaults to them. All results above that use
"all pigs" are historical.

Calibrate-then-track rerun on the four pigs (population probe now from the 3 other pigs):

| median r over pigs 4, 6, 7, 8 | D = 5 | 10 | 20 | 30 | 45 | 60 min |
|---|---|---|---|---|---|---|
| population (= population + offset) | 0.70 | 0.73 | 0.79 | 0.82 | 0.82 | 0.81 |
| calibrated (probe fitted on the first D min) | −0.03 | 0.40 | 0.43 | 0.75 | **0.88** | 0.85 |
| population + slope & offset | −0.24 | 0.45 | −0.33 | 0.46 | 0.82 | 0.81 |

Trend concordance (population) 79–84 %; calibrated 89–93 % for D ≥ 30 min. Per pig, population r:
pig 4 0.83, pig 6 0.86, pig 7 **0.13**, pig 8 0.78 — with only three training animals, pig 7 is
not covered by the population model, while a per-pig probe calibrated on ≥ 45 min (including a
recruitment manoeuvre) recovers r ≈ 0.85–0.88.

## All analyses on pigs 4, 6, 7, 8 only (2026-10-16)

Encoders: E9, E9s1, E9s2 (trained on the four pigs, leave-one-pig-out, seeds 0–2; unaffected by
the rebuild — same data for these pigs). Figures 1–5 in `figures/results/` regenerated from these
runs only (`python -m model.figures`, `python -m model.calibrate`); within-pig analysis rerun with
the E9 encoders (`python -m model.withinpig E9`).

| test (unseen pig unless stated) | trained | untrained | handcrafted | template PCA | handcrafted + imputation |
|---|---|---|---|---|---|
| 21 targets, full input (median r) | 0.76 | 0.74 | **0.99** | 0.94 | – |
| 21 targets, pressures only | 0.69 | 0.66 | **0.88** | 0.77 | – |
| 21 targets, P_ao only | 0.70 | 0.72 | **0.77** | 0.60 | – |
| missing signals, transfer, P_ao only | 0.34 | 0.44 | 0.45 | 0.59 | **0.73** |
| missing signals, transfer, P_fem only | 0.36 | 0.38 | 0.18 | 0.18 | **0.71** |
| missing signals, transfer, P_lv + V_lv | 0.62 | 0.39 | 0.59 | 0.28 | **0.77** |
| PEEP manoeuvre AUC, full input | 0.69 | 0.74 | 0.84 | **0.93** | – |
| PEEP manoeuvre AUC, P_ao only | 0.79 | 0.66 | **0.87** | 0.54 | – |
| same pig, chronological split, P_ao only (r) | 0.47 | 0.48 | **0.85** | 0.64 | – |

Conclusions unchanged with the four pigs: handcrafted beat statistics (with regression
imputation for missing signals) are the best representation on every test; the trained encoder is
not reliably better than an untrained one. Within-pig, the handcrafted probe reaches r = 0.85 on
the later, unseen phases from P_ao alone; the encoders do not (0.47).

## Capacity sweep — does a bigger encoder help? (C1–C3, pigs 4, 6, 7, 8)

Change: encoder architecture configurable (`--beat-dim --d-model --n-layers --n-heads --d-ff
--dropout --d-embed`, `model/encoder.py:config_from_args`); stored in each checkpoint's args, so
all evaluation scripts rebuild the right network. Everything else = E9 (scaled Q_ao / V_lv,
γ 0.1, keep 0.7, σ 0, 10 epochs, leave-one-pig-out over pigs 4, 6, 7, 8, seeds 0–2).

| run | architecture | encoder parameters |
|---|---|---|
| E9 (baseline) | spec: D 128, 4 layers, 4 heads, FF 512, C 64, d_e 32 | 1.10 M |
| C1 deep | 8 layers | 1.89 M |
| C2 wide | D 256, 8 heads, FF 1024 | 3.52 M |
| C3 big | D 256, 8 layers, 8 heads, FF 1024, C 128, d_e 64 | 6.98 M |

36 runs, all completed; benchmark, robustness and tracking evaluated as for E9. Values: median over
targets (or subsets) and test pigs, mean ± sd over 3 seeds.

| trained encoder | base (E9) | deep (C1) | wide (C2) | big (C3) | handcrafted |
|---|---|---|---|---|---|
| 21 targets, full input | **0.77 ± 0.04** | 0.73 ± 0.07 | 0.74 ± 0.03 | 0.75 ± 0.04 | 0.99 |
| 21 targets, pressures | 0.69 ± 0.07 | 0.65 ± 0.05 | 0.67 ± 0.03 | **0.70 ± 0.07** | 0.88 |
| 21 targets, P_ao only | **0.69 ± 0.03** | 0.67 ± 0.11 | 0.65 ± 0.05 | 0.59 ± 0.04 | 0.77 |
| cross-signal, P_ao only | **0.62 ± 0.08** | 0.59 ± 0.18 | 0.61 ± 0.04 | 0.52 ± 0.07 | 0.73 |
| cross-signal, pressures | **0.58 ± 0.10** | 0.48 ± 0.17 | 0.50 ± 0.05 | 0.48 ± 0.11 | 0.50 |
| missing signals, transfer, P_ao only | 0.35 ± 0.06 | **0.51 ± 0.13** | 0.48 ± 0.06 | 0.20 ± 0.07 | 0.73 (imputed) |
| missing signals, transfer, P_lv + V_lv | 0.59 ± 0.09 | 0.53 ± 0.09 | **0.63 ± 0.01** | 0.53 ± 0.06 | 0.77 (imputed) |
| PEEP manoeuvre AUC, P_ao only | 0.77 ± 0.04 | 0.76 ± 0.03 | **0.78 ± 0.11** | 0.70 ± 0.09 | 0.87 |

Reconstruction loss (objective A), median over runs: training 0.25–0.27 for all sizes, validation
1.26–1.29 for all sizes (≈ 5× higher).

Findings:
1. **More capacity does not improve anything consistently.** Differences between base, deep and
   wide are within the seed spread; the biggest model (C3, 7 M parameters) is the worst on most
   tests (e.g. P_ao only 0.59 vs 0.69, missing-signal transfer from P_ao 0.20 vs 0.35).
2. **The gap is not a capacity problem**: every size fits the training pigs equally well
   (reconstruction 0.25) and generalizes equally badly to a new pig (1.27, five times worse).
   With 2 training pigs per fold, extra parameters can only memorize the training animals.
3. All sizes stay below the handcrafted baselines on every test.

Decision: keep the spec architecture (base); capacity is not the lever. Consistent with the overall
diagnosis — the limit is the number of animals, not the network.

## Mixed-pig training — all four pigs in training, split in time (M1, M2)

Question: does the encoder do better when every pig is seen during training (no animal held out)?

Change: `--split interleaved | chrono` (`data/folds.py: make_time_fold`, `build_fold`). Each pig's
recording is split in time, identically for all four pigs:
- **interleaved**: 20 equal blocks; blocks 2, 7, 12, 17 test, 4 and 14 validation, the other 14
  training — test windows come from the same experimental phases as training (interpolation);
- **chronological**: first 70 % training, next 10 % validation, last 20 % test — the late phases
  are never seen (extrapolation in time).

Windows are kept only if their beats, plus a margin of one window + the maximum view shift
(36 beats), lie entirely in their split: 0 beats shared between training (incl. shifted views) and
test windows (checked). The normalizer is fitted on training beats only. Probes are fitted on the
training windows of all pigs and scored on the test windows, still **within pig** (correlation
after centring each pig separately; `model/monitor.center_by`). Checkpoint selection uses the
validation windows. Windows: interleaved 4091 / 553 / 1080, chrono 4636 / 563 / 1034
(train / val / test). Runs M1 (interleaved), M2 (chrono), seeds 0–2; otherwise = E9.

**Reconstruction generalizes now**: validation reconstruction loss 0.31 (interleaved) and 0.43
(chrono) vs 1.29 with a held-out pig (training loss ≈ 0.22–0.25 in all cases). Seeing the animal
removes most of the cross-animal gap of the reconstruction objective.

Benchmark (median within-pig r over the 21 targets and test windows; mean ± sd over 3 seeds):

| split, input | trained | untrained | handcrafted | template PCA |
|---|---|---|---|---|
| interleaved, full | 0.94 ± 0.02 | 0.94 ± 0.01 | **1.00** | 0.99 |
| interleaved, pressures | 0.88 ± 0.02 | 0.87 ± 0.01 | 0.92 | **0.95** |
| interleaved, P_ao only | 0.85 ± 0.03 | 0.85 ± 0.02 | 0.67 | **0.88** |
| interleaved, cross-signal, P_ao only | 0.83 ± 0.02 | 0.82 ± 0.01 | 0.66 | **0.88** |
| interleaved, cross-signal, pressures | 0.77 ± 0.03 | 0.68 ± 0.04 | 0.71 | **0.80** |
| chrono, full | 0.55 ± 0.10 | 0.54 ± 0.15 | **0.99** | 0.96 |
| chrono, P_ao only | **0.61 ± 0.08** | 0.55 ± 0.29 | 0.57 | 0.58 |
| chrono, cross-signal, P_ao only | 0.19 ± 0.17 | 0.39 ± 0.19 | 0.32 | **0.54** |

Missing signals (transfer, no refitting), interleaved: trained 0.39–0.83, clearly above untrained
on P_lv + V_lv (0.67 vs 0.39) and pressures (0.83 vs 0.69), but handcrafted + regression
imputation is best on every subset (0.64–0.92). PEEP-manoeuvre AUC: handcrafted best in both
splits (0.83–1.00); trained 0.62–0.81.

Findings:
1. **Seeing every animal fixes the reconstruction gap** (validation loss 4× lower than with a
   held-out pig), confirming that the earlier failure was cross-animal generalization.
2. **In interpolation (interleaved), the encoder becomes competitive from P_ao alone**: 0.85 overall
   and 0.83 on cross-signal targets, well above the handcrafted statistics (0.66–0.67). But the
   **untrained encoder gets the same** (0.82–0.85), and **the mean-beat PCA is better** (0.88):
   richer waveform features beat summary statistics once the animals are known, yet the
   self-supervised training still adds nothing over random features or a linear PCA of the beat.
3. **The only places where training helps** are cross-signal from the four pressures (0.77 vs 0.68
   untrained) and missing-signal transfer — still below PCA or regression imputation.
4. **Extrapolating to later phases (chrono) remains hard for everything** that is not trivially in
   the input; there, the trained encoder is marginally best from P_ao only (0.61 vs 0.55–0.58) but
   much worse on cross-signal targets (0.19 vs 0.54 for PCA).

Conclusion: with every pig in training, the encoder no longer fails because of unseen animals, but
it still does not beat a linear mean-beat PCA. The limit now is what the self-supervised
objectives learn, not generalization to new animals.

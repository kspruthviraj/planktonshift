# PlanktonShift — Senior-Reviewer Audit, Gap Analysis & Roadmap

**Role:** Senior Reviewer + Computational Ecologist + Computer Vision Researcher
**Target venues:** Methods in Ecology and Evolution (primary), Communications Biology, Nature Communications, ES&T
**Core claim under test:** *Dataset shift in ecological imaging is largely explained by separable frequency-domain components. Biological morphology and instrument signatures occupy different parts of the frequency spectrum. Exploiting this structure produces more robust, reproducible, and ecologically faithful monitoring systems.*

This document is a reasoning log + audit. It was written *before* proposing any code changes, by reading every major script, every key result JSON, the manuscript (`paper/paper.tex`), and the planning docs.

---

## PHASE 1 — DEEP CODEBASE & PIPELINE AUDIT

### 1.1 What exists (project map)

- `code/adverserial_net/` — Fourier analysis, SBA module, SBA training, artifacts, Pillow, OOD detection, RAG.
- `code/datashift/` — RAVL/VLM evaluation, multi-domain data prep, ViT baselines.
- `code/*.py` (top level) — Tier-1 causality/pillow/reverse-transfer/bootstrap scripts, per-channel & CenterCrop SBA fine-tuning, Chen replication.
- `code/chen_repo/` — Chen et al.'s cloned training/eval + `utils/create_test_data.py` (the *correct* preprocessing reference).
- `results/` — per-experiment JSONs + checkpoints. `results/tier1/` holds the causality/pillow/reverse-transfer numbers.
- `paper/paper.tex` — current manuscript (~650 lines, compiles).
- Data live in `data/chen_data/` (ZooLake2.0 + OOD1–10) and in **sibling projects outside this repo**: `/home/sreenath/research-space/Adverserial_net/data/...` (DataShift cross-instrument) and `/home/sreenath/research-space/TraitMind/data/...` (WHOI22, ZooScan20). `config.py`'s WHOI/ZooScan paths (`data/whoi22_full/...`) **do not exist** — the real scripts hardcode the external sibling-project paths.

### 1.2 The three preprocessing pipelines in use (CRITICAL)

Chen's reference pipeline (`chen_repo/utils/create_test_data.py:ResizeWithProportions`) = shrink-with-LANCZOS keeping aspect ratio → pad with black to a square (desired_size **128**) → later resize to 224. This is "Proportional Padding."

The codebase actually uses **three different, incompatible preprocessing pipelines** across the experiments that are supposed to form one causal story:

| Pipeline | Where used | Aspect ratio | Pad | Final resize |
|---|---|---|---|---|
| **A. Proportional Padding (Chen-correct)** | `perchannel_sba_finetune.py`, `tier1_pillow_impact.py`, `final_chen_saa.py` | preserved | black pad to 128 | →224 BILINEAR |
| **B. Direct squash resize** | `train_with_saa.py` (the +5.9% cross-instrument result), `tier1_frequency_masking.py`, `frequency_masking_all_combinations.py`, `fourier_shift_analysis.py` | **destroyed** | none | `Resize((224,224))` / `.resize(224)` |
| **C. Resize + CenterCrop** | `centrecrop_sba_finetune.py`, `sba_centrecrop_tta.json` (the 82.99% number) | **destroyed** | crop, no pad | Resize(224)+CenterCrop(224) |

This is the single most damaging finding for the central claim: the **mechanistic/causal experiment** (frequency masking, Pipeline B) and the **Fourier characterization** (Pipeline B, grayscale `.resize(224)`) use a *different* preprocessing than the **headline SBA temporal-OOD result** (Pipeline A, 83.19%) and a *different* one again from the **82.99% "SBA+TTA" number** (Pipeline C). A reviewer can argue the "separable frequency structure" is partly an artifact of aspect-ratio squashing (Pipeline B) vs padding (Pipeline A), which themselves inject different frequency content (black borders add low-frequency DC + ringing).

### 1.3 Experiment audit tables

Each table follows the requested schema.

#### EXP 1 — Fourier shift characterization (`fourier_shift_analysis.py`)
- **Q:** Is cross-instrument shift encoded in the amplitude spectrum, and in which bands?
- **Hypothesis:** Domain info lives in amplitude (not phase); shift concentrates in identifiable bands.
- **Input:** 224×224, **grayscale** (`convert("L")`), `.resize((224,224))` direct (Pipeline B), 40 imgs/class.
- **Preprocessing:** Pipeline B (squash). **Not** proportional padding.
- **Intermediate:** `np.fft.fft2` → `fftshift` → `log1p(|F|)` amplitude; phase **discarded**. Radial average over 50 bins.
- **Model:** LogisticRegression (domain) / LDA (class separability), 5-fold CV on radial amplitude features.
- **Output:** domain label (instrument) + per-band class separability.
- **Metric:** CV accuracy; "shift energy" = Σ(Δ radial mean)².
- **Eco interpretation:** instrument signatures are spectrally localized.
- **Why in paper:** foundational correlational evidence for the core claim.
- **Result on disk:** domain classifier **83.1%** (std 0.053, n=2993); WHOI band0 separability 47.1% vs ~17% higher bands; shift WHOI→ZooLake energy 19.18.
- **Redundancy:** partially overlaps Exp 2 (masking) which re-derives domain vs species by band.
- **Hidden issues:** (a) Grayscale collapses colour — but ZooLake is RGB field camera; colour itself is a domain cue being discarded. (b) Bands defined as 5 equal radial bins (band_size = max_r//5 ≈ 22). (c) "Most shifted band = band 1 (bins 22–44)" but band 0 of WHOI→ZooScan actually has *higher* mean_abs_diff (0.352) than band 1 (0.369) — nearly tied; the "mid-frequency" emphasis is a soft call.

#### EXP 2 — Frequency masking causality (`tier1_frequency_masking.py`, `frequency_masking_all_combinations.py`)
- **Q:** Is the frequency separation *causal* (does training on a band force the model to use only that information)?
- **Hypothesis:** Low-only → preserves species, reduces domain; Mid-only → maximizes domain, degrades species.
- **Input:** 224×224 RGB → **grayscale mean** → bandpass → replicated to 3 channels. Pipeline B (`resize(224,224) LANCZOS`).
- **Preprocessing:** Pipeline B. **Inconsistent with Pipeline A used for the headline SBA result.**
- **Intermediate:** FFT bandpass (hard 0/1 annular mask), inverse FFT, **min-max normalized per image** (rescales intensity → injects amplitude/contrast changes unrelated to band selection), replicated to 3ch.
- **Model:** `timm vit_base_patch16_224`, 15 epochs, AdamW lr 1e-4, wd 0.01.
- **Output:** species accuracy (on ZooScan) + "domain accuracy."
- **Metric:** species accuracy; domain acc = LogisticRegression 5-fold CV on **CLS features** of the trained model, labels = IFCB(train) vs ZooScan(test).
- **Eco interpretation:** low freq = morphology; mid freq = instrument.
- **Why in paper:** turns correlation (Exp 1) into causation — the mechanistic core.
- **Result on disk:** low species 43.8% / domain 78.6%; mid species 28.5% / domain **92.6%**; high 24.8% / 90.8%; all 46.0% / 80.7%. n_test=**137** (not 293).
- **Redundancy:** with Exp 1 — both answer "which band carries what"; masking is the stronger (causal) version. Exp 1 could be demoted to motivation.
- **Hidden issues (severe):**
  1. **Band-definition mismatch with the narrative.** Code uses *fraction-of-Nyquist* annuli: low 0–0.25, mid 0.25–0.75, high 0.75–1.0 → for r_max≈112: low 0–28, mid **28–84**, high 84–112. The paper labels these "Low (bins 0–22), Mid (bins 22–44), High (bins 44+)". **The "mid" masking band (28–84) is ~3× wider and shifted relative to the Fourier "mid" (22–44).** The paper conflates two different band definitions under one "mid-frequency" label.
  2. **Domain-accuracy confound.** The model is *trained on IFCB*; "domain accuracy" then separates IFCB (in-distribution, partly memorized) from ZooScan (OOD). Even with **no masking** ("all"), domain acc = 80.7% — so ~80% of the 92.6% is just ID-vs-OOD separability, not "mid frequencies encode instrument." The masking *accentuates* an existing ID/OOD gap; it does not cleanly isolate "instrument identity." A cleaner design: train domain classifier on **amplitude features of held-out images from both domains** (as in Exp 1), not on a model trained on one domain.
  3. **Test set size mismatch.** Masking uses n_test=137 (classes auto-discovered from IFCB_DIR=16, intersected with ZooScan → fewer). The cross-instrument Table (293) and the masking Table (137) use **different test subsets** but the paper treats them as one benchmark.
  4. **Per-image min-max normalization** after bandpass changes global contrast — a potential confound for the "species degraded" interpretation.

#### EXP 3 — Cross-instrument SBA ablation (`train_with_saa.py`)
- **Q:** Does SBA calibrated to the shift spectrum improve IFCB→ZooScan accuracy?
- **Input:** 224 RGB, ImageNet-normalized, Pipeline B (`Resize((224,224))`), SBA on **grayscale** then back to RGB.
- **Model:** torchvision ViT-B/16 (and ResNet-50, ConvNeXt-Tiny), 30 epochs.
- **Output:** species label.
- **Metric:** ZooScan accuracy, per-class.
- **Result on disk:** standard 46.7% (baseline), heavy 46.0%, saa_noise 49.6%, saa_band 48.2%, saa_amplitude 46.7%, **saa_band+TTA 52.6% (seed 42)**, phase-preserve 40.9%.
- **Why in paper:** demonstrates practical value of exploiting frequency structure.
- **Hidden issues (severe):**
  1. **Seed cherry-picking.** saa_band ZooScan across seeds: 42→52.6, 999→51.8, 789→50.4, 123→49.6, **456→45.3**. Mean ≈ **49.9%** (+3.2pp), range 45.3–52.6. The paper reports the **single best seed (52.6, +5.9pp)** as the headline. The ensemble-seed table in `DEFINITIVE_PLAN.md` itself lists mean 49.9% — but the manuscript uses 52.6%.
  2. **CIs overlap.** Baseline CI [38.7, 54.7] vs saa_band+TTA [44.5, 60.6] — **overlapping**; +5.9pp is **not** statistically distinguishable. No McNemar/paired test is actually computed (methods promise McNemar; `bootstrap_cis.json["cross_domain"]` is empty `{}`).
  3. **Pipeline B vs the headline temporal result's Pipeline A** — different preprocessing again.
  4. "phase-preserving hurts (−5.8%)" is a **single seed, single run** — presented as a robust counterintuitive finding but with no CI/seed variation.

#### EXP 4 — Temporal OOD: SBA fine-tuning of Chen's BEiT (`final_chen_saa.py`, `perchannel_sba_finetune.py`, `centrecrop_sba_finetune.py`)
- **Q:** Does SBA fine-tuning preserve/improve Chen's temporal OOD benchmark (ZooLake2.0→OOD1–10)?
- **Input:** Chen BEiT, raw [0,1] (no ImageNet norm), 35 classes, geometric 3-model ensemble + 4-rotation TTA.
- **Output:** species label; macro-OOD over 10 days.
- **Metric:** macro accuracy vs Chen 83.05%.
- **Results on disk (three different numbers, three pipelines):**
  - `finetune_results_v4.json` (Pipeline A, grayscale SBA, no TTA): baseline geometric **81.7%**, SBA-finetuned ensemble **82.1%**.
  - `perchannel_sba/results.json` (Pipeline A, **per-channel RGB SBA**, +TTA): macro **83.19%** → **beats Chen by +0.14pp**.
  - `sba_centrecrop_tta.json` (Pipeline C, CenterCrop, "OLD SBA v4", +TTA): macro **82.99%**.
  - `centrecrop_sba/results.json` (Pipeline C, CenterCrop, per-channel SBA, +TTA): 82.19%.
- **Why in paper:** anchors SBA to the field's reference benchmark.
- **Hidden issues (severe — manuscript-vs-data mismatch):**
  1. **The paper's headline "82.99% … within 0.06% of Chen" is the CenterCrop (Pipeline C) number** (`sba_centrecrop_tta.json`, config explicitly says `CenterCrop(224)`), yet §3.7 states "These results use Chen's proportional-padding preprocessing pipeline." **Factual contradiction.**
  2. **The paper's actual best result, 83.19% (per-channel, Pipeline A, beats Chen), is not reported anywhere in the manuscript.** The comparison Table (`tab:comparison`) lists only 82.1 and 80.5. The paper is *understating its own strongest result* and *mislabeled the preprocessing* of the number it does report.
  3. Three SBA variants (grayscale/CenterCrop/per-channel) × two pipelines are conflated into one narrative; only the per-channel + proportional-padding combination both (a) matches Chen's correct pipeline and (b) beats Chen — and it is the one omitted.

#### EXP 5 — Pillow library version drift (`pillow_resize_experiment.py`, `tier1_pillow_impact.py`)
- **Q:** Does a software library update (Pillow 6.x→7.0, NEAREST→BICUBIC) change pixels enough to matter?
- **Input:** 112 IFCB images; OOD1–10 for the accuracy follow-up (Pipeline A, proportional padding).
- **Output:** pixel residuals + accuracy under each resize.
- **Result on disk:** mean residual 0.0062, max 0.541, **49.2% pixels** affected; bicubic 83.24% vs nearest 82.19% → **+1.05pp**.
- **Eco interpretation:** silent pipeline drift breaks longitudinal reproducibility.
- **Why in paper:** practical reproducibility warning; supports "instrument/pipeline signatures" half of the claim.
- **Hidden issues:** the +1.05pp is *micro* accuracy computed as `sum(v*1000)/sum(1000)` assuming each OOD day has exactly 1000 images (OOD3 has 522) — a **weighting error**. Also n=112 for the residual stats is small and one dataset only.

#### EXP 6 — Reverse transfer (`tier1_reverse_transfer.py`)
- **Q:** Is SBA direction-agnostic (does it also help ZooScan→IFCB)?
- **Input:** Pipeline B (direct LANCZOS resize), grayscale SBA, 6 shared classes.
- **Result on disk:** ZooScan→IFCB baseline 38.4% → SBA 50.0% (**+11.6pp**); WHOI→ZooScan **not run** (file only contains `zooscan_to_ifcb`).
- **Hidden issues:** 6 classes, small n (~73 test images); single seed; WHOI→ZooScan promised in code but absent from results. The +11.6pp headline rests on a tiny, single-seed run.

#### EXP 7 — RAVL / VLM grounding (`datashift/rag_dataset_shift.py`, `eval_4domain_v2.json`)
- **Q:** Does injecting a Morphological Catalog into a VLM prompt improve cross-system classification?
- **Input:** Qwen2.5-VL-32B via vLLM, greedy decoding, 4 domains.
- **Result on disk:** IFCB-NES 31.2%→88.8% (+57.5pp), ZooScan 29.2%→37.9% (+8.7pp), ZooLake 21.2%→15.4% (−5.8pp), IFCB-WHOI 19.4%→19.4% (0).
- **Hidden issues:** (a) Requires a 32B VLM endpoint — **low reproducibility**; the only on-disk *reproduction attempt* (`ravl_multivlm/ravl_llava_7b.json`) is catastrophic (0% IFCB-NES, 1.5% ZooLake), i.e., the result is model-specific and not reproduced with another VLM. (b) RAVL is **off the frequency-domain thesis** — it never touches FFT; its connection to the core claim is only "complementary robustness," which weakens focus.

#### EXP 8 — Spectral OOD detection (`spectral_ood_detection.py`)
- **Q:** Can radial amplitude features flag OOD images for routing?
- **Result on disk:** ROC-AUC 0.723 (ZooScan), 0.916 (ZooLake2).
- **Hidden issues:** Isolation Forest on amplitude features — directly on-thesis (good). But the "0.72–0.92" range in the abstract combines a weak (0.72) and a strong (0.92) detector; the 0.92 is the easy marine→freshwater case. PR-AUC and threshold operating points not discussed.

#### EXP 9 — Representation analysis (`representation_analysis.py`)
- **Q:** Does SBA improve embedding species-separability?
- **Result on disk (paper Table):** baseline species sep 93.1% → SBA 96.0% (+2.9pp); domain sep ~81%.
- **Hidden issues:** `bootstrap_cis.json["separability"]` is empty — the +2.9pp has **no CI**. Small effect; single seed.

#### EXP 10 — Ecological impact (`ecological_impact.py`)
- **Q:** Does SBA preserve community-composition metrics (Shannon, Simpson, Bray-Curtis)?
- **Result on disk (paper Table):** Bray-Curtis baseline 0.137 → SBA **0.096** → RAG 0.147.
- **Why in paper:** ecological-faithfulness half of the core claim.
- **Hidden issues:** which dataset/OOD day? n unclear; single community; no bootstrap. Strong conceptually but under-specified.

#### EXP 11 — In-domain preservation (WHOI22 99%)
- **Result (paper §3.9):** WHOI22 ViT+SBA 99.0%, ResNet+SBA 99.4%.
- **Hidden issues:** These WHOI22 numbers are **not produced by any script in this repo** (WHOI22 lives in sibling `TraitMind/`; `indomain_5fold_cv.json` is the 16-class IFCB 88.8%, a *different* in-domain result). Provenance unclear / not reproducible from this repo.

#### EXP 12 — Imaging artifact characterization (`imaging_artifact_experiment.py`)
- 15 artifact types; Poisson 0.072, Gamma 0.017, Platform 0.013, Pillow 0.0062.
- Supports the "instrument signatures" framing. Low risk; could move to supplement.

### 1.4 Statistical-rigor audit (cross-cutting)

`tier1_bootstrap_ci.py` is **not a real bootstrap**. For the domain classifier it synthesizes a 0/1 array from `int(acc*n)` "correct" and resamples *that synthetic array* — equivalent to a binomial proportion CI, **discarding the actual 5-fold CV variance**. For cross-domain it hard-codes `n_test=293` and `int(acc*n)`; for OOD AUC it uses a fixed `n=500` "binomial approximation" — **mathematically invalid for AUC** (AUC is rank-based, not a proportion). Consequently `bootstrap_cis.json` has only `domain_classifier` populated; `separability`, `cross_domain`, `ood_detection` are empty `{}`. **Every CI in the paper that is traceable to this script is a synthetic binomial CI, not a bootstrap over real predictions.** No McNemar test is computed despite being promised in §2.8.

### 1.5 End-to-end data flow for the two key results

**83.19% per-channel SBA (the real best, unreported):**
`data/chen_data/ZooLake2/.../ZooLake2.0` → `resize_with_proportions(128)` LANCZOS + black pad → Chen targeted aug → `resize(224,BILINEAR)` → per-channel FFT SBA (spectral_noise+band_adversarial, shift spectrum from `results/adverserial_net/fourier_analysis/cross_domain/fourier_analysis.json`, WHOI↔ZooScan diff) → BEiT raw [0,1] → 3-model geometric ensemble + 4-rot TTA → macro over OOD1–10 = 0.8319.

**Frequency-masking causal table (the mechanistic core):**
`Adverserial_net/data/cross_domain/.../DataShift_IFCB` (sibling repo!) → `resize(224,224) LANCZOS` (squash) → grayscale mean → annular bandpass (fraction-of-Nyquist, **not** the 22/44 bins) → inverse FFT → per-image min-max → replicate 3ch → ViT-B/16 15ep → ZooScan species acc + LogisticRegression on CLS (IFCB vs ZooScan) "domain acc."

**These two pipelines share almost no preprocessing.** The causal proof (B) and the best SBA result (A) are not run on comparable data treatments.

---

## PHASE 2 — NARRATIVE & STORY GAP ANALYSIS (Reviewer perspective)

### 2.1 Dependency graph toward the core claim

```
Core claim: shift ≈ separable freq components (bio in low-freq, instrument in mid-freq) → exploit for robust monitoring
 │
 ├─[Correlational] Exp1 Fourier: 83.1% domain from amplitude; band0 class-sep 47%  (Pipeline B, grayscale)
 │     └── WEAK LINK: grayscale discards colour domain cue; bands soft; "mid" emphasis not clean
 │
 ├─[Causal] Exp2 Masking: low→species 43.8%, mid→domain 92.6%  (Pipeline B)
 │     └── WEAK LINK: domain-acc confound (ID-vs-OOD ~80% baseline); band defs ≠ paper's "22-44"; n_test=137≠293; per-img norm confound
 │
 ├─[Mechanism exploit] Exp3 SBA cross-instrument +5.9pp  (Pipeline B)
 │     └── WEAK LINK: best-seed only; CIs overlap; no paired test; phase-preserve result single-seed
 │
 ├─[Benchmark anchor] Exp4 temporal OOD 82.99% (reported) / 83.19% (unreported)  (Pipeline C / A)
 │     └── WEAK LINK: mislabeled preprocessing; best result omitted; 3 pipelines conflated
 │
 ├─[Reproducibility/practical] Exp5 Pillow 49% pixels +1.05pp  (Pipeline A)
 │     └── WEAK LINK: micro-acc weighting bug; n=112; one dataset
 │
 ├─[Generality/direction] Exp6 reverse +11.6pp  (Pipeline B)
 │     └── WEAK LINK: 6 classes, ~73 imgs, single seed; WHOI→ZooScan missing
 │
 ├─[Complementary] Exp7 RAVL +57.5pp  (no FFT)
 │     └── WEAK LINK: off-thesis; 32B endpoint; reproduction failed with Llava-7B
 │
 ├─[Deployment] Exp8 OOD detection 0.72-0.92  (on-thesis)
 │     └── OK but range spans weak→easy; no operating-point analysis
 │
 └─[Ecological faithfulness] Exp10 Bray-Curtis 0.096  (on-thesis)
       └── WEAK LINK: under-specified; no CI; single community
```

### 2.2 Gap classification

| # | Gap | Class | Severity |
|---|---|---|---|
| G1 | Three incompatible preprocessing pipelines across causal + SBA + headline experiments | Reproducibility/practicality + hidden assumption | **Critical** |
| G2 | Masking "domain accuracy" confounded by ID-vs-OOD (model trained on source) | Missing causal/mechanistic rigor | **Critical** |
| G3 | Masking band definitions (28–84) ≠ paper's "22–44"; conflation | Missing rigor / misleading | **High** |
| G4 | SBA +5.9pp is best-seed; CIs overlap; no McNemar/paired test | Missing statistical rigor | **High** |
| G5 | Bootstrap CIs are synthetic binomial, not real resamples; AUC CI invalid | Missing statistical rigor | **High** |
| G6 | Headline 82.99% is CenterCrop but paper says proportional-padding; 83.19% (beats Chen) unreported | Factual error + missed result | **High** |
| G7 | Generality only on plankton (transparent organisms) — no non-plankton test | Missing external validation | **High** |
| G8 | Amplitude-vs-phase explored only via one single-seed "phase-preserve hurts" run | Missing mechanistic depth | **Medium** |
| G9 | No comparison vs generic domain-adaptation baselines (DANN, CORAL, TENT) on the same benchmark | Missing baseline comparison | **Medium** |
| G10 | RAVL is off the frequency thesis and not reproduced across VLMs | Redundant/diluting | **Medium** |
| G11 | WHOI22 99% in-domain numbers lack in-repo provenance | Reproducibility | **Medium** |
| G12 | No "information allocation" visualization showing species vs domain per band on one figure | Missing key visualization | **Medium** |
| G13 | Pillow +1.05pp micro-accuracy weighting bug; small n | Statistical rigor | **Medium** |
| G14 | Ecological metrics (Bray-Curtis) under-specified, no CI, one community | Missing eco interpretation/rigor | **Medium** |
| G15 | External data (WHOI22/ZooScan20/DataShift) outside repo; config.py paths wrong | Reproducibility | **Medium** |
| G16 | Segmentation/detection (RFDETR, SAM) experiments exist in repo but are unused in paper | Redundant/unused code | **Low** |

### 2.3 Specific reviewer questions (mandated)

**Q: Can a reviewer still claim "this only works for plankton because they are transparent"?**
Yes, today. All evidence is plankton-only, and plankton are semi-transparent with high background fraction (which is exactly why Pillow/borders matter). **No non-plankton biological imaging dataset is tested.** This is the single biggest external-validity hole. A reviewer at Communications Biology/Nature Comms will demand it.

**Q: Is amplitude vs phase sufficiently justified?**
No. The only direct probe is the single-seed "phase-preserving hurts −5.8pp." There is no controlled amplitude-swap vs phase-swap vs both experiment, no phase-scrambling control, and the masking experiment operates on amplitude-only bandpass (phase is implicitly altered by annular masking). The claim "amplitude carries morphology" is asserted more than proven.

**Q: Does the paper show why frequency decomposition is superior to generic DA?**
No. FDA is mentioned and dismissed in one sentence ("limited effectiveness… smaller training set"). No DANN/CORAL/TENT/style baseline on the same IFCB→ZooScan benchmark. Without this, "frequency-domain is better than generic adaptation" is unsupported.

**Q: Is the practical value for real monitoring networks strong enough?**
Partially. Pillow reproducibility warning + OOD routing + ecological Bray-Curtis are compelling *concepts*, but: the Pillow accuracy delta has a weighting bug; OOD detection's strong number is the easy case; Bray-Curtis is one community with no CI. The monitoring-network story is the right framing but under-evidenced.

---

## PHASE 3 — STRATEGIC EVALUATION OF NEW DATASETS / EXPERIMENTS

Principle: add only what **meaningfully increases confidence in the central frequency-domain claim**, especially generality (G7) and mechanism (G2/G3/G8). Ranked by scientific value vs effort.

### Tier 1 — Highest priority

**T1-a. Non-plankton generality test of the frequency decomposition (BBBC or pollen microscopy).**
- *Why it strengthens the claim:* directly refutes "only works because plankton are transparent." If low-freq≈species / mid-freq≈instrument-acquisition holds for opaque cells (BBBC021/020 cell-body phenotypes) or pollen grains across acquisition settings, the separability claim generalizes.
- *Reusable code:* `tier1_frequency_masking.py` + `fourier_shift_analysis.py`, wrapped with a configurable loader.
- *Effort:* 1–2 days (download + adapt loader + run masking + Fourier). CPU/GPU-light.
- *Impact:* **High** — converts the core claim from plankton-specific to a general imaging-science principle; the difference between MEE and Communications Biology.
- *Risk:* BBBC has few "instruments"; better to use a dataset with an **acquisition/domain** axis (different microscopes/scanners) or simulate domains (different focus/illumination/resize) — the latter is cheap and on-thesis (it *is* "instrument signature").

**T1-b. PlanktoScope / citizen-science hardware.**
- *Why:* demonstrates accessibility + a genuinely different low-cost instrument signature; strengthens the "multi-instrument monitoring" practical claim.
- *Effort:* 2–3 days (data acquisition/download + alignment).
- *Impact:* Medium-High for the practical narrative; Medium for the mechanistic claim.

### Tier 2 — Useful but secondary

**T2-a. EcoTaxa messy long-tailed export** — tests open-set/long-tail realism (supports the limitations/open-world point). Effort 2 days; impact Medium.

**T2-b. Kaggle NDSB 2015** — classic stress test at scale. Effort 2–3 days; impact Medium (mostly a "scale" tick-box, not mechanistic).

### Tier 3 — Do not add (dilutes focus)
- More architectures, more ensembles, segmentation/detection (RFDETR/SAM) — already unused in repo (G16); remove or move to a companion repo.

### Recommended addition (script written)
`code/experiment_frequency_decomposition_generality.py` — a **single, modular** script that re-runs the *causal frequency-masking* experiment (the mechanistic core) on **any** image directory structured as `{domain}/{class}/*`, using the **Proportional Padding pipeline (A) exactly** and a **cleaned domain-accuracy protocol** (amplitude-feature domain classifier on held-out images from *both* domains, not CLS-of-source-trained-model). It is parameterized to accept BBBC/pollen/PlanktoScope directories with zero code change, and it fixes G2 + G3 (uses the *same* band definition as the Fourier analysis, with band edges configurable and defaulting to the paper's 22/44 bins). It does not touch existing code.

---

## PHASE 4 — PRIORITIZED ROADMAP & RECOMMENDATIONS

Effort: S≈2–4h, M≈1 day, L≈2–3 days. "Confidence" = how much harder the action makes the core claim to refute.

### P0 — Fix correctness/consistency bugs (must-do before submission)

| # | Action | Effort | Confidence↑ | Journal impact |
|---|---|---|---|---|
| P0-1 | **Report the real best result.** Replace the 82.99% (CenterCrop) headline with the **per-channel SBA + proportional-padding 83.19% (beats Chen)**; fix the "proportional-padding" mislabel; update `tab:comparison` and abstract. | S | High | High (turns "within 0.06%" into "exceeds") |
| P0-2 | **Unify preprocessing to Pipeline A (Chen-correct proportional padding)** for Fourier + masking + SBA + temporal. Re-run Exp1/Exp2/Exp3 under Pipeline A so the causal story and the SBA gain share one data treatment. | M–L | Critical | Critical (removes the single biggest reviewer weapon) |
| P0-3 | **Fix the masking band definition** to match the paper's "0–22 / 22–44 / 44+" (configurable edges, default to Fourier's 5-equal-band scheme). Stop conflating fraction-of-Nyquist annuli with bins. | S | High | High |
| P0-4 | **De-confound domain accuracy.** Measure "domain acc" as an amplitude-feature classifier on held-out images from *both* domains (Exp1-style), not CLS features of a source-trained model. Report the *delta* over the no-masking baseline. | S–M | Critical | High |
| P0-5 | **Fix the Pillow micro-accuracy weighting** (use true per-day n, not 1000). Add n and a paired per-image sign test. | S | Medium | Medium |
| P0-6 | **Real statistics.** Replace synthetic binomial CIs with bootstrap over *per-image correctness* (and over CV folds for the domain classifier); compute McNemar for baseline-vs-SBA on the *same* test set; report CIs for phase-preserve, reverse-transfer, representation +2.9pp. | M | High | High (MEE reviewers live here) |
| P0-7 | **Report SBA across all seeds** (mean ± range/CI), not best seed. Reframe "+5.9pp" as ensemble mean +3.2pp with the best-seed as upper bound, or run more seeds for a tighter estimate. | S | High | High |

### P1 — Strengthen the mechanistic proof (core claim)

| # | Action | Effort | Confidence↑ | Journal impact |
|---|---|---|---|---|
| P1-1 | **Controlled amplitude-vs-phase experiment.** Three conditions on the same data: (a) phase-scramble (randomize phase, keep amplitude), (b) amplitude-swap between domains (keep phase), (c) both. Measure species & domain accuracy. This is the rigorous version of the single-seed "phase-preserve hurts." | M | Critical | High (the memorable mechanism) |
| P1-2 | **Add generic DA baselines** on IFCB→ZooScan: DANN/CORAL + a strong generic augmentation (RandAugment) + TENT test-time adaptation. Show SBA is competitive/superior *because* it targets the measured bands. | M | High | High (answers "why not generic DA") |
| P1-3 | **"Information allocation" figure (G12).** One panel: x-axis = radial frequency bin; left axis/bars = domain-classifier accuracy per bin; right axis/line = class-separability per bin. The visual *is* the core claim. Add cross-domain overlay. | S | High | High (the takeaway figure) |

### P2 — Generality & ecological faithfulness

| # | Action | Effort | Confidence↑ | Journal impact |
|---|---|---|---|---|
| P2-1 | **Run `experiment_frequency_decomposition_generality.py` on ≥1 non-plankton dataset** (BBBC cell phenotypes with simulated acquisition domains, or pollen). Report the same low/mid/high table. | M | Critical for CommsBio/NatComms; High for MEE | Decisive for higher-impact venues |
| P2-2 | **Ecological metrics with CIs and ≥2 communities/OOD days** (bootstrap Bray-Curtis; report per-day). Tie explicitly to "ecologically faithful monitoring." | S–M | Medium | Medium-High |
| P2-3 | **RAVL reframe as complementary** (per special-focus instruction): move RAVL to a clearly secondary "complementary VLM backstop" role; add the model-size/cost note (32B vs IsolationForest <1ms) as the *practical* contribution, not a competing accuracy claim. Optionally reproduce with a second VLM so +57.5pp is not Qwen-only. | M | Medium | Medium |

### P3 — Reproducibility & focus

| # | Action | Effort | Confidence↑ | Journal impact |
|---|---|---|---|---|
| P3-1 | **Bundle/ document external data.** Either vendor WHOI22/ZooScan20/DataShift into `data/` or fix `config.py` paths + add a `fetch_data.sh`; remove hardcoded `/home/sreenath/...` absolute paths from Tier-1 scripts. | S–M | n/a (repro) | Required by MEE |
| P3-2 | **Provenance for WHOI22 99%** (G11): add the script that produces it, or remove the claim. | S | Medium | Medium |
| P3-3 | **Consolidate/remove off-thesis experiments.** Move segmentation/detection (RFDETR/SAM) and unused logs out of the paper repo; cut any results section not tied to the frequency claim. | S | n/a (focus) | Improves clarity |

### How each P0/P1 action makes the core claim harder to refute
- P0-2/P0-3/P0-4 remove the "your causal proof uses a different pipeline and a confounded metric" attack.
- P0-1 turns the benchmark anchor from "almost as good as Chen" to "exceeds Chen with a frequency-calibrated method."
- P0-6/P0-7 remove the "single seed, overlapping CIs, fake bootstrap" attack.
- P1-1 makes "amplitude carries morphology, phase doesn't" a *demonstrated* mechanism instead of an assertion.
- P1-2 removes "why not just use generic domain adaptation?"
- P1-3 gives reviewers the one figure that encodes the entire claim.
- P2-1 removes "only works for transparent plankton."

### Venue notes
- **MEE (primary):** P0 + P1-2 + P1-3 + P3 are likely sufficient; P2-1 strongly recommended.
- **Communications Biology:** P2-1 (non-plankton generality) is effectively mandatory; add P1-1.
- **Nature Communications:** needs P2-1 + P1-1 + a second non-plankton domain + stronger ecological validation (P2-2) and a cleaner single-message figure (P1-3).
- **ES&T:** reframe around monitoring-network reproducibility (Pillow + OOD routing + multi-instrument unification); P0-5, P2-2, P3-1 essential.

---

## Summary of the single most important findings
1. **The paper's headline 82.99% is mislabeled** (it is the CenterCrop run) and **the actual best result, 83.19% per-channel SBA on Chen's correct pipeline, beats Chen but is not reported.**
2. **Three incompatible preprocessing pipelines** run through the causal proof, the SBA gain, and the benchmark — the causal story and the gains are not measured on one consistent data treatment.
3. **The causal masking "domain accuracy" is confounded** by ID-vs-OOD (≈80% baseline with no masking) and uses band edges (28–84) that do **not** match the paper's "22–44."
4. **All reported CIs are synthetic binomial proxies**, not real bootstraps; no McNemar test is computed; the SBA +5.9pp is the best of 5 seeds with overlapping CIs.
5. **No non-plankton generality test exists** — the claim's generalizability is asserted, not demonstrated.

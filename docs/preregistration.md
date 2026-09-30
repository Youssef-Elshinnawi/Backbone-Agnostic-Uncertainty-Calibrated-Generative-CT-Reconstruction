# Pre-registration — hypotheses and evaluation protocol

Author: Youssef Elshinnawi
Status: **DRAFT — not yet frozen.** It will be frozen by the git tag `prereg-v1`. After freezing, every change goes in §8 (Deviations) with the date and reason; the original text is never edited silently.

Items marked **[TO CONFIRM]** are choices that must be fixed before tagging.

---

## 1. Research question

When should a generative sparse-view CT reconstruction be trusted? Specifically: is the posterior uncertainty of training-free generative CT samplers calibrated, and does the acquisition geometry explain *where* and *why* it is miscalibrated?

## 2. Definitions

- **Operator.** A: the self-written ray-driven fan-beam projector (Phase 1) that has passed the adjoint test, at 256 × 256 image resolution.
- **Frequency bands.** For a given view set, T(k) is the transfer function of AᵀA: the Fourier transform of its point-spread function, computed locally for fan-beam. Bands b are defined by:
  - radial frequency bins; and
  - a *measured/unmeasured* split by thresholding the noise-normalised transfer function T(k)/σ_noise²(k) **[TO CONFIRM: threshold and number of radial bins]**.
- **Per-band quantities** for test image x, posterior samples {xᵢ}, posterior mean x̂, and projector P_b onto band b:
  - error e_b = ‖P_b(x̂ − x)‖²;
  - predicted variance v_b = (1/(S−1)) Σᵢ ‖P_b(xᵢ − x̂)‖²;
  - spectral calibration ratio r_b = E[e_b] / E[v_b], averaged over test images. Perfect calibration gives r_b = 1 (log r_b = 0).

## 3. Hypotheses

- **H1 (location).** For data-consistent samplers (DDS, the FM sampler), v_b is ordered by T(k): low in measured bands, high in unmeasured bands. **DPS is predicted to violate this**, placing non-trivial variance in measured bands. *H1 is expected to hold nearly by construction for hard-data-consistency samplers; it is reported as a sanity check, not as a contribution.*
- **H2 (magnitude — primary).** On simulated data (prior matched to the test distribution), log r_b is close to 0 in all bands for DDS and the FM sampler. **[TO CONFIRM: tolerance for "close to 0" — proposal: |log r_b| < log 1.5, judged by its 95% patient-level bootstrap CI]**
- **H3 (sim → real attribution — primary).** Moving from the simulated arm to the real arm:
  - forward-model mismatch (e.g. the uncorrected flying focal spot) increases miscalibration |log r_b| *mainly in measured bands*;
  - prior/domain shift increases it *mainly in unmeasured bands*.

  Operationalised as the difference-in-differences of |log r_b| (real − sim) × (measured − unmeasured), with a patient-level bootstrap CI.
- **H4 (backbone invariance).** The qualitative conclusions of H1–H3 are the same for the diffusion and the flow-matching backbone when both use an identical data-consistency step. Disagreement is reported as a finding, not suppressed.

## 4. Experimental design

| Factor | Levels |
|---|---|
| Views (uniform over the full scan) | 18, 30, 60, 90, 120 |
| Data arm | simulated (vendor full-dose images forward-projected with A, plus noise model **[TO CONFIRM]**); real (subsampled measured sinograms after fan-beam rebinning) — same patients in both arms |
| Classical baselines | FBP, SART, Chambolle–Pock TV (hyperparameters tuned on the validation split only) |
| Generative samplers | DDS (main baseline), DPS, flow-matching sampler (candidate: PnP-Flow or FlowDPS, chosen so its DC step matches DDS) **[TO CONFIRM]** |
| Posterior samples per image | **[TO CONFIRM — proposal: S = 32]** |
| Resolution | 256 × 256 |

**Reference images.**
- Simulated arm: the vendor full-dose image (exact ground truth for A).
- Real arm: a full-view reconstruction of the same rebinned data with the same operator. It is stated as an approximate reference, not ground truth.

## 5. Data splits

Patient-level splits only: prior training / validation / conformal calibration / test. Counts are **[TO CONFIRM after data inspection]**. The test split is not looked at before the analysis code is frozen.

## 6. Metrics

**Primary:**
- spectral calibration ratio r_b, per band (H2, H3);
- CRPS / energy score.

**Secondary:**
- empirical coverage vs nominal level for central credible intervals (reliability diagram, miscalibration area);
- UCE / ENCE;
- sparsification curves / AUSE and Spearman(σ, |error|), **inside a body mask** and over the whole image;
- simulation-based calibration rank histograms and TARP (on prior draws);
- conformal risk-control interval coverage and mean width (on the calibration split → test);
- reconstruction quality: PSNR, SSIM, RMSE in HU.

**Statistics:**
- patient-level bootstrap (10,000 resamples) for all CIs;
- multiple seeds for the generative samplers **[TO CONFIRM: number]**;
- every result labelled *single run* or *benchmarked*.

## 7. Known limitations (stated in advance)

- Rebinning (helix2fan) does not correct the flying focal spot, so the real arm has a known forward-model mismatch.
- The real-arm reference is itself a reconstruction, not ground truth.
- 2D fan-beam slices, not full helical 3D.
- Sparse-view is a hypothetical acquisition scenario; no claim is made about any specific device.

## 8. Deviations log

*(empty until `prereg-v1` is tagged)*

# Pre-registration — hypotheses and evaluation protocol

Author: Youssef Elshinnawi
Status: **Frozen at git tag `prereg-v1` (2026-09-30).** Every later change goes in §9 (Deviations) with the date and reason; the original text is never edited silently.

---

## 1. Research question

When should a generative sparse-view CT reconstruction be trusted? Specifically: is the posterior uncertainty of training-free generative CT samplers calibrated, and does the acquisition geometry explain *where* and *why* it is miscalibrated?

## 2. Data

- **Source:** TCIA *LDCT-and-Projection-data* (Mayo Clinic), DICOM-CT-PD raw projections plus vendor reconstructions. Access is under the TCIA Restricted License Agreement. No data or derived images are committed to this repository unless the agreement permits it.
- **Anatomy: head (N-series)** only. Rationale: consistency with the point-of-care / stroke motivation. The skull adds beam-hardening forward-model mismatch, which is a deliberate stress test for H3.
- **Vendors:**
  - Raw-projection evaluation uses **Siemens SOMATOM Definition Flash** head scans only (≈50 patients), to keep one geometry and one rebinning path.
  - **GE LightSpeed VCT** head scans (≈49 patients) contribute **reconstructed images only**, to prior training.
- **Dose:** routine (full) dose projections for both arms. The dataset's 25%-dose variant is not used in the primary analysis.
- **Rebinning:** helical → flat-detector fan-beam (helix2fan / Noo et al.), 2D slices at 256 × 256. Only slices with a full 360° of projection coverage are used.

## 3. Data splits

Patient-level splits only; slices from one patient never cross splits.

| Split | Patients | Used for |
|---|---|---|
| Prior training | 20 Siemens + all GE (≈49) | Training both generative priors (images only) |
| Validation | 4 Siemens | Hyperparameters of all methods (classical and generative), band-definition checks |
| Conformal calibration | 13 Siemens | Fitting conformal / risk-control corrections only |
| Test | 13 Siemens | All reported results |

- **Assignment:** Siemens patients are assigned by a seeded random permutation (seed 20260930). The resulting patient-ID lists are committed to `configs/splits.json` **before any test-split data is inspected**. If the actual Siemens head count differs from 50, the calibration and test splits absorb the difference equally, and the change is logged in §9.
- **Evaluation slices:** 16 slices per calibration/test patient, evenly spaced over the axial range with full 360° coverage. The statistical unit is the **patient**, not the slice.

## 4. Definitions

- **Operator.** A: the self-written ray-driven fan-beam projector (Phase 1) that has passed the adjoint test. Image grid 256 × 256.
- **Noise model.** Poisson transmission noise using the per-projection incident beam profile from DICOM-CT-PD tag (7033,1065) *PhotonStatistics* (post-bowtie). Σ denotes the resulting (diagonal, post-log) sinogram noise covariance.
- **Transfer function.** T_W(k) = [F AᵀΣ⁻¹A Fᴴ]_kk, the Fourier-basis diagonal of the noise-weighted normal operator, estimated by randomized diagonal probing (Hutchinson-type, 256 probes) for each view configuration. For fan-beam this is the spatially averaged transfer function; spatial variation is reported descriptively at five fixed image locations.
- **Frequency bands.**
  - *Measured/unmeasured split (Wiener criterion):* SNR(k) = T_W(k) · S_x(k), where S_x is the mean power spectrum of the prior-training images. Frequency k is **measured** if SNR(k) ≥ 1 and **unmeasured** otherwise. SNR = 1 is where the data and the prior contribute equally, i.e. where a Wiener filter's gain is 0.5.
  - *Radial bins:* 16 equal-width radial bins from 0 to Nyquist, each split into its measured and unmeasured parts.
  - *Robustness:* all primary analyses are repeated with the thresholds SNR ≥ 0.1 and SNR ≥ 10.
- **Per-band quantities** for test image x, posterior samples {xᵢ}ᵢ₌₁ˢ, posterior mean x̂, and Fourier-band projector P_b:
  - error e_b = ‖P_b(x̂ − x)‖²;
  - predicted variance v_b = (1/(S−1)) Σᵢ ‖P_b(xᵢ − x̂)‖²;
  - spectral calibration ratio r_b = mean(e_b) / mean(v_b) over test slices, aggregated per patient before bootstrapping. The ratio compares variances; perfect calibration is r_b = 1. Because x̂ is estimated from finite S, e_b is bias-corrected by subtracting v_b / S.

## 5. Hypotheses

- **H1 (location — sanity check).** For the data-consistent samplers (DDS, FM-DDS), v_b is monotonically ordered by SNR: lower in measured than in unmeasured bands, at every view count. **DPS is predicted to violate this** by putting non-trivial variance into measured bands. *H1 is expected to hold nearly by construction for hard-data-consistency samplers; it is reported as a check, not claimed as a contribution.*
- **H2 (magnitude — primary).** On the simulated arm, DDS and FM-DDS are calibrated in both aggregate bands (measured and unmeasured) at every view count.
  - *Equivalence margin:* predicted std within ±25% of observed error, i.e. r_b ∈ [0.64, 1.5625], or |log r_b| ≤ log 1.5625.
  - *Decision rule:* H2 is supported for a sampler if, for all 2 bands × 5 view counts, the **entire 99% patient-level bootstrap CI** of log r_b lies within the margin. This is an equivalence test, Bonferroni-adjusted over 10 comparisons. "CI contains 0" is explicitly **not** the criterion, because an underpowered study would pass it trivially.
  - Otherwise H2 is not supported for that sampler, and the failing bands and view counts are reported.
- **H3 (sim → real attribution — primary).** Moving from the simulated to the real arm:
  - forward-model mismatch increases |log r_b| mainly in **measured** bands;
  - prior/domain shift increases it mainly in **unmeasured** bands.

  *Statistic:* D = [|log r|_real − |log r|_sim]_measured − [|log r|_real − |log r|_sim]_unmeasured, per view count and sampler.

  *Decision rule:* H3 is supported for a sampler if D > 0 with its 99% patient-level bootstrap CI excluding 0 at ≥ 3 of the 5 view counts. The sign and CI of D are reported at all view counts regardless.
- **H4 (backbone invariance).** The H1–H3 decisions are the same for DDS (diffusion) and FM-DDS (flow matching). Any disagreement is reported as a finding, not suppressed.

## 6. Experimental design

| Factor | Levels |
|---|---|
| Views (uniform over 360°) | 18, 30, 60, 90, 120 |
| Data arm | **Simulated:** vendor full-dose Siemens images forward-projected with A, plus the §4 noise model. **Real:** measured full-dose sinograms after rebinning, subsampled to the view count. Same patients and slices in both arms. |
| Classical baselines | FBP, SART, Chambolle–Pock TV (hyperparameters tuned on the validation split only) |
| Generative samplers | **DDS** (main baseline). **DPS** (soft-guidance contrast). **FM-DDS** (primary flow sampler): the DDS update transplanted to the flow-matching prior, with an identical CG data-consistency step, the clean-image estimate from the learned velocity, and *stochastic* re-noising along the interpolation path. **PnP-Flow** (secondary flow sampler, published method for external comparability). |
| Priors | One diffusion model and one flow-matching model, same U-Net backbone family and training data |
| Posterior samples | S = 32 for all test slices. S = 128 on a fixed subset (first 4 test patients × 4 slices) to evaluate 0.95-level intervals and to report metric stability between S = 32 and S = 128. |
| Resolution | 256 × 256 |

**Reference images.**
- Simulated arm: the vendor full-dose image (exact ground truth for A; it contains vendor noise, which is accepted).
- Real arm: a full-view (all measured projections) reconstruction of the same rebinned data with the same operator. It is an approximate reference, not ground truth, and is stated as such.

## 7. Metrics and statistics

**Primary:**
- spectral calibration ratio r_b (H2, H3);
- CRPS (pixelwise, sample-based) and energy score (multivariate).

**Secondary:**
- empirical coverage vs nominal level for central credible intervals at {0.5, 0.6, 0.7, 0.8, 0.9} (and 0.95 on the S = 128 subset); reliability diagram and miscalibration area;
- UCE / ENCE;
- sparsification curves / AUSE and Spearman(σ, |error|), **inside a head mask** and over the whole image;
- simulation-based calibration rank histograms and TARP, on images drawn from each prior (tests the sampler independently of prior misspecification);
- conformal risk control (patient-level split conformal), target miscoverage α = 0.1. With 13 calibration patients, α below 1/14 ≈ 0.07 is not attainable, so 0.1 is the lowest meaningful target. Report coverage and mean interval width on the test split;
- reconstruction quality: PSNR, SSIM, RMSE in HU.

**Statistics:**
- patient-level bootstrap with 10,000 resamples for all CIs;
- one training run per prior; this is stated as a limitation (single training seed);
- every reported number is labelled *single run* or *benchmarked*.

## 8. Known limitations (stated in advance)

- Rebinning does not correct the Siemens flying focal spot, so the real arm has a known forward-model mismatch (part of what H3 is designed to detect).
- The real-arm reference is a reconstruction, not ground truth.
- The prior-training data mixes vendors (Siemens + GE images), so reconstruction-kernel differences are part of the prior.
- One training seed per prior.
- 2D fan-beam slices, not full helical 3D.
- 13 test patients: CIs will be wide, and equivalence (H2) may fail for power reasons. If so, this is reported as inconclusive rather than as miscalibration.
- Sparse-view is a hypothetical acquisition scenario; no claim is made about any specific device.

## 9. Deviations log

*(none yet)*

# Phase 1 — Literature Review (a) and Prior-Code Audit (b)

*Author: Youssef Elshinnawi. Prepared 2026-09-29. Status labels used throughout:*
- **[verified]** — paper located and abstract read during this session (link given).
- **[from memory]** — standard reference I did not re-open this session; verify the citation before it goes into the paper.
- **[measured]** — number produced by a script in this repo, reproducible.
- **[arithmetic]** — derived by calculation, not measured.

---

## 0. Executive summary

1. **The pieces of this project already exist separately; the specific combination does not appear to.** Range/null-space diffusion for CT, flow-matching CT, microlocal "visible/invisible" analysis, conformal calibration of diffusion CT, and sim-to-real DDS studies are all published. I did not find work that (i) tests whether the *spectral* distribution of posterior uncertainty in training-free CT samplers matches what the acquisition geometry predicts, (ii) measures calibration *separately in measured vs. unmeasured Fourier content*, (iii) across diffusion **and** flow priors, (iv) on real raw projections with a matched simulated counterpart. That is the gap. My search was ~30 targeted queries, not a systematic review — re-run it before submission, since preprints in this area appear weekly.
2. **The guidance half of the contribution is at real risk of being a re-derivation.** DDNM (range–null decomposition) and DDS (Krylov/CG data consistency, which by construction only updates the range of Aᵀ) already confine data-driven corrections to what the measurements leave undetermined. A binary "measured vs. unmeasured" Fourier mask is essentially the parallel-beam, continuum version of that. A plausibly novel guidance element would be a *soft, noise-aware, per-frequency* trust weighting — and it has to be ablated against DDS, where it may well tie. **Recommendation: make calibration the headline and treat guidance as secondary,** which also matches your stated priority.
3. **Physics correction to the premise: "unmeasured frequency wedges" is the limited-angle picture, not the sparse-view one.** See §2. For uniformly spaced sparse views the unmeasured set is the gaps *between* radial spokes, and it covers almost the whole frequency disc. This changes what the falsifiable prediction should be.
4. **The "uncertainty concentrates in unmeasured frequencies" prediction is close to tautological for data-consistent samplers.** Any sampler that enforces Ax ≈ y can only vary in the (approximate) null space plus noise. The informative, falsifiable questions are about **magnitude** (is per-band variance calibrated against per-band error?) and about **which samplers violate it** (DPS-style soft guidance should leak variance into measured spokes). §2.3 proposes the sharpened version.
5. **There is direct negative prior evidence you must engage with:** Zhao et al. (2026) find posterior variance in sparse-view CT correlates well with error over the whole volume but poorly in the foreground (Spearman 0.846 → 0.108). That is exactly the failure a trust framework must detect, so it's a baseline finding to reproduce, not to avoid.
6. **Code audit:** none of the prior code is differentiable or in PyTorch, and the Project 1 Python projector pair **fails the adjoint test by 2–84%** [measured]. Part (b) is a new build that reuses your geometry conventions and validation habits, not a wrap of existing code. It also has to come *before* (c), because the SART and TV code in Project 2 sits on the same unmatched operator pair and uses smoothed-TV gradient descent, not Chambolle–Pock.

---

## 1. Literature map

### 1.1 Training-free diffusion posterior sampling for CT
| Work | What it does | Relevance |
|---|---|---|
| **DDS** — Chung, Lee, Ye, ICLR 2024 [verified] ([paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/a8eb51c394d59ffc3324afa672fd24dc-Paper-Conference.pdf), [code](https://github.com/hyungjin-chung/DDS)) | Tweedie denoise → a few CG steps on the data term inside the diffusion step, with no Jacobian through the network. Reports ~100 NFE on par with DiffusionMBIR at 4000 NFE and far better than DPS on sparse-view CT. | Main baseline. The CG updates live in range(Aᵀ), so DDS **already** leaves null-space content to the prior. |
| DPS — Chung et al., ICLR 2023 [from memory] (arXiv 2209.14687) | Gradient guidance through the Tweedie estimate. | Soft consistency. Expected to leak variance into measured content, which makes it a useful contrast. |
| Score-SDE for medical inverse problems — Song et al., ICLR 2022 [from memory] (arXiv 2111.08005) | Sinogram-domain projection onto measured data. | Earliest "keep measured data, let the prior fill the rest" for CT. |
| DiffusionMBIR — Chung et al., CVPR 2023 [from memory] (arXiv 2211.10655) | 2D diffusion + z-TV for 3D sparse-view CT. | DDS's comparison baseline. |
| **DDNM** — Wang et al., ICLR 2023 [from memory] (arXiv 2212.00490) | Explicit range–null decomposition: x = A⁺y + (I − A⁺A)x̄. | **Closest prior art for the guidance idea.** Fourier-slice masks are the parallel-beam analytic special case of (I − A⁺A). |
| **DOLCE** — Liu et al., ICCV 2023 [verified] ([paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_DOLCE_A_Model-Based_Probabilistic_Diffusion_Framework_for_Limited-Angle_CT_Reconstruction_ICCV_2023_paper.pdf)) | Conditional diffusion + data consistency for limited-angle CT; produces variance maps. | Uncertainty maps exist but are qualitative. No calibration study. |
| RN-SDEs — limited-angle, residual null-space SDE [verified] ([arXiv 2409.13930](https://arxiv.org/abs/2409.13930)) | Null-space diffusion for limited-angle CT (trained). | Null-space framing, but trained and without calibration. |
| STRIDE — Zhou et al., 2025 [verified] ([arXiv 2509.05992](https://arxiv.org/abs/2509.05992)) | Sparse-view CT, null-space guided diffusion with temporal reweighting (trained). | Shows "null-space guidance for sparse-view CT" is an active and crowded area. |
| Barba et al., 2026 [verified] ([arXiv 2604.21960](https://arxiv.org/abs/2604.21960)) | Conditional 2D diffusion + explicit DC for 3D sparse-view CT. | Trained and conditional. No UQ. |

### 1.2 Flow matching for CT (second backbone)
| Work | Notes |
|---|---|
| **Evangelista, 2026** [verified] ([arXiv 2608.28256](https://arxiv.org/abs/2608.28256)) | Compares FM solvers (PnP-Flow, FlowDPS, Flower, ICTM) against diffusion solvers (DDRM, DPS, DiffPIR) on Mayo 256², image-domain simulation. Reports FM ≥ diffusion on PSNR/SSIM. **No UQ, no frequency analysis.** Consequence: "we tried two backbones" is *not* novel for reconstruction quality. Backbone-agnosticism only counts as a contribution if the **calibration/uncertainty behaviour** is shown to be invariant (or not). Also a possible source of a released pretrained FM prior, which saves compute. |
| EFMCT, MICCAI 2026 [verified] ([arXiv 2603.00205](https://pith.science/paper/2603.00205), [code](https://github.com/EFMCT/EFMCT)) | Deterministic ODE sampling + repeated DC. Note: a **deterministic** ODE sampler yields no posterior spread unless you randomise the initial noise or use a stochastic variant, and deterministic-ODE "posteriors" are generally not calibrated. That is a real design issue for your FM arm. |
| LUCID, 2026 [verified] ([arXiv 2606.16212](https://arxiv.org/abs/2606.16212)) | Sparsity-adaptive, consistency-guided FM for sparse-view CT. |
| Generic FM inverse solvers: PnP-Flow (Martin et al., ICLR 2025) [verified] ([arXiv 2410.18070](https://arxiv.org/abs/2410.18070)); OT-ODE (Pokle et al. 2024); FlowDPS (Kim et al. 2025); D-Flow; ICTM (Zhang et al. 2024) [names verified via citing papers] | Candidates for the FM arm. Choose one whose DC step can be made *identical* to your DDS arm's, so the backbone is the only variable. |

### 1.3 Geometry-derived (microlocal / Fourier) priors and guidance
| Work | Notes |
|---|---|
| **Bubba et al., "Learning the invisible"**, Inverse Problems 2019 [verified] ([arXiv 1811.04602](https://arxiv.org/abs/1811.04602)) | Uses Quinto's microlocal visibility analysis to learn **only** the invisible shearlet coefficients (limited-angle), with model-based regularisation for the visible part. Philosophically the closest ancestor of your idea. Trained and non-generative. Must be cited prominently. |
| Deep microlocal reconstruction [verified] ([arXiv 2108.05732](https://arxiv.org/abs/2108.05732)); learned microlocal prior [verified] ([arXiv 2201.00656](https://arxiv.org/abs/2201.00656)); visible-singularity-guided net, 2026 [verified] ([arXiv 2602.00184](https://arxiv.org/abs/2602.00184)) | All limited-angle and all trained. |
| FreeSeed, MICCAI 2023 [verified] ([arXiv 2307.05890](https://www.alphaxiv.org/abs/2307.05890)) | Frequency-band-aware network for **sparse-view**. Supervised, no UQ. |

**Observation:** the microlocal/visibility literature is almost entirely **limited-angle**, where the invisible set is a clean wedge. For **sparse-view** there is no invisible wavefront direction; instead there is angular aliasing above a critical radius. This is why §2 matters.

### 1.4 Hallucination and null-space uncertainty
| Work | Notes |
|---|---|
| **Bhadra, Kelkar, Brooks, Anastasio**, IEEE TMI 2021 [verified] ([arXiv 2012.00646](https://arxiv.org/abs/2012.00646), [code](https://github.com/comp-imaging-sci/hallucinations-tomo-recon)) | Decomposes estimates into measurement and null components and defines **hallucination maps**. This is the formal foundation for "unmeasured content = where the prior speaks". Your per-sample null-space component is their hallucination map, applied to a posterior. |
| **Guo, Deng, Grover, 2026** [verified] ([arXiv 2605.15050](https://arxiv.org/abs/2605.15050)) | Separates *intrinsic ambiguity* (null space) from estimation uncertainty in deep generative posteriors and uses **range–null as a calibration tool**, with simulation-based calibration tests. MRI and EEG, not CT. **The closest conceptual competitor for the centerpiece.** Your differentiation: CT, Fourier-slice/sampling-geometry-derived decomposition that is analytic and inspectable, soft noise-aware bands instead of a binary split, real raw projection data, sim-vs-real, two backbones. Read this paper in full before designing experiments. |
| **Zhao, Xu, Liu, 2026** [verified] ([arXiv 2607.13682](https://arxiv.org/abs/2607.13682)) | Sparse-view CT (Gaussian splatting): variance–error Spearman 0.846 over the whole volume vs 0.108 in the foreground. They argue posterior variance is a *constraint map*, not an error map. **Directly relevant negative evidence.** Reproduce this analysis for DDS/FM and report whether your frequency-resolved view explains it. |
| Burns & Fridovich-Keil, 2026 [verified] ([arXiv 2605.30330](https://arxiv.org/abs/2605.30330)) | Popular DPS-style approximations under- or over-estimate posterior spread and hallucinate modes. Expect this; a calibration study should detect it. |

### 1.5 Calibration and conformal UQ for CT
| Work | Notes |
|---|---|
| **K-RCPS** — Teneggi et al., ICML 2023 [verified] ([arXiv 2302.03791](https://arxiv.org/abs/2302.03791), [code](https://github.com/JacopoTeneggi/k-rcps)) | Entrywise conformal risk control of diffusion outputs; abdominal CT (denoising). |
| **sem-CRC** — Teneggi, Stayman, Sulam, MICCAI 2025 [verified] ([arXiv 2503.00136](https://arxiv.org/abs/2503.00136)) | Organ-adaptive conformal risk control for CT. A good template for a *clinically meaningful* calibration target, and for the triage layer. |
| Narnhofer et al., SIIMS 2024 [verified] ([arXiv 2212.12499](https://arxiv.org/abs/2212.12499)) | Posterior variance + conformal → pixel-wise error bounds with coverage guarantees even from approximate samplers (MRI and denoising). |
| Ekmekci & Cetin, 2025 [verified] ([arXiv 2504.07696](https://arxiv.org/abs/2504.07696)) | Conformalised generative Bayesian imaging, includes CT. |
| QUTCC, 2025 [verified] ([arXiv 2507.14760](https://arxiv.org/abs/2507.14760)) | Quantile training + conformal calibration for imaging inverse problems. |

**Implication:** "conformalise a diffusion sampler for CT" is done. The unfilled niche is **diagnosing why and where the raw Bayesian uncertainty is miscalibrated, grounded in acquisition physics**, with conformal correction as the fix, reported alongside it rather than instead of it.

### 1.6 Sim-to-real
**Thomsen, Wang, Lucka, Demircan-Tureyen, 2026** [verified] ([arXiv 2602.12755](https://arxiv.org/abs/2602.12755)). A small correction to how you summarised it:
- Their "real" data is a **physical Shepp–Logan-like phantom** scanned on a lab system, not patient data.
- They use **DDS**, your main baseline, which is convenient.
- Findings: severe domain shift → collapse and hallucinations; **diverse priors match or beat well-matched but narrow priors**; gains "do not immediately translate" to experimental data; **forward-model mismatch pulls samples off the prior manifold** and can be mitigated with annealed likelihood weighting.

For you, the forward-model-mismatch finding is the key one, because Mayo raw data has unmodelled physics (flying focal spot, see §4).

---

## 2. Physics: what "unmeasured" actually means for sparse-view CT

### 2.1 Limited-angle ≠ sparse-view
- **Limited-angle** (views only in [0, θ_max) with θ_max < π): the Fourier-slice theorem gives radial lines only inside a double wedge. A whole **wedge** of directions is missing at every radius. This is the microlocal "invisible singularity" case.
- **Sparse-view** (K views spread uniformly over π): every direction is represented approximately, but neighbouring spokes are πρ/K apart at radius ρ. For an object of diameter D, the Fourier transform is band-limited-like on a 1/D scale, so spokes are adequately dense only for ρ ≲ ρ_c = K/(πD). Above ρ_c you get **inter-spoke gaps that widen linearly with ρ**, which is the origin of streak aliasing.

With detector Nyquist ρ_max = 1/(2Δ) and D = NΔ, ρ_c/ρ_max = 2K/(πN) [arithmetic]:

| K views | 18 | 30 | 60 | 90 | 120 |
|---|---|---|---|---|---|
| ρ_c/ρ_max at N = 512 | 0.022 | 0.037 | 0.075 | 0.112 | 0.149 |

So at every view count in your planned sweep, more than 85% of the radial frequency range is in the aliased regime. Full angular sampling needs K ≈ πN/2 ≈ 804 views [arithmetic]. The unmeasured set is not a wedge. It is a comb-shaped complement of the spokes that fills most of the disc.

### 2.2 Discrete-operator reality
- The continuum argument has a known subtlety: for a compactly supported object the Fourier transform is analytic, so in exact arithmetic nothing is strictly "unmeasured". What matters in practice is the **null space and small-singular-value subspace of the discrete, noisy operator A**. Lower bound: dim null(A) ≥ N² − K·N_det. At N = 512, K = 60 and N_det ≈ 736 that is ≥ ~218k of 262k dimensions [arithmetic; N_det to be read from the DICOM-CT-PD headers].
- **Fan-beam**, which is what the real Mayo data is after rebinning, does not satisfy the Fourier-slice theorem directly, and sparse fan views map to *non-uniform* parallel-beam angles after rebinning. The rigorous generalisation is the **local Fourier transform of the point-spread function of AᵀA** (or of the FBP-with-mask operator) at each image location. For parallel beam that reproduces the spoke pattern. For fan beam it gives a spatially varying pattern. It's computable numerically with your projector, and it's also what makes the framework **geometry-agnostic**, not just backbone-agnostic.
- **Measured spokes are not noise-free.** After ramp filtering, noise variance grows with |ρ|, so the "trust" of a measured frequency is itself a soft, noise-dependent quantity. That motivates soft weights instead of a binary mask.

### 2.3 A sharpened, genuinely falsifiable version of the core prediction
Let x̂ = posterior mean, Σ̂ = sample covariance, and P_b = projector onto frequency band b (radial × angular bins, split into *on-spoke* and *between-spoke*, or more generally binned by the AᵀA transfer-function value).

- **H1 (location):** tr(P_b Σ̂) as a function of b follows the geometry-predicted transfer function: low where AᵀA is large relative to noise, high where it is small. *Expected to hold trivially for hard-DC samplers.* Report it anyway, and test **DPS** as the sampler predicted to violate it.
- **H2 (magnitude — the real test):** per band, E‖P_b(x̂ − x)‖² ≈ tr(P_b Σ̂). The ratio is the **spectral calibration curve**. Miscalibration *in unmeasured bands* is prior error. Miscalibration *in measured bands* is likelihood or forward-model error. **This decomposition is what tells you *why* a reconstruction shouldn't be trusted.** That is the paper's thesis.
- **H3 (sim → real):** on real raw data, forward-model mismatch (FFS, scatter, beam hardening) shows up primarily as miscalibration in *measured* bands, while prior shift shows up in *unmeasured* bands. This one is falsifiable, novel as far as I found, and ties directly to Thomsen et al.
- **H4 (backbone invariance):** H1–H3 hold for both diffusion and FM priors with identical DC steps. If they don't, report that too — it would be an equally interesting result.

---

## 3. Calibration metrics — what to report

Pixelwise coverage and calibration curves are necessary but not sufficient. The standard set, and why each is included:

| Metric | Measures | Note |
|---|---|---|
| **Empirical coverage vs nominal** for central credible intervals at α ∈ {0.5, …, 0.95} → reliability diagram + miscalibration area | Marginal calibration | Pixel-marginal only; ignores spatial correlation. |
| **UCE / ENCE** (binned predicted σ vs observed RMSE) | Regression calibration | Standard in medical-imaging UQ. |
| **CRPS** (sample-based) and **energy score** (multivariate) | Proper scoring rules | Rewards calibration **and** sharpness jointly. Can't be gamed by wide intervals. Prefer these as the headline over NLL. |
| **Sparsification curves / AUSE**, Spearman(σ, \|err\|) **inside a body/foreground mask** | Does uncertainty *rank* errors? | Whole-image numbers are inflated by air (cf. Zhao et al. 2026). Always report foreground. |
| **Simulation-based calibration (rank histograms, Talts et al. 2018)** and **TARP** (Lemos et al. 2023) [from memory] | Is the *sampler* a correct posterior? | Only valid when x is drawn from the prior, i.e. on synthetic draws from your trained model. It's the cleanest test of sampler correctness and separates "sampler wrong" from "prior wrong". |
| **Spectral calibration curve** (§2.3 H2) + **range/null split** (Bhadra-style) | Physics-grounded diagnosis | Your contribution. |
| **Conformalised intervals** (K-RCPS / CRC) — coverage *and* mean width | Guarantee after recalibration | Report raw Bayesian calibration **and** conformal repair. Width tells you how much repair cost. Exchangeability requires **patient-level** calibration/test splits. |
| **Task/ROI-level coverage** (e.g. mean HU in organ ROIs; lesion present/absent) | Clinical meaning | Feeds the triage layer. Cf. sem-CRC. |

All with **patient-level bootstrap CIs**, the same protocol as (c).

---

## 4. Data realities that affect the design

- **TCIA LDCT-and-Projection-data** ([TCIA page](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026)) provides helical, multi-row DICOM-CT-PD projections (>100 scans) plus vendor images.
- **helix2fan** ([repo](https://github.com/faebstn96/helix2fan), Apache-2.0) rebins helical data to flat-detector fan-beam (Noo et al.). Per its README it **does not correctly handle the flying focal spot**, and slices without full 360° coverage get streaks. So "real" data comes with an **unmodelled forward-model mismatch built in**. That's a limitation to state, but also a natural testbed for H3.
- **Reference images:** vendor reconstructions use the full helical/FFS physics, so they are not an exact ground truth for *your* rebinned operator. Proposed protocol:
  - **Simulated arm:** forward-project vendor full-dose images with your operator and subsample views → exact ground truth.
  - **Real arm:** subsample the *measured* rebinned sinogram. Reference = your own full-view reconstruction of the same rebinned data.
  - Same patients in both arms, which makes it a paired sim-vs-real comparison.
- Your MONAI project used **reconstructed DICOM images only** (`src/dicom_loader.py`). Nothing in the prior work reads raw projections yet, so plan real time for this.

---

## 5. Prior-code audit (part b — read-only, no building yet)

| Project | What's there | Reusable for the flagship | Issues found |
|---|---|---|---|
| 1 — Parallel FBP (Py/C++/C#) | `forward_project` = scipy `rotate` + column sum. `back_project` = tile + `rotate(-θ)`, **divided by K**. Ramp filter with no zero-padding (P1). | Cross-language validation habit; Shepp–Logan builder. | **Adjoint test fails** [measured]: relative error 2.1%–83.9% across 9 trials (N ∈ {64, 128}, K ∈ {30, 60, 90}), after undoing the 1/K. Script: `docs/evidence/legacy_p1_adjoint_check.py`. Cause: the transpose of bilinear interpolation is a *splat*, not a rotation by −θ, and `rotate` zeroes out-of-bounds pixels asymmetrically. The P1 ramp filter also lacks the zero-padding P3 added, so it produces circular-convolution wraparound. Not usable as the flagship operator. |
| 2 — SART + TV (C++) | SART uses the same rotate-based pair. TV = explicit gradient descent on smoothed TV (ε). | Semi-convergence demonstration; row/column-sum normalisation logic. | Unmatched operators mean SART isn't quite SART. **This is not Chambolle–Pock** — (c) needs a new PDHG implementation with proper step sizes (τσ‖K‖² < 1, with ‖K‖ from power iteration on the new operator). |
| 3 — Fan/cone FDK + ASTRA | Fan-beam forward = **ASTRA's** `line_fanflat` (not yours). Fan-beam FBP and cone FDK backprojectors are pixel-driven with D²/U² weights. Cone forward = ray-marching with `map_coordinates`. | Geometry conventions (sign conventions debugged against ASTRA), `detector_axis` fix, cosine weights. **Most valuable prior asset.** | Cone forward (ray-march) and FDK backprojector (weighted pixel-driven) are **not transposes of each other**, by design — FDK's backprojector is an inversion formula, not Aᵀ. The ASTRA check compared **FDK outputs** (2.6% NRMSE), not the forward projector. All NRMSEs use **min–max rescaling**, which hides global scale/offset errors; the flagship needs absolute units (μ or HU). Hard-coded paths `/Users/youssef/Project_3/...`. |
| 4 — CUDA backprojection | Parallel-beam pixel-driven BP only, float32; texture variant with 9-bit interpolation weights (you documented this). | Profiling methodology; kernel experience for a matched ray-driven pair later. | No forward kernel. The texture path should **not** be used for an adjoint-matched pair (hardware interpolation breaks exact transposition). |
| 5 — MONAI denoiser + Frontier container | Patient-level split, HU windowing, container packaging. | Data pipeline and packaging patterns; patient-level split discipline. | Image-domain only. 10 patients / 2 test patients, so its bootstrap CIs would be very wide. The flagship needs more patients for the calibration/test split. |

**What (b) will actually need** (for when you say go — nothing built yet):
1. A **ray-driven (Joseph or Siddon) forward operator** for parallel and flat-fan geometry in PyTorch/CUDA, with its **exact transpose** as the backward pass, wrapped in `torch.autograd.Function`.
2. A dot-product test at float64 on CPU (target ~1e-12) and float32 on GPU (target ~1e-5), plus `torch.autograd.gradcheck`.
3. A cross-check against ASTRA's projector with **matched geometry and matched discretisation**. Joseph vs ASTRA's `line`/`strip` kernels differ at the discretisation level, so the expected agreement is not machine precision — we'll pre-register a tolerance.
4. A decision to make at that point: write it yourself (best learning value, and it's what reuses P3/P4) or wrap an existing matched pair (ASTRA via tomosipo, torch-radon, LEAP) as a reference. My recommendation is to do both: yours is the system under test, the library is the oracle.

---

## 6. Scope flags (honest, not discouraging)
- Holoscan + TensorRT for an iterative sampler (≥ 50–100 network evaluations + CG per slice) is a substantial engineering project on its own. Plan it as a **distilled/few-step** or **FBP + calibrated-UQ-head** demo, and say so.
- Phase 2 (motion) and the triage agent depend on the calibration result being positive. Gate them on H2/H3 outcomes.
- A MONAI contribution: MONAI has no CT projector. A small, well-tested contribution such as a calibration metric (e.g. CRPS / sparsification for image outputs) is more realistic than landing a projector. Check open issues before choosing.

## 7. Decisions (resolved 2026-09-29)
1. **Accepted:** calibration is the headline result; H1–H4 are the pre-registered hypotheses.
2. **Resolution: 256².**
3. **Projector:** the user writes it themselves; a library (ASTRA/tomosipo or LEAP) serves as the oracle.

### Original questions
1. Accept the reframing (calibration as headline; H1–H4 as the pre-registered hypotheses)?
2. Resolution for the benchmark: 512² (clinically faithful, costly on Colab) or 256² (standard in the DDS/FM literature)?
3. Build the projector yourself with a library oracle (recommended), or wrap a library?

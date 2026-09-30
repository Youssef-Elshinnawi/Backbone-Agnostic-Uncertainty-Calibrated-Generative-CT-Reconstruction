# Backbone-Agnostic, Uncertainty-Calibrated Generative CT Reconstruction

**A physics-grounded trust framework for point-of-care imaging**

Author: Youssef Elshinnawi, Universität Leipzig (International Physics Studies Program)

> **Status:** Phase 0 complete (literature review, prior-code audit, pre-registration frozen at tag `prereg-v1`). Phase 1 (differentiable projector) not started. **No results yet.** Any number that appears in this repository links to the script and commit that produced it.

## Question

When should a generative sparse-view CT reconstruction be trusted?

This project derives, from the scan's acquisition geometry (the Fourier-slice theorem, generalised through the transfer function of AᵀA), which image content the measurements determine and which they leave to the generative prior. It then tests whether the posterior uncertainty of generative reconstructions is **calibrated**, band by band, in measured versus unmeasured content. The tests are run for:
- two different generative backbones: a diffusion model and a flow-matching model;
- both simulated and real raw projection data (TCIA / Mayo LDCT).

The hypotheses (H1–H4) and the evaluation protocol are fixed in advance in [`docs/preregistration.md`](docs/preregistration.md).

## Roadmap

| Phase | Achieves |
|---|---|
| 0 · Foundations ✅ | Gap identified; hypotheses H1–H4; design decisions ([review](docs/phase1_literature_review_and_code_audit.md)) |
| 1 · Differentiable projector | Self-written ray-driven parallel/fan-beam operator as `torch.autograd.Function`; adjoint test, gradcheck, ASTRA cross-check |
| 2 · Real-data pipeline | Mayo DICOM-CT-PD → fan-beam rebinning → 256²; paired simulated/real arms; patient-level splits |
| 3 · Classical baselines | FBP, SART, Chambolle–Pock TV at 18/30/60/90/120 views; patient-level bootstrap CIs |
| 4 · Generative priors & samplers | Diffusion + flow-matching priors; DDS (main baseline), DPS, FM sampler with a shared data-consistency step |
| 5 · Physics framework | Geometry-derived frequency bands; range/null decomposition; noise-aware guidance (ablated against DDS) |
| 6 · Calibration study | H1–H4; coverage, UCE/ENCE, CRPS, AUSE, SBC/TARP, spectral calibration; conformal repair |
| — gate — | If H2/H3 fail, report as a diagnostic/negative result; Phase 9 proceeds only with a usable calibrated signal |
| 7 · Systems & deployment | CUDA/Nsight optimisation, TensorRT, NVIDIA Holoscan real-time demo |
| 8 · Paper & open source | arXiv + MICCAI / NeurIPS-workshop submission; MONAI contribution |
| 9 · Extensions | Motion-robust reconstruction; agentic triage (escalate vs standard queue) |

## Scope statement

Sparse-view acquisition is studied as a **hypothetical** dose/speed/cost scenario. Real portable and mobile-stroke-unit scanners (e.g. CereTom, OmniTom Elite, SOMATOM On.site) acquire full angular sampling. Nothing here claims to reflect any specific device's operation.

## Repository layout

```
docs/          design documents, literature review, pre-registration, evidence scripts
src/           library code (projector, reconstruction, samplers, metrics)
tests/         unit and property tests (adjoint test, gradcheck, ...)
configs/       experiment configurations
notebooks/     Colab notebooks (committed with outputs stripped)
results/       small metric files (JSON/CSV), each recording config + commit hash
```

## Data and reproducibility policy

- **No data, checkpoints or large arrays are committed.** Raw projections (TCIA LDCT-and-Projection-data) and model weights live on Google Drive and are subject to TCIA's data-use terms.
- Every reported number must come from a committed script and a recorded run. Result files store the git commit hash, the config and the random seeds.
- Results are labelled either *single run / unvalidated* or *benchmarked* (multiple seeds, patient-level bootstrap CIs).
- Negative results are reported.
- Notebooks are committed with outputs stripped (`nbstripout`).

## Prior work by the author

This project builds on five earlier projects:
- [Filtered back-projection (Python/C++/C#)](https://github.com/Youssef-Elshinnawi/Filtered-Back-Projection)
- [Iterative SART + TV](https://github.com/Youssef-Elshinnawi/Iterative-CT-Reconstruction-SART-TV)
- [Fan-beam / cone-beam FDK, validated against ASTRA](https://github.com/Youssef-Elshinnawi/Fan-Beam-Cone-Beam-CT-Reconstruction-FDK-ASTRA)
- [CUDA back-projection](https://github.com/Youssef-Elshinnawi/GPU-accelerated-back-projection-in-CUDA)
- [MONAI low-dose CT denoising](https://github.com/Youssef-Elshinnawi/MONAI-denoising-model-for-CT-reconstruction)

## License

To be decided.

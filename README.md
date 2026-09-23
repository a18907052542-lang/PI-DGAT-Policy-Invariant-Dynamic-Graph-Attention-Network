# PI-DGAT — Policy-Invariant Dynamic Graph Attention Network

Companion code and dataset for the study
**PI-DGAT: A Policy-Invariant Dynamic Graph Attention Network for Dynamic
Early Warning of Credit Default Risk of Small and Micro Enterprises**.

PI-DGAT is a dynamic multi-relational graph model with a dual-channel
attention operator that decouples an invariant channel from an
environment channel, an interventional environment-reweighting training
objective enforcing invariance across policy cycles, a censoring-aware
discrete-time hazard rate head, and a non-exchangeable conformal
decision layer that outputs an admit / review / reject decision with a
statistically controlled false-rejection rate.

This repository releases the complete implementation together with the
dataset used to produce the reported tables and figures.

---

## Repository contents

Two archives are attached to the GitHub release. They are meant to be
downloaded together into the same working directory.

| Archive                     | Contents                                                                                              |
|-----------------------------|-------------------------------------------------------------------------------------------------------|
| **`PI-DGAT_dataset.zip`**   | Eighteen CSV files, one data-dictionary README. 218,436 firm nodes, 1,912,364 firm-quarter observations, 787,522 valid early-warning samples across four policy environments (E1–E4). |
| **`PI-DGAT_code.zip`**      | PyTorch implementation of every equation (1)–(21) and every algorithm (1–3); baselines (LightGBM, GAT, HAN, EvolveGCN, DySAT, DyMGNN); driver scripts for Tables 5–7, the ablation study, the OOD-generalisation study, the conformal-coverage study, the economic-cost study, and Figures 8–13 (Nature style). |

The two archives together reproduce every reported quantitative
result.

---

## Quick start

```bash
# 1. Download and unpack both archives side by side
unzip PI-DGAT_dataset.zip     # → PI-DGAT_dataset/
unzip PI-DGAT_code.zip        # → PI-DGAT_code/

# 2. Install dependencies
cd PI-DGAT_code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. One command runs every experiment
bash run_all.sh
```

`run_all.sh` executes eight steps and writes every output under
`PI-DGAT_code/results/`:

1. Pipeline verification — end-to-end run on the shipped dataset that
   exercises every module and confirms loss descent, finite gradients
   and hazard rates strictly inside (0, 1).
2. Table 5 — early-warning performance comparison on test set E4.
3. Table 6 — out-of-distribution generalisation E1 / E2 / E3 → E4.
4. Table 7 — nominal versus empirical false-rejection rates for the
   non-exchangeable conformal decision layer.
5. Ablation table — dual-channel decoupling, interventional
   reweighting, self-preserving residual, and conformal decision layer.
6. Supporting tables for the attention-shift, lead-time and PCA
   analyses.
7. Cost-of-misjudgment analysis reproducing the 18.8 % reduction in
   aggregate cost relative to the strongest baseline.
8. Figures 8–13 rendered in Nature style (white background, English,
   ≥ 20 pt smallest text).

---

## Method overview

Given a dynamic multi-relational credit graph with four edge types
(guarantee, transaction, equity / executive, regional-industry
proximity) and a firm-level environment vector composed of policy
timing, pilot-zone affiliation, industry environmental standard
stringency and firm-level green exposure, PI-DGAT

1. **Decouples** the invariant channel and the environment channel
   inside the scoring function of the graph attention operator, so
   information about repayment capacity is not mixed with information
   sensitive to policy (Equations (5)–(10)).
2. **Enforces invariance** by intervening on the environment channel at
   training time and adding a cross-environment variance and gradient-
   norm penalty (Equations (11) and (15)).
3. **Estimates a hazard-rate sequence** through a discrete-time,
   censoring-aware head with a per-period baseline (Equations
   (12)–(14)).
4. **Converts hazard rates into decisions** through a non-exchangeable
   conformal quantile with time-decay weighting and a buffer band
   whose width is aligned to review-capacity constraints
   (Equations (18)–(21)).

Full-scale training uses the following configuration:

- 4 096 center nodes per mini-batch, subgraph fan-outs (25, 10);
- 120 epochs, Adam, learning rate 1e-3;
- λ₁ = 0.5 (invariance term), λ₂ = 0.1 (decorrelation term);
- α = 0.10 (nominal false-rejection tolerance);
- ζ = 0.85 (time-decay factor in the weighted conformal quantile);
- five independent training runs, means and standard deviations
  reported.

A single full run takes about 3.7 hours on one NVIDIA A100 40GB, with a
peak memory footprint of about 31 GB.

---

## Reported results

Headline numbers from the reported tables (see
`PI-DGAT_code/results/tables/`):

- **Table 5 (test set E4, mean ± std over five runs).**
  PI-DGAT reaches AUC 0.864 ± 0.007, KS 0.552 ± 0.009, AP 0.283 ± 0.008
  and an early-warning lead time of 2.34 ± 0.19 quarters, best on all
  four metrics.
- **Table 6 (out-of-distribution AUC).** Training on E1 alone and
  testing on E4, PI-DGAT reaches AUC 0.831 versus 0.762 for the
  strongest baseline; the decay ratio is roughly 2.4× smaller.
- **Table 7 (conformal coverage).** At the nominal level α = 0.10,
  PI-DGAT's empirical false-rejection rate is 0.108 with a standard
  deviation of 0.006, versus 0.147 under the standard conformal
  procedure.
- **Cost analysis.** Under matched operating points, PI-DGAT reduces
  the aggregate cost of misjudgment by 18.8 % relative to the strongest
  baseline, of which 2.5 pp comes from lower missed-default losses and
  16.3 pp from lower opportunity cost through tighter false-rejection
  control.

---

## Data provenance

The dataset is assembled from five public streams: firm registry and
business-change records from `gsxt.gov.cn`, guarantee and subrogation
events from the Credit Reference Center of the People's Bank of China,
upstream–downstream trade linkages from customs and stock-exchange
disclosures, industry environmental-standard stringency from NBS and
MEE releases, and green-finance policy and pilot-zone information from
PBoC and State Council designations of the Green Finance Reform and
Innovation Pilot Zones (three batches: 2017Q3, 2019Q4, 2022Q3).

Sectoral encoding follows GB/T 4754-2017. Administrative-division
encoding follows GB/T 2260. Small-and-micro sizing follows the NBS
Statistical Method for Classifying Large, Medium, Small and Micro
Enterprises (2017). Personal identifiers are not stored anywhere in the
archive.

---

## Citation

If this work is useful to your research, please cite the study.

```bibtex
@article{pidgat2026,
  title  = {A Policy-Invariant Dynamic Graph Attention Network for
            Dynamic Early Warning of Credit Default Risk of Small and
            Micro Enterprises},
  author = {\{corresponding author\}},
  year   = {2026}
}
```

---

## License

Released under the MIT License. Third-party dependencies retain their
own licenses.

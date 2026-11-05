# T2I Models Rarely Invert Negation but Often Fail to Suppress Negated Concepts: The T2I-NSI Benchmark for Negation Suppression in Text-to-Image Generation

Official code and benchmark release for the WACV 2027 paper.

**Authors:** Akansha Gautam, Aparna Bharati, Mayank Vatsa, Richa Singh
(Indian Institute of Technology Jodhpur; Lehigh University)

---

## Overview

T2I-NSI is a benchmark and metric for measuring negation sensitivity in
text-to-image (T2I) generation. It pairs each negated prompt with a canonical
positive counterpart, so suppression is measured by contrast. Probes are queried
only with positive concept descriptions.

The benchmark factorizes negation along two axes:

- Linguistic form: lexical/semantic, morphological, syntactic
- Target complexity: L0 object, L1 attribute, L2 relation, L3 composition, L4 nested

The benchmark contains 759,869 paired prompt atoms over a 5,082-object catalog,
with 916,272 images rendered across six generators.

---

## Main Findings

- Negation sensitivity is low: open-weights generators score NSI ≤ 0.038, while
  GPT-Image-1.5 and Gemini-2.5-Flash-Image reach about 0.09.
- Inverse interpretation is essentially absent: `INVERSE` is 0.00% across over
  450K evaluated atom-model pairs.
- The dominant failure is under-suppression. Models render the concept under
  both prompts on 8–19% of atoms, with measurable partial suppression on a
  further 9–25%.
- On identical prompts, GPT-Image-1.5 scores about 3.5× the open-weights mean.
- A blinded human study (420 atoms, nine annotators) corroborates the
  near-absence of inversion and the high persistence rate.

---

## Models

| Tier          | Model                  | Name in this repository | Atoms   | Paired seeds |
|---------------|------------------------|-------------------------|---------|--------------|
| Open-weights  | FLUX.2 (klein-9B)      | `flux`                  | 151,997 | Yes          |
| Open-weights  | Qwen-Image-2512        | `qwen`                  | 151,997 | Yes          |
| Open-weights  | SD 3.5 Large Turbo     | `stable-diffusion`      | 151,997 | Yes          |
| Closed-source | GPT-Image-1.5          | `gpt`                   | 1,200   | No           |
| Closed-source | Gemini-2.5-Flash-Image | `gemini`                | 717     | No           |
| Closed-source | Midjourney v7          | `midjourney`            | 228     | Yes          |

Midjourney v7 covers syntactic atoms only. GPT-Image-1.5 and Gemini-2.5-Flash-Image
expose no seed parameter, so their positive and negated images are independent
samples from matched prompts.

---

## Metrics

Per atom *j*:

- **R<sup>+</sup><sub>j</sub>, R<sup>-</sup><sub>j</sub>** — target presence
  under the positive and negated prompts.
- **C<sub>j</sub> = R<sup>+</sup><sub>j</sub>** — positive realization
  (`coverage` in the code).
- **NSI<sub>j</sub> = R<sup>+</sup><sub>j</sub> − R<sup>-</sup><sub>j</sub>** —
  Negation Sensitivity Index, in [-1, 1].
- **P<sup>-</sup><sub>j</sub>** — preservation of the surrounding scene under
  the negated prompt.
- **NegAlign<sub>j</sub> = ½ (C<sub>j</sub> + (1 − R<sup>-</sup><sub>j</sub>)) P<sup>-</sup><sub>j</sub>**
  — preservation-aware score.

Two estimators are provided:

- **Basic:** one CLIP-L/14 probe.
- **Robust:** the median of CLIP-L/14, InstructBLIP and GroundingDINO, with
  realization filtering and uncertainty weighting
  *w<sub>j</sub> = exp(−β U<sub>j</sub>)*.

Default parameters are ξ = 0.3, θ<sub>pos</sub> = 0.7, θ<sub>neg</sub> = 0.3
and β = 2.0.

### Failure Modes

The Robust scorer labels each atom in the `class_j` column:

| Class            | Condition                         | Meaning                              |
|------------------|-----------------------------------|--------------------------------------|
| `SUCCESS`        | C ≥ θ<sub>pos</sub>, R⁻ ≤ θ<sub>neg</sub> | Rendered when asked, suppressed when negated |
| `ALWAYS_PRESENT` | C ≥ θ<sub>pos</sub>, R⁻ ≥ θ<sub>pos</sub> | Present under both prompts           |
| `ALWAYS_ABSENT`  | C < θ<sub>neg</sub>, R⁻ < θ<sub>neg</sub> | Rarely rendered under either prompt  |
| `INVERSE`        | C ≤ θ<sub>neg</sub>, R⁻ ≥ θ<sub>pos</sub> | Present only under negation          |
| `OTHER`          | Everything else                   | Residual region                      |

The paper reports `OTHER` as two subclasses, which you can derive from `C_j`:
`PARTIAL_SUPPR` (C ≥ θ<sub>pos</sub>) and `WEAKLY_REND` (C < θ<sub>pos</sub>).

These classes are thresholded diagnostic bins, not human-equivalent labels, and
their exact shares depend on the thresholds.

---

## Repository Structure

```text
.
├── code/
│   ├── basic_algo.py        # Basic (CLIP-only) T2I-NSI scorer
│   └── robust_algo.py       # Robust (probe-ensemble) T2I-NSI scorer
│
├── dataset/
│   ├── images/
│   │   └── metadata/                # rendered atoms, one CSV per model
│   │       ├── open_models/
│   │       │   ├── flux.csv                  # 151,997 atoms
│   │       │   ├── qwen.csv                  # 151,997 atoms
│   │       │   └── stable-diffusion.csv      # 151,997 atoms
│   │       │
│   │       └── closed_models/
│   │           ├── gemini.csv                # 717 atoms
│   │           ├── gpt.csv                   # 1,200 atoms
│   │           └── midjourney.csv            # 228 atoms
│   │
│   └── prompts/                     # all generated prompts
│       ├── open_models_deepseek/    # 759,869 atoms
│       │   ├── lexsem_prompts.csv
│       │   ├── morphological_prompts.csv
│       │   └── syntactic_prompts.csv
│       │
│       └── closed_models_claude/    # 1,200 atoms
│           ├── lexsem_prompts.csv
│           ├── morphological_prompts.csv
│           └── syntactic_prompts.csv
│
├── requirements.txt
└── README.md
```

---

## Requirements

Developed against Python 3.9+. Dependencies are listed in `requirements.txt`.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Dataset

`dataset/prompts/` holds every generated prompt. `dataset/images/metadata/`
records the atoms each model rendered. The PNG images will be added later.

| Split         | Folder                 | Prompt LLM        | Atoms (corpus) | Atoms (rendered) | Rendered prompts | Images  |
|---------------|------------------------|-------------------|----------------|------------------|------------------|---------|
| Open-weights  | `open_models_deepseek` | DeepSeek v3.2     | 759,869        | 151,997          | 303,994          | 911,982 |
| Closed-source | `closed_models_claude` | Claude Sonnet 4.6 | 1,200          | 1,200            | 2,400            | 4,290   |

All prompts are LLM-generated and in English, so the benchmark measures a
constructed distribution rather than real user requests.

### Prompt Files

Each folder has one file per negation class:

| File                        | `open_models_deepseek` | `closed_models_claude` |
|-----------------------------|------------------------|------------------------|
| `lexsem_prompts.csv`        | 253,576                | 400                    |
| `morphological_prompts.csv` | 253,044                | 400                    |
| `syntactic_prompts.csv`     | 253,249                | 400                    |

The open-weights folder holds the full prompt corpus, of which 151,997 atoms were
rendered per model.

The closed-source folder holds one shared prompt set. GPT-Image-1.5 rendered all
1,200 atoms, Gemini-2.5-Flash-Image a subset of 717, and Midjourney v7 a
syntactic-only subset of 228. These subsets use identical prompt text.

### Atom ID Format

```text
{NEGATION_CLASS}_{group}_L{complexity}_{sample}
```

Examples: `LEXSEM_0_L0_01`, `MORPH_3_L2_04`, `SYN_18_L4_01`.

Atom IDs are unique within a folder only. The same ID can refer to different
prompts in `open_models_deepseek` and `closed_models_claude`.

### Columns

Every prompt file has these eight columns:

| Column             | Meaning                                                        |
|--------------------|----------------------------------------------------------------|
| `atom_id`          | Atom ID, unique within a folder                                |
| `negation_class`   | `lexsem` / `morphological` / `syntactic`                       |
| `complexity_level` | `L0`–`L4`                                                      |
| `target_type`      | `object` / `attribute` / `relation` / `composition` / `nested` |
| `object`           | High-level subject (diagnostic only)                           |
| `positive_prompt`  | Positive prompt p<sub>j</sub><sup>+</sup>                      |
| `negative_prompt`  | Negated prompt p<sub>j</sub><sup>-</sup>                       |
| `notes`            | Free-text notes from prompt construction                       |

In `closed_models_claude`, six rows use the variant `target_type` labels
`object_presence`, `compositional` and `quantified_nested`.

### Image Metadata

Each file in `dataset/images/metadata/` lists the atoms rendered by one model.
All six files have the same columns:

| Column                                    | Meaning                                                   |
|-------------------------------------------|-----------------------------------------------------------|
| `atom_id`                                 | Atom ID, matching the prompt files                        |
| `negation_class`, `complexity_level`      | As in the prompt files                                    |
| `target_type`, `object`                   | As in the prompt files                                    |
| Six target columns                        | Described under [Probe Scores](#probe-scores)             |
| `positive_image_path`, `negative_image_path` | Image location within `dataset/images/`                |
| `positive_seed`, `negative_seed`          | Generation seeds                                          |

To get the prompt text, join on `atom_id` with the matching folder in
`dataset/prompts/`: `open_models` with `open_models_deepseek`, and `closed_models`
with `closed_models_claude`.

Seed columns are blank for GPT-Image-1.5 and Gemini-2.5-Flash-Image, which expose
no seed parameter. Seeds are paired for Midjourney v7. For the open-weights
models, 20 of 455,991 atom-model pairs have different positive and negative seeds.

Image folders use `lexical` for open-weights lexical/semantic atoms and `lexsem`
for closed-source ones.

---

## Probe Scores

The prompt files alone are not enough to run the scorers. The scorers do not run
the probes, so their input CSV must carry target lists and precomputed scores.

| Target column           | Meaning                                                   |
|-------------------------|-----------------------------------------------------------|
| `neg_targets_json`      | JSON list T<sub>j</sub><sup>neg</sup> (CLIP anchor form)  |
| `pres_targets_json`     | JSON list T<sub>j</sub><sup>pres</sup> (CLIP anchor form) |
| `neg_targets_vqa_json`  | T<sub>j</sub><sup>neg</sup> phrased as VQA questions      |
| `pres_targets_vqa_json` | T<sub>j</sub><sup>pres</sup> phrased as VQA questions     |
| `neg_targets_det_json`  | T<sub>j</sub><sup>neg</sup> phrased as detection prompts  |
| `pres_targets_det_json` | T<sub>j</sub><sup>pres</sup> phrased as detection prompts |

Scores are JSON lists, in the same order as the corresponding target list.

| Algorithm | Required columns for each model `m`                                                                                                                  |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| Basic     | `{m}_clip_{neg,pres}_target_{posimg,negimg}_json`                                                                                                    |
| Robust    | `{m}_{p}_{neg,pres}_target_{posimg,negimg}_json` for each probe `p` in `clip`, `vqa`, `det`, plus `{m}_clip_imgpair_sim` (CLIP image-pair similarity) |

The two scorers expect different CLIP scales:

- **Basic:** raw cosine similarities in [-1, 1]. The scorer applies
  (s + 1) / 2 itself, so do not pass scores that are already mapped.
- **Robust:** all scores already in [0, 1]. CLIP uses (cosine + 1) / 2, VQA uses
  the soft "yes" probability, and the detector uses the maximum detection score.

---

## Evaluation

Replace `scores.csv` with a CSV in the format above.

### Basic (CLIP-only)

```bash
python code/basic_algo.py \
    --meta_csv scores.csv \
    --out_csv results/basic.csv \
    --model_name flux qwen stable-diffusion \
    --coverage_threshold 0.3
```

### Robust (Probe Ensemble)

```bash
python code/robust_algo.py \
    --meta_csv scores.csv \
    --out_csv results/robust.csv \
    --model_name flux qwen stable-diffusion \
    --coverage_threshold 0.3 \
    --theta_pos 0.7 \
    --theta_neg 0.3 \
    --beta 2.0
```

Each `--model_name` value must match the column prefix used in the CSV. Pass
`--model_name all` to score every model found in the CSV.

Generated outputs for `--out_csv results/robust.csv`:

```text
results/robust_<model>.csv
results/robust_<model>_summary.json
results/robust_all_models_summary.json    # when scoring more than one model
```

---

## Citation

```bibtex
@inproceedings{gautam2027t2insi,
  title     = {T2I Models Rarely Invert Negation but Often Fail to Suppress Negated Concepts: The T2I-NSI Benchmark for Negation Suppression in Text-to-Image Generation},
  author    = {Gautam, Akansha and Bharati, Aparna and Vatsa, Mayank and Singh, Richa},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

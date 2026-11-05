"""
Robust T2I-NSI — implementation of the robust algorithm.

Consumes a wide-form per-atom CSV in which every probe in the ensemble
(CLIP, VQA, open-vocabulary detector) has already been run, and the
per-target scores have been merged into JSON-list columns.

================================================================================
EXACT input CSV schema (one row per atom)
================================================================================

Atom metadata (used by the robust per-atom routine and the conditioned-score
aggregator):
    atom_id              — atom primary key, e.g. "LEXSEM_3171_L0_09"
    negation_class       — paper symbol tau_form ∈ {lexsem, morphological, syntactic, lexical, ...}
    complexity_level     — paper symbol ell ∈ {L0..L4}
    target_type          — paper symbol tau_target ∈ {object, attribute, relation, composition, nested, ...}
    object               — high-level subject (diagnostic only)
    positive_prompt      — p_j^+  (one element of P_j^+)
    negative_prompt      — p_j^-  (one element of P_j^-)

Concept-anchor lists (JSON arrays of strings, same order as the score lists):
    neg_targets_json         — T_j^neg         (anchor form used by CLIP)
    pres_targets_json        — T_j^pres        (anchor form used by CLIP)
    neg_targets_vqa_json     — VQA question form of T_j^neg
    pres_targets_vqa_json    — VQA question form of T_j^pres
    neg_targets_det_json     — detection phrase form of T_j^neg
    pres_targets_det_json    — detection phrase form of T_j^pres

Per-probe, per-model score columns (each a JSON list of floats, same length
as the corresponding target list). model m ∈ {flux, qwen, stable-diffusion};
probe p ∈ {clip, vqa, det}:
    {m}_{p}_neg_target_posimg_json    — g_p(I_j^+, t)  for t ∈ T_j^neg    ← s^+
    {m}_{p}_neg_target_negimg_json    — g_p(I_j^-, t)  for t ∈ T_j^neg    ← s^-
    {m}_{p}_pres_target_posimg_json   — g_p(I_j^+, t)  for t ∈ T_j^pres   ← r^+
    {m}_{p}_pres_target_negimg_json   — g_p(I_j^-, t)  for t ∈ T_j^pres   ← r^-

Per-model global image similarity (scalar, from CLIP image tower φ):
    {m}_clip_imgpair_sim   — sim_img(I_j^+, I_j^-)   ← S_j

================================================================================
K=1 simplification
================================================================================

The robust per-atom routine in the paper loops over K seeds. This benchmark
precomputes one (I_j^+, I_j^-) pair per atom × model, so K = 1 and every per-k
accumulator collapses to a single iteration. We keep the K divisions in place
for fidelity with the paper.

================================================================================
Defaults (paper, Robust estimator)
================================================================================
    E = 3 probes (clip, vqa, det)
    xi (coverage threshold)  = 0.3
    theta_pos                = 0.7
    theta_neg                = 0.3
    beta                     = 2.0
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Probe ensemble {g_e}, e = 1..E. E = 3 (paper §Robust Variant).
PROBES: List[str] = ["clip", "vqa", "det"]

FAILURE_MODES: List[str] = [
    "ALWAYS_ABSENT",
    "ALWAYS_PRESENT",
    "SUCCESS",
    "INVERSE",
    "OTHER",
]


# -----------------------------------------------------------------------------
# Column-name helpers — every CSV column the algorithm ever reads is
# constructed here so the schema is in one place.
# -----------------------------------------------------------------------------

# Atom-level metadata columns (input to the robust per-atom routine and to
# the conditioned-score aggregator).
COL_ATOM_ID          = "atom_id"
COL_NEGATION_CLASS   = "negation_class"      # tau_form
COL_COMPLEXITY       = "complexity_level"    # ell
COL_TARGET_TYPE      = "target_type"         # tau_target
COL_OBJECT           = "object"
COL_POS_PROMPT       = "positive_prompt"
COL_NEG_PROMPT       = "negative_prompt"

# Concept-anchor lists.
COL_NEG_TARGETS_JSON  = "neg_targets_json"   # T_j^neg (CLIP form)
COL_PRES_TARGETS_JSON = "pres_targets_json"  # T_j^pres (CLIP form)


def col_probe(model: str, probe: str, role: str, img_src: str) -> str:
    """
    Per-probe score column.

    role    ∈ {"neg", "pres"}     — which target list: T_j^neg vs T_j^pres
    img_src ∈ {"posimg", "negimg"} — which image: I_j^+ vs I_j^-

    E.g. col_probe("flux", "clip", "neg", "posimg")
         → "flux_clip_neg_target_posimg_json"
    """
    assert probe in PROBES, f"Unknown probe {probe!r}"
    assert role in ("neg", "pres")
    assert img_src in ("posimg", "negimg")
    return f"{model}_{probe}_{role}_target_{img_src}_json"


def col_imgpair_sim(model: str) -> str:
    """sim_img(I+, I-) column — S_j in the robust algorithm. The paper uses
    CLIP's φ as the image encoder, so this always lives under the clip
    namespace regardless of which VQA/Det probes we ensemble later."""
    return f"{model}_clip_imgpair_sim"


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------

def parse_concept_list(value) -> List[str]:
    """Parse a JSON array of strings (or a `;`-separated fallback)."""
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() in ("nan", "<na>", "none"):
        return []
    if s.startswith("[") and s.endswith("]"):
        parsed = json.loads(s)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON list, got {type(parsed).__name__}")
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip() for part in s.split(";") if part.strip()]


def parse_score_list(value) -> List[float]:
    """Parse a JSON list of floats. Missing entries surface as NaN."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    s = str(value).strip()
    if not s or s.lower() in ("nan", "<na>", "none"):
        return []
    parsed = json.loads(s)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list of floats, got {type(parsed).__name__}")
    out: List[float] = []
    for v in parsed:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def _scalar_or_nan(value) -> float:
    if value is None:
        return float("nan")
    try:
        if pd.isna(value):
            return float("nan")
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# -----------------------------------------------------------------------------
# Per-atom failure-mode classifier (robust algorithm).
# -----------------------------------------------------------------------------

def classify_failure_mode(C_j: float, R_minus: float, theta_pos: float, theta_neg: float) -> str:
    """Direct transcription of the failure-mode cascade in the paper."""
    if C_j <= theta_neg and R_minus <= theta_neg:
        return "ALWAYS_ABSENT"
    if C_j >= theta_pos and R_minus >= theta_pos:
        return "ALWAYS_PRESENT"
    if C_j >= theta_pos and R_minus <= theta_neg:
        return "SUCCESS"
    if C_j <= theta_neg and R_minus >= theta_pos:
        return "INVERSE"
    return "OTHER"


# -----------------------------------------------------------------------------
# Robust per-atom score (RobustAtomScore, K = 1)
# -----------------------------------------------------------------------------

def _load_probe_matrix(
    atom_row: pd.Series,
    model: str,
    role: str,
    img_src: str,
    expected_len: int,
) -> np.ndarray:
    """
    Stack the per-probe score lists for a given (role, img_src) into a
    matrix of shape (|PROBES|, expected_len).

    Row e holds probe e's scores; column t holds target t.
    NaNs are preserved (callers decide how to handle them).

    Reads these exact columns (for model='flux'):
        role='neg',  img_src='posimg'  →  flux_clip_neg_target_posimg_json
                                          flux_vqa_neg_target_posimg_json
                                          flux_det_neg_target_posimg_json
        …and symmetrically for the other 3 (role, img_src) combos.
    """
    rows: List[List[float]] = []
    for probe in PROBES:
        col = col_probe(model, probe, role, img_src)
        if col not in atom_row.index:
            raise ValueError(f"Column {col!r} missing for atom {atom_row.get(COL_ATOM_ID)!r}.")
        vals = parse_score_list(atom_row[col])
        if len(vals) != expected_len:
            raise ValueError(
                f"Atom {atom_row.get(COL_ATOM_ID)!r} model={model} col={col}: "
                f"expected {expected_len} scores (one per target), got {len(vals)}."
            )
        rows.append(vals)
    return np.asarray(rows, dtype=np.float32)


def robust_atom_score_wide(
    atom_row: pd.Series,
    model: str,
    theta_pos: float,
    theta_neg: float,
    beta: float,
) -> Optional[Dict[str, Any]]:
    """
    Implements the per-atom robust scoring routine (`RobustAtomScore`).

    Returns a dict of (NSI_j, C_j, NegAlign_j, w_j, class_j, S_j, U_j) plus
    diagnostics. Caller is `run_robust_t2i_nsi`, which plays the role of the
    outer robust algorithm and aggregates over j.
    """
    atom_id = atom_row.get(COL_ATOM_ID, "")

    # --- Input unpacking ----------------------------------------------------
    T_neg = parse_concept_list(atom_row.get(COL_NEG_TARGETS_JSON))
    T_pres = parse_concept_list(atom_row.get(COL_PRES_TARGETS_JSON))
    if not T_neg:
        raise ValueError(f"Atom {atom_id!r} has empty {COL_NEG_TARGETS_JSON}.")
    R_j = len(T_neg)                                       # R_j = |T_j^neg|
    K = 1                                                   # benchmark stores 1 (I+,I-) pair per atom

    # Pre-stack per-probe score matrices.
    # Shape (|PROBES|, R_j) for negated concepts, (|PROBES|, |T_pres|) for preserved.
    s_plus_mat  = _load_probe_matrix(atom_row, model, role="neg",  img_src="posimg", expected_len=len(T_neg))
    s_minus_mat = _load_probe_matrix(atom_row, model, role="neg",  img_src="negimg", expected_len=len(T_neg))
    r_plus_mat  = _load_probe_matrix(atom_row, model, role="pres", img_src="posimg", expected_len=len(T_pres))
    r_minus_mat = _load_probe_matrix(atom_row, model, role="pres", img_src="negimg", expected_len=len(T_pres))

    # --- Initialize accumulators --------------------------------------------
    R_plus   = 0.0
    R_minus  = 0.0
    P_minus  = 0.0
    U_accum  = 0.0
    S_accum  = 0.0
    N_P      = 0

    # ========================================================================
    # Single iteration over the (I+, I-) pair (K = 1)
    # ========================================================================
    # (a) Negated-concept loop.
    V: List[float] = []   # per-sample uncertainty set
    for t_idx, concept in enumerate(T_neg):
        s_plus_probes  = s_plus_mat[:,  t_idx]   # s_e^+(t) for e=1..E
        s_minus_probes = s_minus_mat[:, t_idx]
        if np.isnan(s_plus_probes).any() or np.isnan(s_minus_probes).any():
            raise ValueError(
                f"NaN probe score for atom {atom_id!r} model={model} neg_target='{concept}'."
            )
        R_plus  += float(np.median(s_plus_probes))
        R_minus += float(np.median(s_minus_probes))
        V.append(float(np.var(s_plus_probes)))
        V.append(float(np.var(s_minus_probes)))
    u_k = max(V) if V else 0.0
    U_accum += u_k

    # (b) Preservation loop — gated by positive-side presence.
    for t_idx, concept in enumerate(T_pres):
        r_plus_probes = r_plus_mat[:, t_idx]             # r_e^+(t) for e=1..E
        if np.isnan(r_plus_probes).any():
            continue                                     # probe missing → skip (conservative)
        r_bar_plus = float(np.median(r_plus_probes))
        if r_bar_plus < theta_pos:                       # gate: skip iff below threshold
            continue
        r_minus_probes = r_minus_mat[:, t_idx]
        if np.isnan(r_minus_probes).any():
            continue
        P_minus += float(np.median(r_minus_probes))
        N_P     += 1

    # (c) Scene similarity. Stored precomputed as a scalar.
    S_accum += _scalar_or_nan(atom_row.get(col_imgpair_sim(model)))

    # ========================================================================
    # Post-loop normalization
    # ========================================================================
    R_plus  /= (K * R_j)
    R_minus /= (K * R_j)
    U_j      = U_accum / K
    S_j      = S_accum / K
    if N_P > 0:
        P_minus /= N_P
    else:
        P_minus = 1.0                                    # nothing to preserve

    # ========================================================================
    # Atom-level outputs
    # ========================================================================
    NSI_j      = R_plus - R_minus
    C_j        = R_plus
    w_j        = math.exp(-beta * U_j)
    NegAlign_j = 0.5 * (C_j + (1.0 - R_minus)) * P_minus
    class_j    = classify_failure_mode(C_j, R_minus, theta_pos, theta_neg)

    # Preserve tau_form alias: paper symbol is "tau_form", CSV column is "negation_class".
    tau_form = atom_row.get("tau_form")
    if tau_form is None or (isinstance(tau_form, float) and math.isnan(tau_form)):
        tau_form = atom_row.get(COL_NEGATION_CLASS, "")

    return {
        "atom_id":           atom_id,
        "model_name":        model,
        "negation_class":    atom_row.get(COL_NEGATION_CLASS, ""),
        "tau_form":          tau_form,
        "complexity_level":  atom_row.get(COL_COMPLEXITY, ""),
        "target_type":       atom_row.get(COL_TARGET_TYPE, ""),
        "object":            atom_row.get(COL_OBJECT, ""),
        "positive_prompt":   atom_row.get(COL_POS_PROMPT, ""),
        "negative_prompt":   atom_row.get(COL_NEG_PROMPT, ""),
        "neg_targets_json":  atom_row.get(COL_NEG_TARGETS_JSON, "[]"),
        "pres_targets_json": atom_row.get(COL_PRES_TARGETS_JSON, "[]"),
        # Paper symbols:
        "R_plus":            float(R_plus),              # R_j^+
        "R_minus":           float(R_minus),             # R_j^-
        "P_minus":           float(P_minus),             # P_j^-
        "NSI_j":             float(NSI_j),               # NSI_j
        "C_j":               float(C_j),                 # C_j
        "NegAlign_j":        float(NegAlign_j),          # NegAlign_j
        "w_j":               float(w_j),                 # w_j
        "class_j":           class_j,                    # class_j
        "S_j":               float(S_j),                 # S_j
        "U_j":               float(U_j),                 # U_j
        # Counts for audit trails:
        "num_neg_targets":       int(len(T_neg)),
        "num_pres_targets":      int(len(T_pres)),
        "num_samples":           int(K),
        "num_pres_pairs_used":   int(N_P),
    }


# -----------------------------------------------------------------------------
# Robust algorithm — global aggregation over atoms
# -----------------------------------------------------------------------------

def _weighted_mean(group: pd.DataFrame, value_col: str, weight_col: str) -> float:
    w_sum = group[weight_col].sum()
    if w_sum <= 0:
        return float("nan")
    return float((group[value_col] * group[weight_col]).sum() / w_sum)


def _group_weighted_mean_by_group(df: pd.DataFrame, group_col: str, value_col: str, weight_col: str) -> Dict[str, float]:
    if group_col not in df.columns:
        return {}
    out = {}
    for key, gdf in df.dropna(subset=[group_col]).groupby(group_col):
        out[str(key)] = _weighted_mean(gdf, value_col, weight_col)
    return out


def _group_weighted_mean_multi(df: pd.DataFrame, group_cols: List[str], value_col: str, weight_col: str) -> Dict[str, float]:
    if any(col not in df.columns for col in group_cols):
        return {}
    out = {}
    for key, gdf in df.dropna(subset=group_cols).groupby(group_cols):
        w_sum = gdf[weight_col].sum()
        if w_sum > 0:
            label = " | ".join(map(str, key if isinstance(key, tuple) else (key,)))
            out[label] = float((gdf[value_col] * gdf[weight_col]).sum() / w_sum)
    return out


def build_summary(
    results_df: pd.DataFrame,
    coverage_threshold: float,
    theta_pos: float,
    theta_neg: float,
    beta: float,
) -> Dict[str, Any]:
    """
    Implements the post-loop step of the robust algorithm:
      - J = { j | C_j >= xi }
      - NSI(M) = sum_{j in J} w_j NSI_j / sum_{j in J} w_j
      - conditioned scores use { NSI_j, w_j } and per-atom metadata
      - coverage-filtered scores + failure-mode frequencies
    """
    summary: Dict[str, Any] = {
        "num_atoms":          int(len(results_df)),
        "coverage_threshold": float(coverage_threshold),   # xi
        "theta_pos":          float(theta_pos),
        "theta_neg":          float(theta_neg),
        "beta":               float(beta),
        "NSI(M)":             None,
        "S_cov":              [],
        "S_align":            [],
        "S_w":                [],
        "S_class":            [],
        "S_sim":              [],
        "S_unc":              [],
        "global":             {},
        "conditioned_scores": {},
        "failure_mode_distribution": {},
    }
    if results_df.empty:
        return summary

    # Per-atom diagnostic lists (robust algorithm accumulators).
    summary["S_cov"]   = [float(x) for x in results_df["C_j"].tolist()]
    summary["S_align"] = [float(x) for x in results_df["NegAlign_j"].tolist()]
    summary["S_w"]     = [float(x) for x in results_df["w_j"].tolist()]
    summary["S_class"] = results_df["class_j"].astype(str).tolist()
    summary["S_sim"]   = [float(x) for x in results_df["S_j"].tolist()]
    summary["S_unc"]   = [float(x) for x in results_df["U_j"].tolist()]

    # J = { j | C_j >= xi }
    covered = results_df[results_df["C_j"] >= coverage_threshold].copy()
    # NSI(M) as weighted mean over J.
    if not covered.empty and covered["w_j"].sum() > 0:
        summary["NSI(M)"] = float(
            (covered["NSI_j"] * covered["w_j"]).sum() / covered["w_j"].sum()
        )

    summary["global"] = {
        "weighted_nsi_cov_ge_threshold": summary["NSI(M)"],
        "coverage_mean":     float(results_df["C_j"].mean()),
        "negalign_mean":     float(results_df["NegAlign_j"].mean()),
        "weight_mean":       float(results_df["w_j"].mean()),
        "similarity_mean":   float(results_df["S_j"].mean()),
        "uncertainty_mean":  float(results_df["U_j"].mean()),
        "num_atoms_cov_ge_threshold": int(len(covered)),
    }

    # Conditioned scores — NSI stratified by tau_target / tau_form / ell.
    if not covered.empty:
        summary["conditioned_scores"] = {
            "NSI_tau_target": _group_weighted_mean_by_group(covered, "target_type",      "NSI_j", "w_j"),
            "NSI_tau_form":   _group_weighted_mean_by_group(covered, "tau_form",         "NSI_j", "w_j"),
            "NSI_l":          _group_weighted_mean_by_group(covered, "complexity_level", "NSI_j", "w_j"),
            "NSI_tau_target_tau_form_l": _group_weighted_mean_multi(
                covered,
                ["target_type", "tau_form", "complexity_level"],
                "NSI_j",
                "w_j",
            ),
            "NSI_cov_ge_threshold": summary["NSI(M)"],
        }

    # Failure-mode distribution.
    class_freq = results_df["class_j"].value_counts(normalize=True)
    for mode in FAILURE_MODES:
        summary["failure_mode_distribution"][mode] = float(class_freq.get(mode, 0.0))

    return summary


# -----------------------------------------------------------------------------
# Driver — plays the role of the robust algorithm's outer for-loop over j = 1..J.
# -----------------------------------------------------------------------------

def _models_present(df: pd.DataFrame) -> List[str]:
    """Infer available model names from '<model>_clip_imgpair_sim' columns
    (S_j must exist to be scorable)."""
    suffix = "_clip_imgpair_sim"
    return sorted({c[: -len(suffix)] for c in df.columns if c.endswith(suffix)})


def run_robust_t2i_nsi(
    meta_csv: Path,
    out_csv: Path,
    coverage_threshold: float,
    theta_pos: float,
    theta_neg: float,
    beta: float,
    model_names: List[str],
) -> Dict[str, Any]:
    df = pd.read_csv(meta_csv)
    print(f"[NSI/robust] Loaded {len(df)} rows from {meta_csv}")

    # tau_form is stored under CSV column `negation_class`; alias it.
    if "tau_form" not in df.columns and COL_NEGATION_CLASS in df.columns:
        df["tau_form"] = df[COL_NEGATION_CLASS]

    available = _models_present(df)
    if not available:
        raise ValueError(
            f"No model columns found. Expected columns like '<model>{col_imgpair_sim('')}'."
        )

    requested = model_names
    if "all" in requested:
        requested = available
    else:
        unknown = [m for m in requested if m not in available]
        if unknown:
            raise ValueError(f"Requested models {unknown} not found in CSV. Available: {available}")
    print(f"[NSI/robust] Models to score: {requested} (available: {available})")

    base_out_csv = Path(out_csv)
    combined: Dict[str, Any] = {"meta_csv": str(meta_csv), "models": {}}

    # for j = 1 to J — but here we loop over models first (one NSI(M) per
    # model), then over atoms j.
    for model_name in requested:
        atom_results: List[Dict[str, Any]] = []
        for _, atom_row in df.iterrows():
            try:
                result = robust_atom_score_wide(
                    atom_row=atom_row,
                    model=model_name,
                    theta_pos=theta_pos,
                    theta_neg=theta_neg,
                    beta=beta,
                )
                if result is not None:
                    atom_results.append(result)
            except Exception as exc:
                print(
                    f"[NSI/robust][WARN] model={model_name} "
                    f"atom_id={atom_row.get(COL_ATOM_ID)!r}: {exc}"
                )

        results_df = pd.DataFrame(atom_results)
        if results_df.empty:
            print(f"[NSI/robust][WARN] No successful atoms for model '{model_name}'. Skipping.")
            continue

        model_out_csv = base_out_csv.with_name(f"{base_out_csv.stem}_{model_name}{base_out_csv.suffix}")
        results_df.to_csv(model_out_csv, index=False)
        print(f"[NSI/robust] Saved per-atom metrics to {model_out_csv}")

        summary = build_summary(results_df, coverage_threshold, theta_pos, theta_neg, beta)
        summary["model_name"] = model_name
        summary["meta_csv"] = str(meta_csv)
        summary["per_atom_metrics_csv"] = str(model_out_csv)
        summary_path = model_out_csv.with_name(f"{model_out_csv.stem}_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[NSI/robust] Saved summary metrics to {summary_path}")

        # Headline block — parallel to basic_algo.py.
        global_stats = summary["global"]
        nsi_m = summary["NSI(M)"]
        nsi_m_str = f"{nsi_m:.4f}" if nsi_m is not None else "n/a"
        print("\n=== Global T2I-NSI (robust algorithm, paired, K=1) ===")
        print(f"Model:                  {model_name}")
        print(f"NSI(M):                 {nsi_m_str}   (weighted, C_j >= {coverage_threshold:.2f})")
        print(f"Coverage:               {global_stats['coverage_mean']:.4f}")
        print(f"NegAlign:               {global_stats['negalign_mean']:.4f}")
        print(f"Weight:                 {global_stats['weight_mean']:.4f}")
        print(f"Similarity:             {global_stats['similarity_mean']:.4f}")
        print(f"Uncertainty:            {global_stats['uncertainty_mean']:.4f}")

        cs = summary.get("conditioned_scores", {})
        if cs:
            print("\n--- NSI_tau_target ---")
            print(pd.Series(cs.get("NSI_tau_target", {})))
            print("\n--- NSI_tau_form ---")
            print(pd.Series(cs.get("NSI_tau_form", {})))
            print("\n--- NSI_l ---")
            print(pd.Series(cs.get("NSI_l", {})))
            print("\n--- NSI_tau_target | tau_form | l ---")
            print(pd.Series(cs.get("NSI_tau_target_tau_form_l", {})))

        combined["models"][model_name] = {
            "model_name": model_name,
            "per_atom_metrics_csv": str(model_out_csv),
            "summary_json": str(summary_path),
            "summary": summary,
        }

    if len(requested) > 1:
        combined_summary_path = base_out_csv.with_name(f"{base_out_csv.stem}_all_models_summary.json")
        combined_summary_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        print(f"\n[NSI/robust] Saved combined multi-model summary to {combined_summary_path}")
        combined["combined_summary_json"] = str(combined_summary_path)
    return combined


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robust T2I-NSI (probe-ensemble algorithm) over a wide-form per-atom CSV."
    )
    parser.add_argument("--meta_csv", type=str, required=True,
                        help="Wide-form atom CSV with pre-computed probe scores.")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="Base path for per-model robust outputs.")
    parser.add_argument("--coverage_threshold", type=float, default=0.3,
                        help="xi (paper default: 0.3).")
    parser.add_argument("--theta_pos", type=float, default=0.7,
                        help="theta_pos (paper default: 0.7).")
    parser.add_argument("--theta_neg", type=float, default=0.3,
                        help="theta_neg (paper default: 0.3).")
    parser.add_argument("--beta", type=float, default=2.0,
                        help="beta (paper default: 2.0).")
    parser.add_argument(
        "--model_name",
        nargs="+",
        default=["all"],
        help="Model names to score (e.g. flux qwen stable-diffusion) or 'all' to score "
             "every model inferred from '<model>_clip_imgpair_sim' columns.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_robust_t2i_nsi(
        meta_csv=Path(args.meta_csv),
        out_csv=Path(args.out_csv),
        coverage_threshold=args.coverage_threshold,
        theta_pos=args.theta_pos,
        theta_neg=args.theta_neg,
        beta=args.beta,
        model_names=args.model_name,
    )
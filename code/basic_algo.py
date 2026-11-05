"""
Basic T2I-NSI — implementation of the basic algorithm.

Computes the Negation Sensitivity Index, coverage, and preservation-aware
NegAlign over a wide-form per-atom CSV of pre-computed CLIP scores. One
row per atom; image generation and probe scoring are assumed to have
already happened upstream.

================================================================================
Expected input CSV columns (per atom)
================================================================================

Atom metadata:
    atom_id              — primary key, e.g. "SYN_0_L0_01"
    negation_class       — one of {syntactic, lexical, morphological, ...}
    complexity_level     — one of {L0, L1, L2, L3, L4}
    target_type          — one of {object, attribute, relation, composition,
                                   nested}
    object               — high-level subject (diagnostic only)
    neg_targets_json     — JSON list of T_j^neg concept anchors
    pres_targets_json    — JSON list of T_j^pres concept anchors

Per-model image-path columns (for at least one model m):
    {m}_positive_path, {m}_negative_path
    {m}_positive_seed,  {m}_negative_seed

Per-model pre-computed CLIP score columns (each a JSON list of raw cosine
similarities in [-1, 1], in the same order as the corresponding target list:
    {m}_clip_neg_target_posimg_json   — g(I+, t) for t in T_j^neg
    {m}_clip_neg_target_negimg_json   — g(I-, t) for t in T_j^neg
    {m}_clip_pres_target_posimg_json  — g(I+, t) for t in T_j^pres
    {m}_clip_pres_target_negimg_json  — g(I-, t) for t in T_j^pres
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

def _parse_score_list(value) -> List[float]:
    """Parse a JSON list of floats from a cell, mapping cos-sim to [0,1]."""
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    try:
        raw = json.loads(s)
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    return [(float(v) + 1.0) / 2.0 for v in raw]

def parse_concept_list(value) -> List[str]:
    """
    Turn a serialized concept field into a list of concept anchors.
    Supported forms:
    - empty / NaN -> []
    - JSON list string -> parsed list
    - semicolon-separated string -> split on ';'
    - plain string -> single-item list
    """
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON concept list: {s}") from exc
        if not isinstance(parsed, list):
            raise ValueError(f"Concept field must decode to a list, got: {type(parsed).__name__}")
        return [str(item).strip() for item in parsed if str(item).strip()]
    if ";" in s:
        return [p.strip() for p in s.split(";") if p.strip()]
    return [s]


def _validate_atom_metadata(atom_id, gdf: pd.DataFrame, columns: List[str]) -> Dict[str, str]:
    metadata = {}
    for col in columns:
        if col not in gdf.columns:
            metadata[col] = None
            continue
        non_null = gdf[col].dropna().astype(str).str.strip().unique()
        if len(non_null) > 1:
            raise ValueError(
                f"Atom {atom_id} has inconsistent values for '{col}': {list(non_null)}"
            )
        metadata[col] = gdf[col].iloc[0]
    return metadata


def _validate_paired_rows(atom_id, gdf: pd.DataFrame, pos_col: str, neg_col: str) -> None:
    missing_pos = gdf[pos_col].isna().sum()
    missing_neg = gdf[neg_col].isna().sum()
    if missing_pos or missing_neg:
        raise ValueError(
            f"Atom {atom_id} has missing image paths "
            f"(pos missing={missing_pos}, neg missing={missing_neg})."
        )

    if "seed" in gdf.columns:
        duplicated = gdf["seed"].duplicated().any()
        if duplicated:
            raise ValueError(f"Atom {atom_id} has duplicate seeds; expected one paired row per seed.")


def _group_mean(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, float]:
    if group_col not in df.columns:
        return {}
    grouped = (
        df.dropna(subset=[group_col])
        .groupby(group_col)[value_col]
        .mean()
        .sort_index()
    )
    return {str(k): float(v) for k, v in grouped.items()}


def _group_mean_multi(df: pd.DataFrame, group_cols: List[str], value_col: str) -> Dict[str, float]:
    present_cols = [col for col in group_cols if col in df.columns]
    if len(present_cols) != len(group_cols):
        return {}
    grouped = (
        df.dropna(subset=present_cols)
        .groupby(present_cols)[value_col]
        .mean()
        .sort_index()
    )
    return {" | ".join(map(str, k if isinstance(k, tuple) else (k,))): float(v) for k, v in grouped.items()}


def build_summary(metrics_df: pd.DataFrame, coverage_threshold: float) -> Dict[str, object]:
    summary = {
        "num_atoms": int(len(metrics_df)),
        "coverage_threshold": float(coverage_threshold),
        "NSI(M)": None,
        "S_cov": [],
        "S_align": [],
        "global": {},
        "conditioned_scores": {},
    }

    if metrics_df.empty:
        return summary

    covered_df = metrics_df[metrics_df["C_j"] >= coverage_threshold]
    # Global T2I-NSI: NSI(M) = mean_j NSI_j
    # Intuition: across the whole benchmark, how much more present are target concepts
    # in positive images than in their negated counterparts?
    summary["NSI(M)"] = float(metrics_df["NSI_j"].mean())
    # Coverage list S_cov = {C_j}_j and preservation-aware list S_align = {NegAlign_j}_j
    # S_cov stores, per atom, whether the model can realize the target concept at all
    # under the positive prompt. S_align stores, per atom, whether the model both
    # suppresses the negated target and keeps the rest of the scene intact.
    summary["S_cov"] = [float(x) for x in metrics_df["C_j"].tolist()]
    summary["S_align"] = [float(x) for x in metrics_df["NegAlign_j"].tolist()]
    summary["global"] = {
        "nsi_mean": float(metrics_df["NSI_j"].mean()),
        "coverage_mean": float(metrics_df["C_j"].mean()),
        "negalign_mean": float(metrics_df["NegAlign_j"].mean()),
        "nsi_cov_ge_threshold_mean": float(covered_df["NSI_j"].mean()) if not covered_df.empty else None,
        "num_atoms_cov_ge_threshold": int(len(covered_df)),
    }

    summary["conditioned_scores"] = {
        # NSI_{tau^target}: mean NSI_j within each target-type bucket
        # This tells us whether the model handles object, attribute, or relation
        # negation differently.
        "NSI_tau_target": _group_mean(metrics_df, "target_type", "NSI_j"),
        # NSI_{tau^form}: mean NSI_j within each negation-form bucket
        # This reveals which linguistic form of negation the model struggles with:
        # lexical, morphological, syntactic, etc.
        "NSI_tau_form": _group_mean(metrics_df, "tau_form", "NSI_j"),
        # NSI_l: mean NSI_j within each complexity-level bucket
        # This shows how negation performance changes from simple atoms (L0/L1)
        # to harder compositional or nested atoms (L3/L4).
        "NSI_l": _group_mean(metrics_df, "complexity_level", "NSI_j"),
        # Joint conditioned score NSI_{tau^target, tau^form, l}
        # This is the most fine-grained profile: for example, attribute negation,
        # expressed lexically, at complexity L1.
        "NSI_tau_target_tau_form_l": _group_mean_multi(
            metrics_df,
            ["target_type", "tau_form", "complexity_level"],
            "NSI_j",
        ),
        # Coverage-filtered global score NSI_{cov >= xi}
        # If coverage is too low, the atom is unreliable because the model never
        # learned to render the target clearly even in the positive condition.
        "NSI_cov_ge_threshold": float(covered_df["NSI_j"].mean()) if not covered_df.empty else None,
    }
    return summary


def resolve_model_columns(df: pd.DataFrame, model_name: str) -> Tuple[pd.DataFrame, str, str, str]:
    prefix = model_name.lower()
    pos_col = f"{prefix}_positive_path"
    neg_col = f"{prefix}_negative_path"
    pos_seed_col = f"{prefix}_positive_seed"
    neg_seed_col = f"{prefix}_negative_seed"

    required = [pos_col, neg_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Metadata CSV missing model-specific columns for model='{model_name}': {missing}"
        )

    if pos_seed_col in df.columns and neg_seed_col in df.columns:
        mismatched = (df[pos_seed_col].astype(str) != df[neg_seed_col].astype(str)) & (
            df[pos_seed_col].notna() & df[neg_seed_col].notna()
        )
        if mismatched.any():
            mismatch_count = int(mismatched.sum())
            print(
                f"[NSI/basic][WARN] Skipping {mismatch_count} rows with mismatched "
                f"positive/negative seeds for model='{model_name}'."
            )
            df = df.loc[~mismatched].copy()
        seed_col = pos_seed_col
    else:
        seed_col = ""

    return df, pos_col, neg_col, seed_col


def resolve_clip_score_columns(model_name: str) -> Tuple[str, str, str, str]:
    """Return the four pre-computed CLIP score column names for a model."""
    prefix = model_name.lower()
    return (
        f"{prefix}_clip_neg_target_posimg_json",    # g(I+, t) for t in T^neg
        f"{prefix}_clip_neg_target_negimg_json",    # g(I-, t) for t in T^neg
        f"{prefix}_clip_pres_target_posimg_json",   # g(I+, t) for t in T^pres (unused but available)
        f"{prefix}_clip_pres_target_negimg_json",   # g(I-, t) for t in T^pres
    )


def run_t2i_nsi_basic(meta_csv: Path, out_csv: Path, coverage_threshold: float, model_name: str) -> Dict[str, object]:
    df = pd.read_csv(meta_csv)
    print(f"[NSI/basic] Loaded {len(df)} rows from {meta_csv}")

    df, pos_col, neg_col, seed_col = resolve_model_columns(df, model_name)
    df = df.copy()
    # No need to resolve image paths — we use pre-computed CLIP scores, not actual images.
    if seed_col:
        df["seed"] = df[seed_col]
    if "tau_form" not in df.columns and "negation_class" in df.columns:
        df["tau_form"] = df["negation_class"]

    # Resolve pre-computed CLIP score columns for this model
    neg_posimg_col, neg_negimg_col, _pres_posimg_col, pres_negimg_col = resolve_clip_score_columns(model_name)

    # Validate that the required score columns exist
    required_score_cols = [neg_posimg_col, neg_negimg_col, pres_negimg_col]
    missing_score_cols = [c for c in required_score_cols if c not in df.columns]
    if missing_score_cols:
        raise ValueError(
            f"Metadata CSV missing pre-computed CLIP score columns for model='{model_name}': {missing_score_cols}"
        )

    atom_metrics = []

    # One group = one atom a_j, with K rows (paired pos/neg images)
    groups = df.groupby("atom_id")

    for atom_id, gdf in tqdm(groups, desc="Atoms"):
        _validate_paired_rows(atom_id, gdf, pos_col, neg_col)
        metadata = _validate_atom_metadata(
            atom_id,
            gdf,
            [
                "negation_class",
                "tau_form",
                "complexity_level",
                "target_type",
                "object",
                "neg_targets_json",
                "pres_targets_json",
                "notes",
            ],
        )

        negation_class = metadata["negation_class"]
        tau_form = metadata["tau_form"]
        complexity_level = metadata["complexity_level"]
        target_type = metadata["target_type"]
        obj = metadata["object"]
        neg_targets_json = metadata["neg_targets_json"]
        pres_targets_json = metadata["pres_targets_json"]
        notes = metadata["notes"]

        # Build T_j^neg and T_j^pres (needed for counts)
        neg_targets = parse_concept_list(neg_targets_json)          # T_j^{neg}
        pres_targets = parse_concept_list(pres_targets_json)        # T_j^{pres}
        R_j = len(neg_targets)
        if R_j == 0:
            print(f"[NSI/basic][WARN] Atom {atom_id} has empty T_j^neg (neg_targets_json). Skipping.")
            continue

        # Number of samples per prompt K (should equal to your K in image gen)
        K = len(gdf)

        # Accumulators (basic algorithm)
        R_j_plus = 0.0   # sum over K * |T_j^{neg}| for positive images
        R_j_minus = 0.0  # sum over K * |T_j^{neg}| for negative images
        P_j_minus = 0.0  # sum over K * |T_j^{pres}| for negative images

        # Iterate over each (pos, neg) pair — use pre-computed scores
        for _, row in gdf.iterrows():
            # Read pre-computed CLIP scores (raw cosine sims → mapped to [0,1])
            neg_posimg_scores = _parse_score_list(row.get(neg_posimg_col))
            neg_negimg_scores = _parse_score_list(row.get(neg_negimg_col))
            pres_negimg_scores = _parse_score_list(row.get(pres_negimg_col))

            # Positive: score negation targets under p_j^+
            R_j_plus += sum(neg_posimg_scores)

            # Negative: score negation targets under p_j^-
            R_j_minus += sum(neg_negimg_scores)

            # Negative: score preserved targets under p_j^-
            P_j_minus += sum(pres_negimg_scores)

        # Normalise to expectations
        # R_j^+ and R_j^- are averages over paired samples and negation targets.
        # R_j^+ answers: when we ask for the target concept positively, how strongly
        # does the probe see it? R_j^- answers the same question under negation.
        R_j_plus /= (K * R_j)
        R_j_minus /= (K * R_j)

        if len(pres_targets) > 0 and K > 0:
            # P_j^- averages preservation over samples and preserved targets.
            # A high P_j^- means the model hasn't unnecessarily destroyed the
            # surrounding scene.
            P_j_minus /= (K * len(pres_targets))
        else:
            P_j_minus = 1.0  # by convention when |T_j^{pres}| = 0

        # Atom-level scores: NSI_j, coverage C_j, and preservation-aware NegAlign_j.
        # NSI_j: did negation actually reduce the target concept?
        # C_j: can the model render the target at all in the positive prompt?
        # NegAlign_j: does the model realize the target positively, suppress it
        # negatively, and still preserve the rest of the scene?
        NSI_j = R_j_plus - R_j_minus                # NSI_j \in [-1,1]
        C_j = R_j_plus                              # coverage under positive prompts
        NegAlign_j = 0.5 * (C_j + (1.0 - R_j_minus)) * P_j_minus

        atom_metrics.append({
            "atom_id": atom_id,
            "model_name": model_name,
            "negation_class": negation_class,
            "tau_form": tau_form,
            "complexity_level": complexity_level,
            "target_type": target_type,
            "object": obj,
            "neg_targets_json": neg_targets_json,
            "pres_targets_json": pres_targets_json,
            "notes": notes,
            "R_j_plus": R_j_plus,
            "R_j_minus": R_j_minus,
            "C_j": C_j,
            "P_j_minus": P_j_minus,
            "NSI_j": NSI_j,
            "NegAlign_j": NegAlign_j,
        })

    # -----------------------------------
    # Save per-atom metrics + aggregates
    # -----------------------------------
    metrics_df = pd.DataFrame(atom_metrics)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_csv, index=False)
    print(f"[NSI/basic] Saved per-atom metrics to {out_csv}")

    result = {
        "model_name": model_name,
        "meta_csv": str(meta_csv),
        "per_atom_metrics_csv": str(out_csv),
        "summary_json": None,
        "summary": None,
    }

    if not metrics_df.empty:
        summary = build_summary(metrics_df, coverage_threshold)
        summary["model_name"] = model_name
        summary["meta_csv"] = str(meta_csv)
        summary["per_atom_metrics_csv"] = str(out_csv)
        global_stats = summary["global"]
        summary_path = out_csv.with_name(f"{out_csv.stem}_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[NSI/basic] Saved summary metrics to {summary_path}")
        result["summary_json"] = str(summary_path)
        result["summary"] = summary

        nsi_cov = global_stats["nsi_cov_ge_threshold_mean"]
        nsi_cov_str = f"{nsi_cov:.4f}" if nsi_cov is not None else "n/a"
        print("\n=== Global T2I-NSI (basic algorithm, paired, K=1) ===")
        print(f"Model:                  {model_name}")
        print(f"NSI(M):                 {summary['NSI(M)']:.4f}")
        print(f"Coverage:               {global_stats['coverage_mean']:.4f}")
        print(f"NegAlign:               {global_stats['negalign_mean']:.4f}")
        print(f"NSI(C >= {coverage_threshold:.2f}):         {nsi_cov_str}")

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

    return result


def _models_present(df: pd.DataFrame) -> List[str]:
    """Infer available model names from '<model>_clip_neg_target_posimg_json'
    columns. A model is considered scorable if it has the negated-target
    score columns required by the basic algorithm."""
    suffix = "_clip_neg_target_posimg_json"
    return sorted({c[: -len(suffix)] for c in df.columns if c.endswith(suffix)})


def run_all_models(
    meta_csv: Path,
    out_csv: Path,
    coverage_threshold: float,
    model_names: List[str],
) -> Dict[str, object]:
    """Resolve 'all' against the CSV's model columns, then score each model."""
    df_head = pd.read_csv(meta_csv)
    available = _models_present(df_head)
    if not available:
        raise ValueError(
            f"No model columns found in {meta_csv}. "
            f"Expected columns like '<model>_clip_neg_target_posimg_json'."
        )

    if "all" in model_names:
        requested = available
    else:
        unknown = [m for m in model_names if m not in available]
        if unknown:
            raise ValueError(
                f"Requested models {unknown} not found in CSV. "
                f"Available: {available}"
            )
        requested = model_names

    print(f"[NSI/basic] Models to score: {requested} (available: {available})")

    combined = {
        "meta_csv": str(meta_csv),
        "models": {},
    }

    for model_name in requested:
        model_out_csv = out_csv.with_name(f"{out_csv.stem}_{model_name}{out_csv.suffix}")
        result = run_t2i_nsi_basic(meta_csv, model_out_csv, coverage_threshold, model_name)
        combined["models"][model_name] = result

    if len(requested) > 1:
        combined_summary_path = out_csv.with_name(f"{out_csv.stem}_all_models_summary.json")
        combined_summary_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        print(f"\n[NSI/basic] Saved combined multi-model summary to {combined_summary_path}")
        combined["combined_summary_json"] = str(combined_summary_path)

    return combined


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Basic T2I-NSI over paired pos/neg images, "
                    "using pre-computed CLIP scores.",
    )
    ap.add_argument(
        "--meta_csv",
        type=str,
        required=True,
        help="Wide-form per-atom CSV with pre-computed CLIP score columns.",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="t2i_nsi_basic_results.csv",
        help="Base path for per-model per-atom metric CSVs and summary JSONs.",
    )
    ap.add_argument(
        "--coverage_threshold",
        type=float,
        default=0.3,
        help="Coverage threshold xi for reporting NSI_{cov >= xi} (default: 0.3).",
    )
    ap.add_argument(
        "--model_name",
        type=str,
        required=True,
        nargs="+",
        help="Model name(s) to score, matching the prefix used in the CSV "
             "(e.g. 'flux', 'qwen', 'stable-diffusion', 'gpt', 'gemini', 'midjourney'). "
             "Pass 'all' to score every model inferred from "
             "'<model>_clip_neg_target_posimg_json' columns.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_models(
        Path(args.meta_csv),
        Path(args.out_csv),
        args.coverage_threshold,
        args.model_name,
    )

"""
ablation_study.py
=================
Ablation study comparing multiple prompt strategies for VLM plankton classification:

  Condition 1 (bare):     "Classify this image as one of: [list]"
  Condition 2 (list):     "Identify this plankton from: [list]. Reply with only the name."
  Condition 3 (expert):   "You are a marine biologist. Identify this plankton from: [list]"
  Condition 4 (rag-full): Full morphological catalog injected
  Condition 5 (rag+few):  RAG + 1 few-shot example per class (text description of a typical image)

This isolates WHICH component of the prompt drives the improvement.
"""

import argparse
import base64
import json
import logging
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASSES = [
    "Amphipoda", "Annelida", "Ceratium", "Chaetognatha",
    "Coscinodiscus", "Noctiluca",
]
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ---------------------------------------------------------------------------
# Morphological catalog
# ---------------------------------------------------------------------------
MORPHO_CATALOG = {
    "Coscinodiscus": "A perfectly CIRCULAR disc-shaped organism. Radial symmetry with concentric rings of tiny dots (areolae). No appendages, horns, or tail. Looks like a round coin or button.",
    "Ceratium": "A dinoflagellate with 2-4 distinctive HORN-LIKE projections extending from the body. One horn points forward and 1-3 point backward. The body has a visible groove around the middle.",
    "Chaetognatha": "An arrow-shaped worm with an elongated, torpedo-shaped body. Transparent with visible lateral fins along the sides. Small hooked spines near the mouth. No legs, no horns, no circular shape.",
    "Noctiluca": "A large SPHERICAL or round organism. Single prominent tentacle visible. Translucent with cytoplasm pushed to edges by a large central vacuole. Looks like a glowing ball.",
    "Amphipoda": "A small crustacean with a CURVED, laterally flattened body (like a shrimp curled sideways). Multiple visible legs (7+ pairs). Distinct head with compound eyes. No shell covering.",
    "Annelida": "A SEGMENTED worm with visible body rings. Elongated cylindrical shape. May have small bristle-like appendages along the sides. Worm-like appearance.",
}

FEW_SHOT_EXAMPLES = {
    "Coscinodiscus": "A typical Coscinodiscus image shows a perfectly round, coin-shaped organism with fine concentric rings visible on the surface, often against a dark background.",
    "Ceratium": "A typical Ceratium image shows an organism with 2-4 visible horn-like arms extending from a central body, resembling a tiny antler or star.",
    "Chaetognatha": "A typical Chaetognatha image shows an elongated, transparent, arrow-shaped body with visible internal structures and small fins along the sides.",
    "Noctiluca": "A typical Noctiluca image shows a large, round, bubble-like organism with thin cytoplasm at the edges and a large empty center.",
    "Amphipoda": "A typical Amphipoda image shows a small curved crustacean with visible legs and a distinct head, often curled into a C-shape.",
    "Annelida": "A typical Annelida image shows a long, thin worm-like body with visible segments (rings) and sometimes small bristle-like appendages.",
}


# ---------------------------------------------------------------------------
# Prompt conditions
# ---------------------------------------------------------------------------
def prompt_bare():
    return f"Classify this image as one of: {', '.join(CLASSES)}"


def prompt_list():
    return (
        f"Identify this plankton organism from the following list: {', '.join(CLASSES)}. "
        "Reply with ONLY the organism name. Nothing else."
    )


def prompt_expert():
    return (
        "You are a marine biologist expert at plankton identification. "
        f"Look at this microscope image and identify the organism from: {', '.join(CLASSES)}. "
        "Examine the shape, symmetry, appendages, and surface features carefully. "
        "Reply with ONLY the single best matching organism name."
    )


def prompt_rag_full():
    catalog = "\n".join(f"- {name}: {desc}" for name, desc in MORPHO_CATALOG.items())
    return (
        "You are a marine biologist expert. Examine this microscope image and identify "
        "the organism by matching its morphological features against this catalog:\n\n"
        f"## Morphological Catalog\n{catalog}\n\n"
        "## Instructions\n"
        "1. Examine shape, symmetry, appendages, surface features.\n"
        "2. Compare against EACH catalog entry.\n"
        "3. Select the single best match.\n\n"
        "Reply with ONLY the organism name."
    )


def prompt_rag_fewshot():
    catalog = "\n".join(f"- {name}: {desc}" for name, desc in MORPHO_CATALOG.items())
    examples = "\n".join(f"- {name}: {ex}" for name, ex in FEW_SHOT_EXAMPLES.items())
    return (
        "You are a marine biologist expert. Examine this microscope image and identify "
        "the organism by matching its features against this catalog:\n\n"
        f"## Morphological Catalog\n{catalog}\n\n"
        f"## Typical Image Descriptions\n{examples}\n\n"
        "## Instructions\n"
        "1. Examine the image for diagnostic features.\n"
        "2. Compare against the catalog AND typical image descriptions.\n"
        "3. Select the single best match.\n\n"
        "Reply with ONLY the organism name."
    )


PROMPT_BUILDERS = {
    "bare": prompt_bare,
    "list": prompt_list,
    "expert": prompt_expert,
    "rag_full": prompt_rag_full,
    "rag_fewshot": prompt_rag_fewshot,
}


# ---------------------------------------------------------------------------
# VLM call
# ---------------------------------------------------------------------------
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_vlm(image_path: str, prompt: str, endpoint: str, model: str, timeout: int = 120) -> str:
    b64 = encode_image(image_path)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": 32,
        "temperature": 0.0,
    }
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def parse_class(raw: str) -> str:
    cleaned = raw.strip().strip('"').strip("'").strip(".")
    for cls in CLASSES:
        if cleaned.lower() == cls.lower():
            return cls
    for cls in CLASSES:
        if cls.lower() in cleaned.lower():
            return cls
    return cleaned


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_samples(data_root: str, domain: str) -> list:
    samples = []
    domain_dir = Path(data_root) / domain
    for cls in CLASSES:
        cls_dir = domain_dir / cls
        if not cls_dir.is_dir():
            continue
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in SUPPORTED_EXT:
                samples.append({"path": str(img), "true_label": cls, "domain": domain})
    return samples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def run_condition(samples: list, condition: str, endpoint: str, model: str) -> dict:
    prompt_fn = PROMPT_BUILDERS[condition]
    prompt = prompt_fn()
    results = []
    for i, s in enumerate(samples):
        try:
            raw = call_vlm(s["path"], prompt, endpoint, model)
        except Exception as e:
            logger.error("Failed %s: %s", s["path"], e)
            raw = ""
        pred = parse_class(raw)
        correct = pred.lower() == s["true_label"].lower()
        results.append({
            "path": s["path"], "true_label": s["true_label"], "domain": s["domain"],
            "predicted": pred, "correct": correct, "raw_response": raw,
        })
        if (i + 1) % 20 == 0 or (i + 1) == len(samples):
            acc = sum(1 for r in results if r["correct"]) / len(results)
            logger.info("  [%s] %d/%d  acc=%.1f%%", condition, i + 1, len(samples), acc * 100)
    return {"condition": condition, "results": results}


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_ci(binary_list, n=1000, ci=0.95):
    rng = np.random.RandomState(42)
    arr = np.array(binary_list)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    alpha = (1 - ci) / 2
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(boots, alpha * 100)),
        "ci_high": float(np.percentile(boots, (1 - alpha) * 100)),
    }


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------
def mcnemar_test(results_a, results_b):
    """McNemar's test comparing two conditions."""
    a_correct = [r["correct"] for r in results_a]
    b_correct = [r["correct"] for r in results_b]
    # b01: A wrong, B right; b10: A right, B wrong
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    n = b01 + b10
    if n == 0:
        return {"statistic": 0, "p_value": 1.0, "b01": 0, "b10": 0}
    # McNemar chi-squared with continuity correction
    stat = (abs(b01 - b10) - 1) ** 2 / n
    # p-value from chi-squared distribution with 1 df
    from scipy import stats
    p = 1 - stats.chi2.cdf(stat, df=1)
    return {"statistic": float(stat), "p_value": float(p), "b01": b01, "b10": b10}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ablation study on prompt strategies.")
    parser.add_argument("--data-root", type=str, default="data/eval")
    parser.add_argument("--conditions", nargs="+", default=list(PROMPT_BUILDERS.keys()))
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", type=str,
                        default="/models/models--Qwen--Qwen2.5-VL-32B-Instruct-AWQ/snapshots/66c370b74a18e7b1e871c97918f032ed3578dfef")
    parser.add_argument("--output", type=str, default="results/ablation.json")
    args = parser.parse_args()

    # Load all samples
    all_samples = load_samples(args.data_root, "IFCB") + load_samples(args.data_root, "ZooScan")
    logger.info("Loaded %d total samples (%d IFCB, %d ZooScan)",
                len(all_samples),
                sum(1 for s in all_samples if s["domain"] == "IFCB"),
                sum(1 for s in all_samples if s["domain"] == "ZooScan"))

    # Run each condition
    all_results = {}
    for cond in args.conditions:
        logger.info("Running condition: %s", cond)
        t0 = time.time()
        all_results[cond] = run_condition(all_samples, cond, args.endpoint, args.model)
        elapsed = time.time() - t0
        logger.info("  Completed in %.1fs", elapsed)

    # Compute summary statistics
    summary = {}
    for cond, data in all_results.items():
        results = data["results"]
        summary[cond] = {}
        for domain in ["IFCB", "ZooScan"]:
            domain_r = [r for r in results if r["domain"] == domain]
            binary = [1 if r["correct"] else 0 for r in domain_r]
            ci = bootstrap_ci(binary)
            per_class = {}
            for cls in CLASSES:
                cls_r = [r for r in domain_r if r["true_label"] == cls]
                if cls_r:
                    per_class[cls] = sum(1 for r in cls_r if r["correct"]) / len(cls_r)
            summary[cond][domain] = {
                "accuracy": ci["mean"],
                "ci_95": [ci["ci_low"], ci["ci_high"]],
                "n": len(domain_r),
                "per_class": per_class,
            }

    # Print summary table
    logger.info("=" * 90)
    logger.info("  ABLATION STUDY RESULTS")
    logger.info("=" * 90)
    header = f"{'Condition':<14} {'IFCB Acc':>10} {'95% CI':>16} {'ZS Acc':>10} {'95% CI':>16} {'Gap':>8}"
    logger.info(header)
    logger.info("-" * 90)
    for cond in args.conditions:
        ifcb = summary[cond]["IFCB"]
        zs = summary[cond]["ZooScan"]
        gap = ifcb["accuracy"] - zs["accuracy"]
        logger.info("%-14s %9.1f%% [%5.1f-%5.1f%%] %9.1f%% [%5.1f-%5.1f%%] %+7.1f%%",
                    cond,
                    ifcb["accuracy"] * 100, ifcb["ci_95"][0] * 100, ifcb["ci_95"][1] * 100,
                    zs["accuracy"] * 100, zs["ci_95"][0] * 100, zs["ci_95"][1] * 100,
                    gap * 100)
    logger.info("=" * 90)

    # McNemar's tests: compare each condition to "list" baseline
    if "list" in all_results:
        logger.info("\nMcNemar's test vs 'list' baseline:")
        for cond in args.conditions:
            if cond == "list":
                continue
            for domain in ["IFCB", "ZooScan"]:
                base_r = [r for r in all_results["list"]["results"] if r["domain"] == domain]
                cond_r = [r for r in all_results[cond]["results"] if r["domain"] == domain]
                if len(base_r) == len(cond_r):
                    test = mcnemar_test(base_r, cond_r)
                    sig = "***" if test["p_value"] < 0.001 else "**" if test["p_value"] < 0.01 else "*" if test["p_value"] < 0.05 else "ns"
                    logger.info("  %s vs list on %s: chi2=%.2f  p=%.4f  %s  (b01=%d, b10=%d)",
                                cond, domain, test["statistic"], test["p_value"], sig, test["b01"], test["b10"])

    # Save
    report = {
        "conditions": args.conditions,
        "classes": CLASSES,
        "summary": summary,
        "per_sample": {cond: data["results"] for cond, data in all_results.items()},
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Full report saved to %s", args.output)


if __name__ == "__main__":
    main()

"""
artifact_aware_rag.py
=====================
Artifact-Aware RAG (AA-RAG) for cross-domain plankton classification.

Extends the morphological catalog from the DataShift project with:
1. An imaging artifact catalog describing domain-specific visual artifacts
2. Shift-informed prompting that tells the VLM to ignore artifacts
3. Cross-domain reference image retrieval

Usage:
    python artifact_aware_rag.py \
        --data-dir data/cross_domain/cross_instrument \
        --endpoint http://localhost:8000/v1/chat/completions \
        --model Qwen2.5-VL-32B-Instruct-AWQ \
        --output results/aa_rag_results.json
"""

import argparse
import base64
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ---------------------------------------------------------------------------
# Morphological Catalog (from DataShift project)
# ---------------------------------------------------------------------------
MORPHOLOGICAL_CATALOG = {
    "Amphipoda": "A small crustacean with a CURVED, laterally flattened body (like a shrimp curled sideways). Multiple visible legs (7+ pairs). Distinct head with compound eyes. No shell or carapace covering the body.",
    "Annelida": "A SEGMENTED worm with visible body rings. Elongated cylindrical shape. May have small bristle-like appendages (parapodia) along the sides. Worm-like appearance.",
    "Calanoida": "A copepod with an elongated, torpedo-shaped body divided into prosome and urosome. Long first antennae. Single red eye at the front. Often has visible egg sacs.",
    "Ceratium": "A dinoflagellate with 2-4 distinctive HORN-LIKE projections extending from the body. One horn forward, 1-3 backward. Visible groove (cingulum) around the middle.",
    "Chaetognatha": "An arrow-shaped worm with elongated, torpedo-shaped body. Transparent with lateral fins. Small hooked spines near mouth. No legs, no horns. Smooth, streamlined.",
    "Oithonidae": "A small copepod with compact, oval body. Short first antennae. Raptorial second antennae. Often carries egg sacs ventrally.",
    "Appendicularia": "A free-swimming tunicate with an oval trunk and a long tail. Transparent gelatinous body. Tail used for filter feeding via mucous house.",
    "Coscinodiscus": "A perfectly CIRCULAR disc-shaped organism. Radial symmetry with concentric rings of tiny dots (areolae). No appendages, horns, or tail.",
    "Noctiluca": "A large SPHERICAL or round organism. Single prominent tentacle visible. Translucent with cytoplasm pushed to edges by a large central vacuole.",
    "Doliolida": "A barrel-shaped tunicate with transparent gelatinous body. Band-like muscle bands visible around the body. Free-swimming.",
    "Ostracoda": "A small crustacean enclosed in a bivalve shell (like a tiny clam). Two valves visible from the side. Short appendages may protrude.",
    "Salpida": "A free-swimming tunicate with barrel-shaped, transparent gelatinous body. Often found in chains. Muscle bands visible.",
}

# ---------------------------------------------------------------------------
# Artifact Catalog (NEW — domain-specific imaging artifacts)
# ---------------------------------------------------------------------------
ARTIFACT_CATALOG = {
    "IFCB": {
        "instrument": "Imaging FlowCytobot (IFCB) — dark-field flow-through cytometer",
        "artifacts": [
            "Dark/black background from dark-field illumination",
            "High contrast between organism and background",
            "Potential motion blur at edges from flow-through capture",
            "Consistent illumination across the image",
            "Organism centered in field of view",
            "No color information (grayscale or false-color)",
            "Possible bubble artifacts or debris in flow cell",
        ],
        "ignore": "background darkness, contrast levels, edge sharpness",
    },
    "ZooScan": {
        "instrument": "ZooScan — flatbed scanner for preserved zooplankton",
        "artifacts": [
            "Variable background brightness (gray to white)",
            "Flatbed scan artifacts: potential reflections, shadows",
            "Lower contrast than IFCB",
            "Possible scanner noise or banding artifacts",
            "Organisms may be at various positions (not centered)",
            "Grayscale images with variable lighting",
            "Preserved organisms may appear different from live",
        ],
        "ignore": "background brightness, scanner artifacts, contrast levels, organism position",
    },
    "ZooLake35": {
        "instrument": "Digital Scientific Plankton Camera (DSPC) — field camera",
        "artifacts": [
            "Color images (RGB) from field camera",
            "Variable natural lighting conditions",
            "Water turbidity effects on image quality",
            "Background may contain natural particles, debris",
            "Variable focus quality",
            "Organisms captured in natural swimming姿态",
            "Possible motion blur from live organisms",
        ],
        "ignore": "lighting conditions, water turbidity, background particles, color cast",
    },
}

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_baseline_prompt(classes: list) -> str:
    """VLM baseline: expert role + class list only."""
    class_list = ", ".join(classes)
    return (
        "You are a marine biologist expert. Look at this microscope image of a "
        "plankton organism carefully. Identify which organism it is from this "
        f"list: [{class_list}].\n\n"
        "Consider the shape, size, symmetry, and visible features. "
        "Reply with ONLY the single best matching organism name from the list."
    )


def build_morphological_rag_prompt(classes: list) -> str:
    """Morphological RAG: catalog of diagnostic features."""
    catalog_lines = []
    for cls in classes:
        if cls in MORPHOLOGICAL_CATALOG:
            catalog_lines.append(f"- {cls}: {MORPHOLOGICAL_CATALOG[cls]}")
    catalog = "\n".join(catalog_lines)

    return (
        "You are a marine biologist expert. Examine this microscope image and "
        "identify the organism by matching its visible morphological features "
        "against the catalog below.\n\n"
        f"## Morphological Catalog\n{catalog}\n\n"
        "## Instructions\n"
        "1. Examine the image for shape, symmetry, appendages, and surface features.\n"
        "2. Compare against EACH catalog entry above.\n"
        "3. Select the single best match.\n\n"
        "Reply with ONLY the organism name."
    )


def build_aa_rag_prompt(classes: list, source_domain: str, target_domain: str) -> str:
    """Artifact-Aware RAG: morphological catalog + artifact awareness."""
    # Morphological catalog
    catalog_lines = []
    for cls in classes:
        if cls in MORPHOLOGICAL_CATALOG:
            catalog_lines.append(f"- {cls}: {MORPHOLOGICAL_CATALOG[cls]}")
    catalog = "\n".join(catalog_lines)

    # Artifact catalogs
    src_artifacts = ARTIFACT_CATALOG.get(source_domain, {})
    tgt_artifacts = ARTIFACT_CATALOG.get(target_domain, {})

    artifact_section = ""
    if src_artifacts or tgt_artifacts:
        artifact_section = "\n\n## Imaging System Awareness\n"
        if src_artifacts:
            artifact_section += f"\n### Training images ({source_domain})\n"
            artifact_section += f"Instrument: {src_artifacts['instrument']}\n"
            artifact_section += "Artifacts: " + "; ".join(src_artifacts['artifacts'][:4]) + "\n"
        if tgt_artifacts:
            artifact_section += f"\n### Test images ({target_domain})\n"
            artifact_section += f"Instrument: {tgt_artifacts['instrument']}\n"
            artifact_section += "Artifacts: " + "; ".join(tgt_artifacts['artifacts'][:4]) + "\n"
        artifact_section += (
            "\n**IMPORTANT**: Different imaging systems produce different visual artifacts. "
            "Focus ONLY on morphological features (shape, symmetry, appendages, surface structure). "
            "IGNORE differences in background, contrast, illumination, and image quality."
        )

    return (
        "You are a marine biologist expert. Examine this microscope image and "
        "identify the organism by matching its visible morphological features "
        "against the catalog below.\n\n"
        f"## Morphological Catalog\n{catalog}"
        f"{artifact_section}\n\n"
        "## Instructions\n"
        "1. Examine the image for morphological features ONLY (shape, symmetry, appendages, surface).\n"
        "2. Compare against EACH catalog entry above.\n"
        "3. Select the single best match.\n\n"
        "Reply with ONLY the organism name."
    )


def build_aa_rag_with_ref_prompt(
    classes: list, source_domain: str, target_domain: str,
    ref_images: dict = None
) -> str:
    """AA-RAG with reference images: includes retrieved cross-domain references."""
    base_prompt = build_aa_rag_prompt(classes, source_domain, target_domain)

    if ref_images:
        ref_section = "\n\n## Reference Images from Multiple Domains\n"
        ref_section += "For each class, here are reference images from different imaging systems:\n"
        for cls, refs in ref_images.items():
            if cls in classes:
                ref_section += f"- {cls}: "
                domains = [r['domain'] for r in refs]
                ref_section += f"{len(refs)} reference(s) from {', '.join(set(domains))}\n"
        ref_section += "\nUse these references to understand how the same organism looks across imaging systems."
        return base_prompt + ref_section

    return base_prompt


# ---------------------------------------------------------------------------
# VLM client
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


def parse_class(raw: str, classes: list) -> str:
    cleaned = raw.strip().strip('"').strip("'").strip(".")
    for cls in classes:
        if cleaned.lower() == cls.lower():
            return cls
    for cls in classes:
        if cls.lower() in cleaned.lower():
            return cls
    return cleaned


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_samples(data_dir: str, domain: str, classes: list) -> list:
    samples = []
    domain_dir = Path(data_dir) / domain
    if not domain_dir.is_dir():
        return samples
    for cls in classes:
        cls_dir = domain_dir / cls
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in SUPPORTED_EXT:
                samples.append({"path": str(img_path), "true_label": cls, "domain": domain})
    return samples


# ---------------------------------------------------------------------------
# Cross-domain reference retrieval
# ---------------------------------------------------------------------------
def retrieve_reference_images(
    data_dir: str, source_domain: str, target_domain: str,
    classes: list, max_refs: int = 2
) -> dict:
    """Retrieve reference images from both source and target domains."""
    refs = {}
    for cls in classes:
        refs[cls] = []
        for domain in [source_domain, target_domain]:
            domain_dir = Path(data_dir) / domain / cls
            if not domain_dir.is_dir():
                continue
            images = sorted(domain_dir.glob("*.png"))[:max_refs]
            for img in images:
                refs[cls].append({"path": str(img), "domain": domain})
    return refs


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
# Main evaluation
# ---------------------------------------------------------------------------
def run_evaluation(samples: list, condition: str, prompt_fn, endpoint: str, model: str, classes: list) -> list:
    results = []
    for i, s in enumerate(samples):
        prompt = prompt_fn()
        try:
            raw = call_vlm(s["path"], prompt, endpoint, model)
        except Exception as e:
            logger.error("Failed %s: %s", s["path"], e)
            raw = ""
        pred = parse_class(raw, classes)
        correct = pred.lower() == s["true_label"].lower()
        results.append({
            "path": s["path"], "true_label": s["true_label"], "domain": s["domain"],
            "predicted": pred, "correct": correct, "raw_response": raw, "condition": condition,
        })
        if (i + 1) % 20 == 0 or (i + 1) == len(samples):
            acc = sum(1 for r in results if r["correct"]) / len(results)
            logger.info("  [%s] %d/%d  acc=%.1f%%", condition, i + 1, len(results), acc * 100)
    return results


def main():
    parser = argparse.ArgumentParser(description="Artifact-Aware RAG evaluation.")
    parser.add_argument("--data-dir", type=str, default="data/cross_domain/cross_instrument")
    parser.add_argument("--source-domain", type=str, default="train/DataShift_IFCB")
    parser.add_argument("--target-domain", type=str, default="test/DataShift_ZooScan")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-32B-Instruct-AWQ")
    parser.add_argument("--output", type=str, default="results/aa_rag_results.json")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    # Determine overlapping classes
    src_dir = Path(args.data_dir) / args.source_domain
    tgt_dir = Path(args.data_dir) / args.target_domain
    src_classes = {d.name for d in src_dir.iterdir() if d.is_dir()} if src_dir.is_dir() else set()
    tgt_classes = {d.name for d in tgt_dir.iterdir() if d.is_dir()} if tgt_dir.is_dir() else set()
    classes = sorted(src_classes & tgt_classes)
    logger.info("Overlapping classes: %s", classes)

    # Load target domain samples
    samples = load_samples(args.data_dir, args.target_domain, classes)
    if args.max_samples > 0:
        random.seed(42)
        samples = random.sample(samples, min(args.max_samples, len(samples)))
    logger.info("Loaded %d evaluation samples", len(samples))

    # Retrieve reference images
    ref_images = retrieve_reference_images(
        args.data_dir, args.source_domain, args.target_domain, classes
    )

    # Define conditions
    conditions = {
        "baseline": lambda: build_baseline_prompt(classes),
        "morphological_rag": lambda: build_morphological_rag_prompt(classes),
        "aa_rag": lambda: build_aa_rag_prompt(classes, args.source_domain.split("/")[-1], args.target_domain.split("/")[-1]),
        "aa_rag_ref": lambda: build_aa_rag_with_ref_prompt(
            classes, args.source_domain.split("/")[-1], args.target_domain.split("/")[-1], ref_images
        ),
    }

    all_results = {}
    for cond_name, prompt_fn in conditions.items():
        logger.info("Running condition: %s", cond_name)
        t0 = time.time()
        results = run_evaluation(samples, cond_name, prompt_fn, args.endpoint, args.model, classes)
        elapsed = time.time() - t0

        binary = [1 if r["correct"] else 0 for r in results]
        ci = bootstrap_ci(binary)

        per_class = {}
        for cls in classes:
            cls_results = [r for r in results if r["true_label"] == cls]
            if cls_results:
                per_class[cls] = {
                    "accuracy": sum(1 for r in cls_results if r["correct"]) / len(cls_results),
                    "n": len(cls_results),
                }

        all_results[cond_name] = {
            "accuracy": ci["mean"],
            "ci_95": [ci["ci_low"], ci["ci_high"]],
            "n_samples": len(results),
            "elapsed_s": elapsed,
            "per_class": per_class,
            "results": results,
        }

        logger.info("  %s: %.1f%% [95%% CI: %.1f-%.1f%%] (%.1fs)",
                     cond_name, ci["mean"] * 100, ci["ci_low"] * 100, ci["ci_high"] * 100, elapsed)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    report = {
        "conditions": {k: {kk: vv for kk, vv in v.items() if kk != "results"} for k, v in all_results.items()},
        "per_sample": {k: v["results"] for k, v in all_results.items()},
        "classes": classes,
        "metadata": vars(args),
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", args.output)

    # Print summary
    logger.info("=" * 72)
    logger.info("  AA-RAG EVALUATION SUMMARY")
    logger.info("=" * 72)
    for cond_name, data in all_results.items():
        logger.info("  %-20s  %.1f%% [%.1f-%.1f%%]", cond_name,
                     data["accuracy"] * 100, data["ci_95"][0] * 100, data["ci_95"][1] * 100)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

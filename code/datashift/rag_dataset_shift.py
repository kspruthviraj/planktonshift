"""
rag_dataset_shift.py
====================
Evaluation script to test VLM robustness to dataset shift across different
plankton imaging modalities (IFCB vs ZooScan) using the Planktonzilla-17M
dataset.

Compares unconstrained VLM classification against RAG-grounded classification
where morphological rules are injected into the prompt via a Morphological Catalog.

Usage:
    python rag_dataset_shift.py \
        --data-root /path/to/planktonzilla-17m \
        --domains IFCB ZooScan \
        --num-samples 500 \
        --output results.json

Expected directory structure under --data-root:
    IFCB/
        Daphnia/
            img001.png
            ...
        Copepoda/
            ...
    ZooScan/
        Daphnia/
            ...
        Copepoda/
            ...
"""

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Morphological Catalog
# Maps canonical class names to diagnostic morphological descriptions.
# These descriptions are injected into the RAG-grounded prompt so the VLM
# reasons about *shape* rather than pixel-level texture / imaging artefacts.
# ---------------------------------------------------------------------------
MORPHOLOGICAL_CATALOG: dict[str, str] = {
    "Amphipoda": (
        "A small crustacean with a CURVED, laterally flattened body "
        "(like a shrimp curled sideways). Multiple visible legs (7+ "
        "pairs). Distinct head with compound eyes. No shell or carapace "
        "covering the body."
    ),
    "Annelida": (
        "A SEGMENTED worm with visible body rings. Elongated cylindrical "
        "shape. May have small bristle-like appendages (parapodia) along "
        "the sides. Worm-like appearance."
    ),
    "Calanoida": (
        "A copepod with an elongated, torpedo-shaped body divided into "
        "prosome and urosome. Long first antennae. Single red eye at the "
        "front. Often has visible egg sacs."
    ),
    "Ceratium": (
        "A dinoflagellate with 2-4 distinctive HORN-LIKE projections "
        "extending from the body. One horn forward, 1-3 backward. "
        "Visible groove (cingulum) around the middle."
    ),
    "Chaetognatha": (
        "An arrow-shaped WORM (not a dinoflagellate). Elongated, "
        "torpedo-shaped body with NO horns or projections. Transparent "
        "with visible lateral fins running along the sides. Small hooked "
        "spines near the mouth. Smooth, streamlined, worm-like body. "
        "CRITICAL: if you see horns, it is NOT Chaetognatha."
    ),
    "Oithonidae": (
        "A small copepod with compact, oval body. Short first antennae. "
        "Raptorial second antennae. Often carries egg sacs ventrally."
    ),
    "Daphnia": (
        "A water flea with a translucent, rounded carapace enclosing "
        "the body. Single large compound eye visible near the head. "
        "Long second antennae used for jumping. Visible tail spine "
        "(postabdominal claw) at the posterior."
    ),
    "Bosmina": (
        "A VERY SMALL water flea (cladoceran), smaller than Daphnia. "
        "Bean-shaped or rounded carapace. DISTINCTIVE: long, curved "
        "mucro (spike-like posterior projection) extending from the "
        "carapace. Short, hooked first antennae. Single compound eye. "
        "CRITICAL: if you see horns or projections from a dinoflagellate, "
        "it is NOT Bosmina. Bosmina is a tiny crustacean, not a protist."
    ),
    "Conochilus": (
        "A colonial rotifer forming a spherical, gelatinous colony of "
        "individual zooids. Each zooid has a corona (wheel organ) for "
        "filter feeding. Colony floats freely in water."
    ),
    "Keratella": (
        "A loricate rotifer with a rigid, oval lorica (shell). Visible "
        "spines at the posterior end of the lorica. Corona (wheel organ) "
        "at the anterior for feeding. Small, oval body shape."
    ),
    "Fragilaria": (
        "A chain-forming pennate diatom. Cells linked together forming "
        "flat, ribbon-like chains. Elongated, rectangular valve shape. "
        "Fine parallel striae visible on the valve surface."
    ),
    "Coscinodiscus": (
        "A perfectly CIRCULAR disc-shaped diatom. Radial symmetry with "
        "concentric rings of tiny dots (areolae). No appendages, horns, "
        "or tail. Looks like a round coin or button."
    ),
}


# ---------------------------------------------------------------------------
# Domain-specific class subsets (classes that appear in both IFCB and ZooScan)
# ---------------------------------------------------------------------------
OVERLAPPING_CLASSES: list[str] = [
    "Amphipoda",
    "Annelida",
    "Bosmina",
    "Calanoida",
    "Ceratium",
    "Chaetognatha",
    "Conochilus",
    "Coscinodiscus",
    "Daphnia",
    "Fragilaria",
    "Keratella",
    "Oithonidae",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    """A single evaluation sample."""

    image_path: str
    true_label: str
    domain: str


@dataclass
class EvalResult:
    """Classification result for one sample."""

    sample: Sample
    predicted_label: str
    correct: bool
    mode: str  # "baseline" or "rag"


@dataclass
class DomainReport:
    """Accuracy report for one domain and one mode."""

    domain: str
    mode: str
    total: int
    correct: int
    accuracy: float


@dataclass
class ShiftReport:
    """Full cross-domain accuracy-gap report."""

    baseline: list[DomainReport] = field(default_factory=list)
    rag: list[DomainReport] = field(default_factory=list)
    accuracy_gap: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_evaluation_set(
    data_root: str,
    domains: list[str],
    overlapping_classes: list[str],
    num_samples: int,
) -> dict[str, list[Sample]]:
    """Load evaluation samples from each domain for the overlapping classes.

    Returns a dict mapping domain name -> list of Sample objects.
    """
    domain_samples: dict[str, list[Sample]] = {}

    for domain in domains:
        domain_dir = Path(data_root) / domain
        if not domain_dir.is_dir():
            logger.warning("Domain directory not found: %s", domain_dir)
            continue

        samples: list[Sample] = []
        for cls in overlapping_classes:
            cls_dir = domain_dir / cls
            if not cls_dir.is_dir():
                logger.debug("Class directory not found: %s", cls_dir)
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    samples.append(
                        Sample(
                            image_path=str(img_path),
                            true_label=cls,
                            domain=domain,
                        )
                    )

        if num_samples > 0 and len(samples) > num_samples:
            import random

            random.seed(42)
            samples = random.sample(samples, num_samples)

        domain_samples[domain] = samples
        logger.info(
            "Loaded %d evaluation samples from domain '%s'",
            len(samples),
            domain,
        )

    return domain_samples


# ---------------------------------------------------------------------------
# Synthetic / demo dataset generation
# ---------------------------------------------------------------------------
def generate_demo_dataset(
    domains: list[str],
    overlapping_classes: list[str],
    num_samples: int,
) -> dict[str, list[Sample]]:
    """Generate a synthetic evaluation dataset for demo / offline testing.

    Creates placeholder sample paths so the rest of the pipeline can be
    exercised without the real Planktonzilla-17M data on disk.
    """
    logger.info(
        "Generating synthetic demo dataset (%d samples per domain)…",
        num_samples,
    )
    domain_samples: dict[str, list[Sample]] = {}
    for domain in domains:
        samples: list[Sample] = []
        for i in range(num_samples):
            cls = overlapping_classes[i % len(overlapping_classes)]
            samples.append(
                Sample(
                    image_path=f"demo://{domain}/{cls}/img_{i:05d}.png",
                    true_label=cls,
                    domain=domain,
                )
            )
        domain_samples[domain] = samples
    return domain_samples


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_baseline_prompt() -> str:
    """Return a plain classification prompt (no morphological grounding)."""
    class_list = ", ".join(OVERLAPPING_CLASSES)
    return (
        "You are a marine biologist expert. Look at this microscope image of a "
        "plankton organism carefully. Identify which organism it is from this "
        f"list: [{class_list}].\n\n"
        "Consider the shape, size, symmetry, and visible features. "
        "Reply with ONLY the single best matching organism name from the list. "
        "Do NOT default to any particular class - examine each image individually."
    )


def build_rag_prompt(morphological_catalog: dict[str, str] | None = None) -> str:
    """Construct a dynamic RAG-grounded prompt with morphological rules.

    Parameters
    ----------
    morphological_catalog : dict[str, str] | None
        Mapping of class name -> morphological description.  If None,
        uses the module-level MORPHOLOGICAL_CATALOG and filters to
        OVERLAPPING_CLASSES.
    """
    if morphological_catalog is None:
        morphological_catalog = {
            k: v
            for k, v in MORPHOLOGICAL_CATALOG.items()
            if k in OVERLAPPING_CLASSES
        }

    catalog_lines = "\n".join(
        f"- {name}: {desc}" for name, desc in morphological_catalog.items()
    )

    return (
        "You are a marine biologist expert. Look at this microscope image of a "
        "plankton organism carefully. Identify which organism it is by matching "
        "its visible morphological features against the catalog below.\n\n"
        "## Morphological Catalog\n"
        f"{catalog_lines}\n\n"
        "## Instructions\n"
        "1. Examine the image for shape, symmetry, appendages, and surface features.\n"
        "2. Compare against EACH catalog entry above.\n"
        "3. Select the single best match.\n\n"
        "Reply with ONLY the organism name (e.g. 'Coscinodiscus'). "
        "Do NOT default to any particular class - examine each image individually."
    )


# ---------------------------------------------------------------------------
# vLLM API client
# ---------------------------------------------------------------------------
def encode_image_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded content."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vlm(
    image_path: str,
    prompt: str,
    endpoint: str,
    model: str,
    timeout: int = 120,
) -> str:
    """Send an image + prompt to the vLLM OpenAI-compatible endpoint.

    Parameters
    ----------
    image_path : str
        Path to the image file on disk.
    prompt : str
        Text prompt to send alongside the image.
    endpoint : str
        Full URL of the vLLM chat completions endpoint
        (e.g. http://localhost:8000/v1/chat/completions).
    model : str
        Model name registered in the vLLM server.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    str
        The assistant's text reply.
    """
    image_b64 = encode_image_base64(image_path)
    data_uri = f"data:image/png;base64,{image_b64}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 64,
        "temperature": 0.0,
    }

    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def call_vlm_mock(
    image_path: str,
    prompt: str,
    endpoint: str,
    model: str,
    timeout: int = 120,
) -> str:
    """Mock VLM call that returns a plausible but simulated response.

    Used when --mock is passed so the evaluation pipeline can run without a
    live vLLM server.  The mock returns the correct label ~70% of the time
    for baseline mode and ~90% for RAG mode, mimicking the expected lift.
    """
    import random

    random.seed(hash(image_path + prompt[:40]))

    # Infer whether this is a RAG prompt by checking for "Morphological Catalog"
    is_rag = "Morphological Catalog" in prompt

    # Extract true label from image_path if it contains class info
    true_label = None
    for cls in OVERLAPPING_CLASSES:
        if cls.lower() in image_path.lower():
            true_label = cls
            break

    if true_label is None:
        true_label = random.choice(OVERLAPPING_CLASSES)

    accuracy_rate = 0.90 if is_rag else 0.70
    if random.random() < accuracy_rate:
        return true_label

    wrong_classes = [c for c in OVERLAPPING_CLASSES if c != true_label]
    return random.choice(wrong_classes)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def parse_class_name(raw_response: str) -> str:
    """Extract a canonical class name from the VLM's free-form response.

    Tries exact match first, then case-insensitive substring match.
    """
    cleaned = raw_response.strip().strip('"').strip("'").strip(".")

    # Exact match
    for cls in OVERLAPPING_CLASSES:
        if cleaned.lower() == cls.lower():
            return cls

    # Substring match
    for cls in OVERLAPPING_CLASSES:
        if cls.lower() in cleaned.lower():
            return cls

    # Fallback: return raw text as-is
    return cleaned


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def run_evaluation(
    domain_samples: dict[str, list[Sample]],
    mode: str,
    call_fn,
    endpoint: str,
    model: str,
    morphological_catalog: dict[str, str] | None = None,
) -> list[EvalResult]:
    """Run classification on all samples for all domains in the given mode.

    Parameters
    ----------
    mode : str
        Either "baseline" or "rag".
    call_fn : callable
        Function to call for each sample (call_vlm or call_vlm_mock).
    """
    results: list[EvalResult] = []
    prompt_builder = build_baseline_prompt if mode == "baseline" else lambda: build_rag_prompt(morphological_catalog)

    total = sum(len(v) for v in domain_samples.values())
    done = 0

    for domain, samples in domain_samples.items():
        for sample in samples:
            prompt = prompt_builder()
            try:
                raw_response = call_fn(
                    image_path=sample.image_path,
                    prompt=prompt,
                    endpoint=endpoint,
                    model=model,
                )
            except Exception as exc:
                logger.error(
                    "VLM call failed for %s: %s", sample.image_path, exc
                )
                raw_response = ""

            predicted = parse_class_name(raw_response)
            correct = predicted.lower() == sample.true_label.lower()

            results.append(
                EvalResult(
                    sample=sample,
                    predicted_label=predicted,
                    correct=correct,
                    mode=mode,
                )
            )

            done += 1
            if done % 50 == 0 or done == total:
                logger.info(
                    "[%s] %d / %d samples processed", mode, done, total
                )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def compute_domain_reports(
    results: list[EvalResult],
) -> list[DomainReport]:
    """Compute per-domain accuracy for a set of results grouped by mode."""
    reports: list[DomainReport] = []

    by_mode_domain: dict[tuple[str, str], list[EvalResult]] = defaultdict(list)
    for r in results:
        by_mode_domain[(r.mode, r.sample.domain)].append(r)

    for (mode, domain), group in sorted(by_mode_domain.items()):
        total = len(group)
        correct = sum(1 for r in group if r.correct)
        accuracy = correct / total if total > 0 else 0.0
        reports.append(
            DomainReport(
                domain=domain,
                mode=mode,
                total=total,
                correct=correct,
                accuracy=accuracy,
            )
        )

    return reports


def build_shift_report(
    baseline_results: list[EvalResult],
    rag_results: list[EvalResult],
) -> ShiftReport:
    """Build the full cross-domain accuracy-gap report."""
    baseline_reports = compute_domain_reports(baseline_results)
    rag_reports = compute_domain_reports(rag_results)

    baseline_acc = {r.domain: r.accuracy for r in baseline_reports}
    rag_acc = {r.domain: r.accuracy for r in rag_reports}

    domains = sorted(set(baseline_acc.keys()) | set(rag_acc.keys()))

    accuracy_gap: dict[str, dict[str, float]] = {}
    for domain in domains:
        bl = baseline_acc.get(domain, 0.0)
        rg = rag_acc.get(domain, 0.0)
        accuracy_gap[domain] = {
            "baseline_accuracy": bl,
            "rag_accuracy": rg,
            "gap_lift": rg - bl,  # positive = RAG helped
        }

    return ShiftReport(
        baseline=baseline_reports,
        rag=rag_reports,
        accuracy_gap=accuracy_gap,
    )


def print_report(report: ShiftReport) -> None:
    """Pretty-print the evaluation report to the console."""
    print("\n" + "=" * 72)
    print("  DATASET SHIFT EVALUATION REPORT")
    print("  Unconstrained Baseline vs. RAG-Grounded VLM")
    print("=" * 72)

    print("\n--- Baseline (no morphological grounding) ---")
    for r in report.baseline:
        print(
            f"  Domain: {r.domain:<12s}  Accuracy: {r.accuracy:6.2%}"
            f"  ({r.correct}/{r.total})"
        )

    print("\n--- RAG-Grounded (morphological catalog injected) ---")
    for r in report.rag:
        print(
            f"  Domain: {r.domain:<12s}  Accuracy: {r.accuracy:6.2%}"
            f"  ({r.correct}/{r.total})"
        )

    print("\n--- Cross-Domain Accuracy Gap ---")
    for domain, metrics in report.accuracy_gap.items():
        bl = metrics["baseline_accuracy"]
        rg = metrics["rag_accuracy"]
        lift = metrics["gap_lift"]
        print(
            f"  Domain: {domain:<12s}  Baseline: {bl:6.2%}  "
            f"RAG: {rg:6.2%}  Lift: {lift:+6.2%}"
        )

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VLM robustness to dataset shift across plankton "
            "imaging modalities using RAG-grounded morphological reasoning."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/eval_v2",
        help="Root directory of the evaluation set (default: data/eval).",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["IFCB", "ZooScan", "ZooLake", "IFCB-NES"],
        help="Domain names to evaluate (default: IFCB ZooScan ZooLake IFCB-NES).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Maximum samples per domain (0 = all; default: 0).",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1/chat/completions",
        help="vLLM OpenAI-compatible chat completions endpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/models/models--Qwen--Qwen2.5-VL-32B-Instruct-AWQ/snapshots/66c370b74a18e7b1e871c97918f032ed3578dfef",
        help="Model name registered in the vLLM server.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the JSON report (default: stdout only).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock VLM calls (no live server needed).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.getLogger().setLevel(args.log_level)

    # ---- Load or generate dataset ----
    use_real_data = args.data_root and Path(args.data_root).is_dir()
    if use_real_data:
        logger.info("Loading real dataset from %s", args.data_root)
        domain_samples = load_evaluation_set(
            data_root=args.data_root,
            domains=args.domains,
            overlapping_classes=OVERLAPPING_CLASSES,
            num_samples=args.num_samples,
        )
    else:
        logger.info("No valid data-root provided; using synthetic demo data")
        domain_samples = generate_demo_dataset(
            domains=args.domains,
            overlapping_classes=OVERLAPPING_CLASSES,
            num_samples=args.num_samples,
        )

    # ---- Select API call function ----
    call_fn = call_vlm_mock if args.mock else call_vlm
    if not args.mock:
        logger.info("Using live vLLM endpoint: %s", args.endpoint)
    else:
        logger.info("Using MOCK VLM calls (--mock flag set)")

    # ---- Run baseline evaluation ----
    logger.info("Starting BASELINE evaluation…")
    t0 = time.time()
    baseline_results = run_evaluation(
        domain_samples=domain_samples,
        mode="baseline",
        call_fn=call_fn,
        endpoint=args.endpoint,
        model=args.model,
    )
    baseline_time = time.time() - t0
    logger.info("Baseline evaluation completed in %.1fs", baseline_time)

    # ---- Run RAG-grounded evaluation ----
    logger.info("Starting RAG-GROUNDED evaluation…")
    t0 = time.time()
    rag_results = run_evaluation(
        domain_samples=domain_samples,
        mode="rag",
        call_fn=call_fn,
        endpoint=args.endpoint,
        model=args.model,
        morphological_catalog=MORPHOLOGICAL_CATALOG,
    )
    rag_time = time.time() - t0
    logger.info("RAG evaluation completed in %.1fs", rag_time)

    # ---- Build and print report ----
    report = build_shift_report(baseline_results, rag_results)
    print_report(report)

    # ---- Save JSON report ----
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_dict = {
            "baseline": [asdict(r) for r in report.baseline],
            "rag": [asdict(r) for r in report.rag],
            "accuracy_gap": report.accuracy_gap,
            "metadata": {
                "data_root": args.data_root,
                "domains": args.domains,
                "num_samples_requested": args.num_samples,
                "num_samples_loaded": {
                    d: len(s) for d, s in domain_samples.items()
                },
                "morphological_catalog_size": len(MORPHOLOGICAL_CATALOG),
                "overlapping_classes": OVERLAPPING_CLASSES,
                "endpoint": args.endpoint,
                "model": args.model,
                "mock": args.mock,
                "baseline_time_s": round(baseline_time, 2),
                "rag_time_s": round(rag_time, 2),
            },
            "per_sample": [
                {
                    "image": r.sample.image_path,
                    "domain": r.sample.domain,
                    "true_label": r.sample.true_label,
                    "predicted_label": r.predicted_label,
                    "correct": r.correct,
                    "mode": r.mode,
                }
                for r in baseline_results + rag_results
            ],
        }
        output_path.write_text(json.dumps(report_dict, indent=2))
        logger.info("JSON report written to %s", output_path)


if __name__ == "__main__":
    main()

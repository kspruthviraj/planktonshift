"""
run_ravl_multivlm.py — Run RAVL with multiple VLMs for comparison.
Supports: Ollama models (llava, qwen2.5-vl, gemma3) and transformers-based models.
"""

import sys, json, argparse, base64, os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from morphological_catalog import (
    FRESHWATER_CATALOG, MARINE_CATALOG,
    get_catalog, format_catalog_for_prompt
)

ROOT = Path(__file__).resolve().parent.parent
OOD_DIR = ROOT / "data" / "chen_data" / "OOD_data" / "OODs"
RESULTS_DIR = ROOT / "results" / "ravl_multivlm"

# Domain data paths
DOMAINS = {
    "IFCB": Path("/home/sreenath/research-space/Adverserial_net/data/cross_domain/cross_instrument/train/DataShift_IFCB"),
    "ZooScan": Path("/home/sreenath/research-space/Adverserial_net/data/cross_domain/cross_instrument/test/DataShift_ZooScan"),
    "ZooLake": ROOT / "data" / "chen_data" / "ZooLake2" / "ZooLake2" / "ZooLake2.0",
    "IFCB-NES": OOD_DIR,  # Use OOD days as IFCB-NES proxy
}


def build_rag_prompt(catalog, domain):
    """Build RAG-enhanced prompt with morphological catalog."""
    catalog_text = format_catalog_for_prompt(catalog)
    
    if "IFCB" in domain:
        imaging_info = "These images are from an IFCB (Imaging FlowCytobot) using dark-field illumination. Expect high contrast against black background. Edge halos are artifacts, not biological features. Organisms may appear fragmented due to flow-through imaging."
    elif "ZooScan" in domain:
        imaging_info = "These images are from a ZooScan flatbed scanner. Organisms are preserved and laid flat. Background is grey. Some organisms may be folded or damaged from preservation."
    elif "ZooLake" in domain:
        imaging_info = "These images are from a DSPC field camera at Lake Greifensee. Variable natural lighting, water turbidity, background particles. Organisms are alive and may be in motion."
    else:
        imaging_info = "These are microscope images of plankton. Focus on morphological features."
    
    prompt = (
        "You are an expert plankton taxonomist. Examine this microscope image and "
        "identify the organism by matching its visible morphological features "
        "against the catalog below.\n\n"
        f"## Morphological Catalog\n{catalog_text}\n\n"
        f"## Imaging System Awareness\n{imaging_info}\n"
        "IMPORTANT: Focus ONLY on morphological features (shape, symmetry, appendages). "
        "IGNORE differences in background, contrast, illumination.\n\n"
        "Reply with ONLY the organism name from the catalog."
    )
    return prompt


def build_baseline_prompt(catalog, domain):
    """Build baseline prompt (list of names only)."""
    names = list(catalog.keys())
    return (
        f"You are a plankton taxonomist. Look at this microscope image and "
        f"identify which organism it is from this list: [{', '.join(names)}].\n"
        "Reply with ONLY the single best matching organism name."
    )


def call_ollama(image_path, prompt, model_name, endpoint="http://localhost:11434"):
    """Call Ollama VLM."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 32,
        }
    }
    
    resp = requests.post(f"{endpoint}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def parse_class(raw, classes):
    """Parse VLM output to extract class name."""
    cleaned = raw.strip().strip('"').strip("'").strip(".").lower()
    # Direct match
    for cls in classes:
        if cleaned == cls.lower():
            return cls
    # Partial match
    for cls in classes:
        if cls.lower() in cleaned:
            return cls
    return cleaned


def evaluate_domain(domain, domain_path, classes, model_name, use_rag, endpoint):
    """Evaluate on a single domain."""
    catalog = get_catalog(domain)
    
    if use_rag:
        prompt = build_rag_prompt(catalog, domain)
    else:
        prompt = build_baseline_prompt(catalog, domain)
    
    images, labels = [], []
    if domain == "IFCB-NES":
        # Use first OOD day as proxy
        ood_path = domain_path / "OOD1"
        if not ood_path.exists():
            ood_path = domain_path
        for cls_dir in sorted(ood_path.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in classes:
                continue
            cls_idx = np.where(classes == cls_dir.name)[0][0]
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                    images.append(str(img_path))
                    labels.append(cls_idx)
    else:
        for cls_dir in sorted(domain_path.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in classes:
                continue
            cls_idx = np.where(classes == cls_dir.name)[0][0]
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                    images.append(str(img_path))
                    labels.append(cls_idx)
    
    if not images:
        return None
    
    # Limit to 40 images per class for speed
    max_per_class = 40
    class_counts = {}
    filtered_images, filtered_labels = [], []
    for img, label in zip(images, labels):
        if class_counts.get(label, 0) < max_per_class:
            filtered_images.append(img)
            filtered_labels.append(label)
            class_counts[label] = class_counts.get(label, 0) + 1
    
    images, labels = filtered_images, np.array(filtered_labels)
    
    correct = 0
    predictions = []
    for i, img_path in enumerate(tqdm(images, desc=f"  {domain}", leave=False)):
        try:
            raw = call_ollama(img_path, prompt, model_name, endpoint)
            pred = parse_class(raw, classes)
            pred_idx = np.where(classes == pred)[0][0] if pred in classes else -1
            if pred_idx == labels[i]:
                correct += 1
            predictions.append({"raw": raw, "parsed": pred, "correct": pred_idx == labels[i]})
        except Exception as e:
            predictions.append({"raw": str(e), "parsed": "error", "correct": False})
    
    acc = correct / len(labels)
    return {"accuracy": float(acc), "correct": correct, "total": len(labels), "predictions": predictions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava:7b",
                        help="Ollama model name (e.g., llava:7b, qwen2.5-vl:7b, gemma3:4b)")
    parser.add_argument("--endpoint", type=str, default="http://localhost:11434",
                        help="Ollama API endpoint")
    parser.add_argument("--domains", nargs="*", default=["IFCB", "ZooScan", "ZooLake", "IFCB-NES"],
                        help="Domains to evaluate")
    parser.add_argument("--no-rag", action="store_true", help="Run baseline only (no RAG)")
    parser.add_argument("--rag-only", action="store_true", help="Run RAG only")
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load classes
    classes_path = ROOT / "data" / "chen_models" / "beit_models" / "trained_BEiT_models" / "classes.npy"
    classes = np.load(str(classes_path), allow_pickle=True)
    print(f"Classes: {len(classes)}")
    
    # Check model availability
    try:
        resp = requests.get(f"{args.endpoint}/api/tags", timeout=5)
        available = [m["name"] for m in resp.json().get("models", [])]
        print(f"Available models: {available}")
        if args.model not in available:
            print(f"WARNING: {args.model} not in available models!")
            print(f"Pull it first: ollama pull {args.model}")
            return
    except Exception as e:
        print(f"Cannot connect to Ollama: {e}")
        return
    
    results = {}
    
    # Run baseline (no RAG)
    if not args.rag_only:
        print(f"\n{'='*60}")
        print(f"BASELINE (no RAG) — {args.model}")
        print(f"{'='*60}")
        for domain in args.domains:
            domain_path = DOMAINS.get(domain)
            if domain_path and domain_path.exists():
                print(f"\n  {domain}:")
                res = evaluate_domain(domain, domain_path, classes, args.model, False, args.endpoint)
                if res:
                    results[f"{domain}_baseline"] = res
                    print(f"    Accuracy: {res['accuracy']:.4f} ({res['correct']}/{res['total']})")
    
    # Run RAG
    if not args.no_rag:
        print(f"\n{'='*60}")
        print(f"RAG (with Morphological Catalog) — {args.model}")
        print(f"{'='*60}")
        for domain in args.domains:
            domain_path = DOMAINS.get(domain)
            if domain_path and domain_path.exists():
                print(f"\n  {domain}:")
                res = evaluate_domain(domain, domain_path, classes, args.model, True, args.endpoint)
                if res:
                    results[f"{domain}_rag"] = res
                    print(f"    Accuracy: {res['accuracy']:.4f} ({res['correct']}/{res['total']})")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {args.model}")
    print(f"{'='*60}")
    print(f"{'Domain':<15} {'Baseline':>10} {'RAG':>10} {'Lift':>10}")
    print("-"*45)
    for domain in args.domains:
        bl = results.get(f"{domain}_baseline", {}).get("accuracy", 0)
        rag = results.get(f"{domain}_rag", {}).get("accuracy", 0)
        lift = (rag - bl) * 100
        print(f"{domain:<15} {bl:>10.1%} {rag:>10.1%} {lift:>+10.1f}%")
    
    # Save
    model_safe = args.model.replace(":", "_").replace("/", "_")
    out_path = RESULTS_DIR / f"ravl_{model_safe}.json"
    output = {
        "model": args.model,
        "results": {k: {"accuracy": v["accuracy"], "correct": v["correct"], "total": v["total"]}
                    for k, v in results.items()},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

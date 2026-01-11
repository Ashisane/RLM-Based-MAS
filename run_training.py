"""
Training Pipeline for Narrative Consistency Checker.

Runs the complete pipeline on training dataset (80 samples) and generates metrics.
"""
import sys
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from config import cost_tracker, CACHE_DIR, REPORT_DIR
from data_loader import load_training_data, get_novel_text
from constraint_extraction import extract_constraints_dense, cache_constraints
from ensemble_verification import (
    verify_samples_ensemble,
    EnsembleLogger,
    load_all_constraints,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run training pipeline on 80 training samples"
    )
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before running")
    parser.add_argument("--extract", action="store_true", help="Run constraint extraction")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N samples")
    parser.add_argument("--output", type=str, default="training_results.csv", help="Output CSV")
    return parser.parse_args()


def clear_cache():
    print("[CLEAR] Removing cached constraints...")
    for f in CACHE_DIR.glob("constraints_*.json"):
        f.unlink()
    print("[CLEAR] Cache cleared")


def ensure_constraints(books: list[str], force_extract: bool = False):
    print("\n" + "=" * 70)
    print("CONSTRAINT EXTRACTION")
    print("=" * 70)
    
    constraints_cache = load_all_constraints()
    
    for book in books:
        if book in constraints_cache and not force_extract:
            print(f"[CACHED] {book}")
            continue
        
        print(f"\n[EXTRACT] {book}")
        novel_text = get_novel_text(book)
        print(f"  Novel: {len(novel_text):,} chars")
        
        cost_before = cost_tracker.get_total_cost_inr()
        constraints = extract_constraints_dense(novel_text, book)
        cost_after = cost_tracker.get_total_cost_inr()
        
        cache_constraints(book, constraints)
        
        print(f"  Events: {len(constraints.get('events', []))}")
        print(f"  Relationships: {len(constraints.get('relationships', []))}")
        print(f"  Cost: INR {cost_after - cost_before:.2f}")
    
    return load_all_constraints()


def run_verification(samples, constraints_cache, novels_cache):
    print("\n" + "=" * 70)
    print("ENSEMBLE VERIFICATION")
    print("=" * 70)
    print(f"Samples: {len(samples)}")
    
    logger = EnsembleLogger("training_run")
    cost_before = cost_tracker.get_total_cost_inr()
    
    results = verify_samples_ensemble(samples, constraints_cache, novels_cache, logger)
    
    cost_after = cost_tracker.get_total_cost_inr()
    print(f"\n[DONE] Verified {len(results)} samples")
    print(f"[COST] INR {cost_after - cost_before:.2f}")
    
    return results


def calculate_metrics(results):
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    
    consistent = [r for r in results if r.actual_label == 'consistent']
    contradict = [r for r in results if r.actual_label == 'contradict']
    
    fp = sum(1 for r in consistent if r.final_verdict == 'contradict')
    fn = sum(1 for r in contradict if r.final_verdict == 'consistent')
    tp = sum(1 for r in contradict if r.final_verdict == 'contradict')
    tn = sum(1 for r in consistent if r.final_verdict == 'consistent')
    
    agreements = sum(1 for r in results if r.agents_agree)
    
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total * 100 if total > 0 else 0,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "fp_rate": fp / len(consistent) * 100 if consistent else 0,
        "fn_rate": fn / len(contradict) * 100 if contradict else 0,
        "precision": tp / (tp + fp) * 100 if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) * 100 if (tp + fn) > 0 else 0,
        "agreements": agreements,
        "agreement_rate": agreements / total * 100 if total > 0 else 0,
    }


def generate_csv(results, output_path):
    print("\n" + "=" * 70)
    print("GENERATING CSV")
    print("=" * 70)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Story ID', 'Prediction', 'Actual', 'Correct', 'Rationale'])
        
        for r in results:
            prediction = 1 if r.final_verdict == "consistent" else 0
            actual = 1 if r.actual_label == "consistent" else 0
            correct = "YES" if r.is_correct else "NO"
            
            if r.agents_agree:
                reason = r.conservative.reasoning
            else:
                reason = r.aggressive.reasoning if r.supervisor_chosen == "aggressive" else r.conservative.reasoning
            
            reason = ' '.join(reason.split())[:130]
            writer.writerow([r.sample_id, prediction, actual, correct, reason])
    
    print(f"[SAVED] {output_path}")


def generate_report(results, metrics, training_cost):
    report = f"""# Training Pipeline Report

## Summary
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Samples**: {metrics['total']}
- **Accuracy**: {metrics['accuracy']:.1f}%
- **Training Cost**: INR {training_cost:.2f}

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.1f}% ({metrics['correct']}/{metrics['total']}) |
| Precision | {metrics['precision']:.1f}% |
| Recall | {metrics['recall']:.1f}% |
| False Positive Rate | {metrics['fp_rate']:.1f}% |
| False Negative Rate | {metrics['fn_rate']:.1f}% |
| Agent Agreement Rate | {metrics['agreement_rate']:.1f}% |

## Confusion Matrix

|  | Predicted Consistent | Predicted Contradict |
|--|---------------------|---------------------|
| **Actual Consistent** | {metrics['true_negative']} (TN) | {metrics['false_positive']} (FP) |
| **Actual Contradict** | {metrics['false_negative']} (FN) | {metrics['true_positive']} (TP) |

## Cost
- Total Cost: INR {cost_tracker.get_total_cost_inr():.2f}
"""
    
    report_path = REPORT_DIR / "training_report.md"
    report_path.write_text(report, encoding='utf-8')
    print(f"[REPORT] {report_path}")


def main():
    args = parse_args()
    
    print("=" * 70)
    print("NARRATIVE CONSISTENCY CHECKER - TRAINING PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.clear_cache:
        clear_cache()
    
    print("\n[LOAD] Loading training data...")
    all_samples = load_training_data()
    
    if args.limit:
        samples = all_samples[:args.limit]
        print(f"[LIMIT] Using {len(samples)}/{len(all_samples)} samples")
    else:
        samples = all_samples
        print(f"[LOAD] {len(samples)} training samples")
    
    # Count labels
    consistent = sum(1 for s in samples if s.label == 'consistent')
    contradict = len(samples) - consistent
    print(f"  Consistent: {consistent}, Contradict: {contradict}")
    
    books = list(set(s.book_name for s in samples))
    print(f"[BOOKS] {books}")
    
    if args.extract or args.clear_cache:
        constraints_cache = ensure_constraints(books, force_extract=True)
    else:
        constraints_cache = load_all_constraints()
        missing = [b for b in books if b not in constraints_cache]
        if missing:
            print(f"\n[WARNING] Missing: {missing}")
            constraints_cache = ensure_constraints(missing, force_extract=True)
            constraints_cache = load_all_constraints()
    
    print("\n[LOAD] Loading novels...")
    novels_cache = {}
    for book in books:
        novels_cache[book] = get_novel_text(book)
        print(f"  {book}: {len(novels_cache[book]):,} chars")
    
    cost_before = cost_tracker.get_total_cost_inr()
    results = run_verification(samples, constraints_cache, novels_cache)
    training_cost = cost_tracker.get_total_cost_inr() - cost_before
    
    metrics = calculate_metrics(results)
    generate_csv(results, args.output)
    generate_report(results, metrics, training_cost)
    
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Accuracy: {metrics['accuracy']:.1f}%")
    print(f"FP Rate: {metrics['fp_rate']:.1f}% | FN Rate: {metrics['fn_rate']:.1f}%")
    print(f"Total Cost: INR {cost_tracker.get_total_cost_inr():.2f}")
    
    if metrics['accuracy'] >= 65:
        print("\n[PASS] Training complete - Ready for test set!")
    else:
        print("\n[WARN] Accuracy below 65% target")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED]")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

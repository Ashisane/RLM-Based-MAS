"""
Phase 3: Full Training Pipeline

Process all 80 training samples using ensemble approach,
tune threshold, perform error analysis, and finalize classifier.
"""
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent))

from config import cost_tracker, REPORT_DIR, CACHE_DIR
from data_loader import load_training_data, get_novel_text
from ensemble_verification import (
    verify_samples_ensemble,
    EnsembleLogger,
    load_all_constraints,
    EnsembleResult,
    verify_single_sample,
    filter_constraints_by_character
)

# Results cache
RESULTS_CACHE = CACHE_DIR / "training_results.json"


def run_full_training():
    """
    Process all 80 training samples with ensemble approach.
    """
    print("=" * 70)
    print("PHASE 3: FULL TRAINING PIPELINE")
    print("=" * 70)
    
    # Load all samples
    all_samples = load_training_data()
    print(f"\n[INFO] Total samples: {len(all_samples)}")
    
    # Count labels
    consistent = sum(1 for s in all_samples if s.label == 'consistent')
    contradict = sum(1 for s in all_samples if s.label == 'contradict')
    print(f"   Consistent: {consistent}, Contradict: {contradict}")
    
    # Load constraints and novels
    print("\n[LOAD] Loading constraints...")
    constraints_cache = load_all_constraints()
    print(f"   Books: {list(constraints_cache.keys())}")
    
    print("[LOAD] Loading novels...")
    novels_cache = {}
    for book in set(s.book_name for s in all_samples):
        novels_cache[book] = get_novel_text(book)
        print(f"   {book}: {len(novels_cache[book]):,} chars")
    
    # Run ensemble
    logger = EnsembleLogger("phase3_training")
    
    cost_before = cost_tracker.get_total_cost_inr()
    print(f"\n[TRAINING] Starting at budget: INR {cost_before:.2f}")
    
    results = verify_samples_ensemble(all_samples, constraints_cache, novels_cache, logger)
    
    cost_after = cost_tracker.get_total_cost_inr()
    training_cost = cost_after - cost_before
    
    # Save results
    save_results(results)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Threshold tuning
    threshold_results = tune_threshold(results)
    
    # Error analysis
    error_analysis = analyze_errors(results)
    
    # Generate report
    generate_training_report(results, metrics, threshold_results, error_analysis, training_cost)
    
    return results, metrics


def save_results(results: list[EnsembleResult]):
    """Save results to cache."""
    data = []
    for r in results:
        data.append({
            "sample_id": r.sample_id,
            "book_name": r.book_name,
            "character": r.character,
            "actual_label": r.actual_label,
            "conservative": r.conservative.verdict if r.conservative else None,
            "aggressive": r.aggressive.verdict if r.aggressive else None,
            "agents_agree": r.agents_agree,
            "supervisor_chosen": r.supervisor_chosen,
            "final_verdict": r.final_verdict,
            "is_correct": r.is_correct,
            "cost": r.total_cost
        })
    
    RESULTS_CACHE.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f"\n[CACHE] Results saved to: {RESULTS_CACHE}")


def calculate_metrics(results: list[EnsembleResult]) -> dict:
    """Calculate accuracy metrics."""
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    
    consistent = [r for r in results if r.actual_label == 'consistent']
    contradict = [r for r in results if r.actual_label == 'contradict']
    
    fp = sum(1 for r in consistent if r.final_verdict == 'contradict')
    fn = sum(1 for r in contradict if r.final_verdict == 'consistent')
    tp = sum(1 for r in contradict if r.final_verdict == 'contradict')
    tn = sum(1 for r in consistent if r.final_verdict == 'consistent')
    
    agreements = sum(1 for r in results if r.agents_agree)
    
    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total * 100,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "fp_rate": fp / len(consistent) * 100 if consistent else 0,
        "fn_rate": fn / len(contradict) * 100 if contradict else 0,
        "precision": tp / (tp + fp) * 100 if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) * 100 if (tp + fn) > 0 else 0,
        "agreements": agreements,
        "agreement_rate": agreements / total * 100,
        "supervisor_calls": total - agreements
    }
    
    return metrics


def tune_threshold(results: list[EnsembleResult]) -> dict:
    """
    Tune classification thresholds.
    
    For our ensemble:
    - When agents agree, use that
    - When they disagree, test different supervisor biases
    """
    print("\n[TUNING] Testing threshold strategies...")
    
    strategies = {
        "current": lambda r: r.final_verdict,
        "always_conservative": lambda r: r.conservative.verdict if r.conservative else "consistent",
        "always_aggressive": lambda r: r.aggressive.verdict if r.aggressive else "consistent",
        "agree_or_conservative": lambda r: r.conservative.verdict if r.conservative else "consistent" if not r.agents_agree else r.final_verdict,
        "agree_or_aggressive": lambda r: r.aggressive.verdict if r.aggressive else "consistent" if not r.agents_agree else r.final_verdict,
    }
    
    threshold_results = {}
    
    for name, strategy in strategies.items():
        correct = sum(1 for r in results if strategy(r) == r.actual_label)
        accuracy = correct / len(results) * 100
        threshold_results[name] = {
            "accuracy": accuracy,
            "correct": correct
        }
        print(f"   {name}: {accuracy:.1f}% ({correct}/{len(results)})")
    
    # Find best
    best = max(threshold_results.items(), key=lambda x: x[1]["accuracy"])
    threshold_results["best_strategy"] = best[0]
    threshold_results["best_accuracy"] = best[1]["accuracy"]
    
    print(f"\n[BEST] Strategy: {best[0]} at {best[1]['accuracy']:.1f}%")
    
    return threshold_results


def analyze_errors(results: list[EnsembleResult]) -> dict:
    """Analyze error patterns."""
    print("\n[ANALYSIS] Analyzing errors...")
    
    false_positives = [r for r in results if r.actual_label == 'consistent' and r.final_verdict == 'contradict']
    false_negatives = [r for r in results if r.actual_label == 'contradict' and r.final_verdict == 'consistent']
    
    # Categorize by pattern
    fp_patterns = {"both_wrong": 0, "supervisor_wrong": 0, "aggressive_only": 0}
    fn_patterns = {"both_wrong": 0, "supervisor_wrong": 0, "conservative_only": 0}
    
    for r in false_positives:
        if r.agents_agree:
            fp_patterns["both_wrong"] += 1
        elif r.supervisor_chosen == "aggressive":
            fp_patterns["supervisor_wrong"] += 1
        else:
            fp_patterns["aggressive_only"] += 1
    
    for r in false_negatives:
        if r.agents_agree:
            fn_patterns["both_wrong"] += 1
        elif r.supervisor_chosen == "conservative":
            fn_patterns["supervisor_wrong"] += 1
        else:
            fn_patterns["conservative_only"] += 1
    
    # Character analysis
    char_errors = {}
    for r in false_positives + false_negatives:
        char = r.character.split("/")[0]
        char_errors[char] = char_errors.get(char, 0) + 1
    
    analysis = {
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "fp_patterns": fp_patterns,
        "fn_patterns": fn_patterns,
        "character_errors": dict(sorted(char_errors.items(), key=lambda x: -x[1])[:10]),
        "fp_samples": [{"id": r.sample_id, "char": r.character} for r in false_positives[:5]],
        "fn_samples": [{"id": r.sample_id, "char": r.character} for r in false_negatives[:5]]
    }
    
    print(f"   False Positives: {len(false_positives)}")
    print(f"     Both agents wrong: {fp_patterns['both_wrong']}")
    print(f"     Supervisor chose wrong: {fp_patterns['supervisor_wrong']}")
    
    print(f"   False Negatives: {len(false_negatives)}")
    print(f"     Both agents wrong: {fn_patterns['both_wrong']}")
    print(f"     Supervisor chose wrong: {fn_patterns['supervisor_wrong']}")
    
    print(f"   Characters with most errors: {list(char_errors.keys())[:5]}")
    
    return analysis


def generate_training_report(results: list[EnsembleResult], metrics: dict, 
                             threshold_results: dict, error_analysis: dict,
                             training_cost: float):
    """Generate Phase 3 training report."""
    
    report = f"""# Phase 3 Report: Full Training Pipeline

## Overview
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Samples Processed**: {metrics['total']}
- **Accuracy**: {metrics['accuracy']:.1f}%
- **Training Cost**: INR {training_cost:.2f}
- **Total Cost**: INR {cost_tracker.get_total_cost_inr():.2f}

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.1f}% ({metrics['correct']}/{metrics['total']}) |
| Precision | {metrics['precision']:.1f}% |
| Recall | {metrics['recall']:.1f}% |
| False Positive Rate | {metrics['fp_rate']:.1f}% |
| False Negative Rate | {metrics['fn_rate']:.1f}% |
| Agent Agreements | {metrics['agreement_rate']:.1f}% |
| Supervisor Calls | {metrics['supervisor_calls']} |

## Confusion Matrix

|  | Predicted Consistent | Predicted Contradict |
|--|---------------------|---------------------|
| **Actual Consistent** | {metrics['true_negative']} (TN) | {metrics['false_positive']} (FP) |
| **Actual Contradict** | {metrics['false_negative']} (FN) | {metrics['true_positive']} (TP) |

## Threshold Tuning

| Strategy | Accuracy |
|----------|----------|
"""
    
    for name, data in threshold_results.items():
        if name not in ["best_strategy", "best_accuracy"]:
            report += f"| {name} | {data['accuracy']:.1f}% |\n"
    
    report += f"""
**Best Strategy**: {threshold_results.get('best_strategy', 'current')} at {threshold_results.get('best_accuracy', metrics['accuracy']):.1f}%

## Error Analysis

### False Positives ({error_analysis['false_positives']})
- Both agents wrong: {error_analysis['fp_patterns']['both_wrong']}
- Supervisor chose wrong: {error_analysis['fp_patterns']['supervisor_wrong']}

### False Negatives ({error_analysis['false_negatives']})
- Both agents wrong: {error_analysis['fn_patterns']['both_wrong']}
- Supervisor chose wrong: {error_analysis['fn_patterns']['supervisor_wrong']}

### Characters with Most Errors
"""
    
    for char, count in list(error_analysis['character_errors'].items())[:5]:
        report += f"- {char}: {count} errors\n"
    
    report += f"""

## Cost Breakdown
{cost_tracker.summary()}

## Next Steps
"""
    
    if metrics['accuracy'] >= 65:
        report += "- ✅ Accuracy target (65%) achieved - Proceed to Phase 4 (Test Set)\n"
    else:
        report += "- ⚠️ Accuracy below target - Consider prompt tuning\n"
    
    if cost_tracker.get_total_cost_inr() < 10000:
        report += f"- ✅ Budget healthy: INR {cost_tracker.get_total_cost_inr():.2f} / INR 12,000\n"
    else:
        report += f"- ⚠️ Budget alert: INR {cost_tracker.get_total_cost_inr():.2f} approaching limit\n"
    
    report_path = REPORT_DIR / "stage_3.md"
    report_path.write_text(report, encoding='utf-8')
    print(f"\n[REPORT] Saved to: {report_path}")


if __name__ == "__main__":
    try:
        results, metrics = run_full_training()
        
        print("\n" + "=" * 70)
        print("PHASE 3 SUMMARY")
        print("=" * 70)
        print(f"[ACCURACY] {metrics['accuracy']:.1f}%")
        print(f"[TOTAL COST] INR {cost_tracker.get_total_cost_inr():.2f}")
        
        if metrics['accuracy'] >= 65:
            print("\n[PASS] Phase 3 Complete - Ready for Phase 4!")
        else:
            print("\n[WARN] Accuracy below target - Review errors")
        
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

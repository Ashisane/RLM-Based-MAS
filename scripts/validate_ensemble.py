"""
Validation script for ensemble verification.
Tests on a small subset of training data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import cost_tracker
from data_loader import load_training_data, get_novel_text
from ensemble_verification import (
    verify_samples_ensemble,
    EnsembleLogger,
    load_all_constraints,
)


def validate_ensemble(sample_count: int = 5):
    """Validate ensemble on small sample."""
    print("=" * 70)
    print("PHASE 2 ENSEMBLE VALIDATION")
    print("=" * 70)
    
    # Load samples
    all_samples = load_training_data()
    samples = get_balanced_samples(all_samples, n_consistent=10, n_contradict=10)
    
    print(f"\n[INFO] Samples: {len(samples)} (10 consistent, 10 contradict)")
    
    # Load data
    print("[LOAD] Loading constraints...")
    constraints_cache = load_all_constraints()
    print(f"   Books: {list(constraints_cache.keys())}")
    
    print("[LOAD] Loading novels...")
    novels_cache = {}
    for book in set(s.book_name for s in samples):
        novels_cache[book] = get_novel_text(book)
        print(f"   {book}: {len(novels_cache[book]):,} chars")
    
    # Run ensemble verification
    logger = EnsembleLogger("ensemble_validation")
    
    cost_before = cost_tracker.get_total_cost_inr()
    results = verify_samples_ensemble(samples, constraints_cache, novels_cache, logger)
    cost_after = cost_tracker.get_total_cost_inr()
    
    # Calculate metrics
    correct = sum(1 for r in results if r.is_correct)
    accuracy = (correct / len(results)) * 100
    
    consistent = [r for r in results if r.actual_label == 'consistent']
    contradict = [r for r in results if r.actual_label == 'contradict']
    
    fp = sum(1 for r in consistent if r.final_verdict == 'contradict')
    fn = sum(1 for r in contradict if r.final_verdict == 'consistent')
    
    fp_rate = (fp / len(consistent)) * 100 if consistent else 0
    fn_rate = (fn / len(contradict)) * 100 if contradict else 0
    
    agreements = sum(1 for r in results if r.agents_agree)
    
    # Generate report
    generate_report(results, accuracy, fp_rate, fn_rate, cost_after - cost_before)
    
    # Print summary
    print("\n" + "=" * 70)
    print("ENSEMBLE VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\n[ACCURACY] {correct}/{len(results)} = {accuracy:.1f}%")
    print(f"[FALSE POSITIVE RATE] {fp}/{len(consistent)} = {fp_rate:.1f}%")
    print(f"[FALSE NEGATIVE RATE] {fn}/{len(contradict)} = {fn_rate:.1f}%")
    print(f"[AGENT AGREEMENTS] {agreements}/{len(results)} ({agreements/len(results)*100:.1f}%)")
    print(f"[SUPERVISOR CALLS] {len(results) - agreements}")
    print(f"[PHASE 2 COST] INR {cost_after - cost_before:.2f}")
    print(f"[TOTAL COST] INR {cost_tracker.get_total_cost_inr():.2f}")
    print(cost_tracker.summary())
    
    # Show details
    print("\n[AGREEMENT CASES]")
    for r in results:
        if r.agents_agree:
            status = "OK" if r.is_correct else "WRONG"
            print(f"  [{status}] ID={r.sample_id}: Both say {r.final_verdict} (actual: {r.actual_label})")
    
    print("\n[SUPERVISOR CASES]")
    for r in results:
        if not r.agents_agree:
            status = "OK" if r.is_correct else "WRONG"
            print(f"  [{status}] ID={r.sample_id}: Con={r.conservative.verdict}, Agg={r.aggressive.verdict}")
            print(f"        Supervisor chose: {r.supervisor_chosen} -> {r.final_verdict} (actual: {r.actual_label})")
    
    passed = accuracy >= 65
    print(f"\n{'[PASS]' if passed else '[WARN]'} Ensemble Validation: {accuracy:.1f}%")
    
    if accuracy >= 70:
        print("[SUCCESS] Target accuracy >= 70% achieved!")
    
    return results


def generate_report(results: list[EnsembleResult], accuracy: float, 
                   fp_rate: float, fn_rate: float, phase_cost: float):
    """Generate ensemble report."""
    import datetime
    
    agreements = sum(1 for r in results if r.agents_agree)
    
    report = f"""# Phase 2 Report: Ensemble Verification

## Overview
- **Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Approach**: Dual-Agent Ensemble with Supervisor Arbitration
- **Accuracy**: {accuracy:.1f}%
- **False Positive Rate**: {fp_rate:.1f}%
- **False Negative Rate**: {fn_rate:.1f}%
- **Agent Agreements**: {agreements}/{len(results)} ({agreements/len(results)*100:.1f}%)
- **Supervisor Calls**: {len(results) - agreements}
- **Phase 2 Cost**: INR {phase_cost:.2f}
- **Total Cost**: INR {cost_tracker.get_total_cost_inr():.2f}

## Methodology

### Dual-Agent Approach
1. **Conservative Agent**: Leans toward "consistent", only flags explicit conflicts
2. **Aggressive Agent**: Looks for any error, flags even subtle issues
3. **Supervisor**: When agents disagree, evaluates reasoning quality (WITHOUT seeing novel facts)

### Decision Logic
- If both agents agree → use that verdict
- If they disagree → supervisor picks the more logically sound reasoning

## Results

### Correct Predictions ({sum(1 for r in results if r.is_correct)}/{len(results)})
| ID | Character | Agents | Final | Supervisor |
|----|-----------|--------|-------|------------|
"""
    
    for r in sorted(results, key=lambda x: x.sample_id):
        if r.is_correct:
            agents = "AGREE" if r.agents_agree else f"C:{r.conservative.verdict[0].upper()}/A:{r.aggressive.verdict[0].upper()}"
            sup = r.supervisor_chosen[:3].upper() if r.supervisor_chosen else "-"
            report += f"| {r.sample_id} | {r.character[:15]} | {agents} | {r.final_verdict} | {sup} |\n"
    
    report += f"""

### Incorrect Predictions ({sum(1 for r in results if not r.is_correct)}/{len(results)})
| ID | Character | Actual | Final | Conservative | Aggressive | Supervisor |
|----|-----------|--------|-------|--------------|------------|------------|
"""
    
    for r in sorted(results, key=lambda x: x.sample_id):
        if not r.is_correct:
            sup = r.supervisor_chosen if r.supervisor_chosen else "-"
            report += f"| {r.sample_id} | {r.character[:12]} | {r.actual_label} | {r.final_verdict} | {r.conservative.verdict} | {r.aggressive.verdict} | {sup} |\n"
    
    report += f"""

## Analysis

### Why Ensemble Works
- Conservative catches "obviously consistent" cases
- Aggressive catches "obvious contradictions"
- Supervisor resolves edge cases by evaluating reasoning quality
- Removes bias from individual prompts

## Cost Breakdown
{cost_tracker.summary()}

## Next Steps
{"- Proceed to Phase 3 (full training)" if accuracy >= 65 else "- Consider prompt tuning"}
"""
    
    report_path = REPORT_DIR / "stage_2.md"
    report_path.write_text(report, encoding='utf-8')
    print(f"\n[REPORT] Saved to: {report_path}")


if __name__ == "__main__":
    try:
        results = validate_ensemble()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

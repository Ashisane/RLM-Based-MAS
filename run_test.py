"""
End-to-End Test Pipeline for Narrative Consistency Checker.

Runs the complete pipeline on test dataset and generates results.csv.
"""
import sys
import csv
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from config import cost_tracker, CACHE_DIR, REPORT_DIR
from data_loader import load_test_data, get_novel_text
from constraint_extraction import extract_constraints_dense, cache_constraints
from ensemble_verification import (
    verify_samples_ensemble,
    EnsembleLogger,
    load_all_constraints,
    ProgressTracker
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run narrative consistency checker on test dataset"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache before running (forces re-extraction)"
    )
    
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Run constraint extraction (if not cached)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N test samples (default: all)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results.csv",
        help="Output CSV file (default: results.csv)"
    )
    
    return parser.parse_args()


def clear_cache():
    """Clear all cached data."""
    print("[CLEAR] Removing cached constraints...")
    for f in CACHE_DIR.glob("constraints_*.json"):
        f.unlink()
    print("[CLEAR] Cache cleared")


def ensure_constraints(books: list[str], force_extract: bool = False):
    """Ensure constraints exist for all books."""
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
        print(f"  Novel length: {len(novel_text):,} chars")
        
        cost_before = cost_tracker.get_total_cost_inr()
        constraints = extract_constraints_dense(novel_text, book)
        cost_after = cost_tracker.get_total_cost_inr()
        
        cache_constraints(book, constraints)
        
        print(f"  Extracted constraints:")
        print(f"    Events: {len(constraints.get('events', []))}")
        print(f"    Relationships: {len(constraints.get('relationships', []))}")
        print(f"    Capabilities: {len(constraints.get('capabilities', []))}")
        print(f"    Cost: INR {cost_after - cost_before:.2f}")
    
    return load_all_constraints()


def run_verification(samples: list, constraints_cache: dict, novels_cache: dict):
    """Run ensemble verification on samples."""
    print("\n" + "=" * 70)
    print("ENSEMBLE VERIFICATION")
    print("=" * 70)
    print(f"Samples: {len(samples)}")
    
    logger = EnsembleLogger("test_run")
    cost_before = cost_tracker.get_total_cost_inr()
    
    results = verify_samples_ensemble(samples, constraints_cache, novels_cache, logger)
    
    cost_after = cost_tracker.get_total_cost_inr()
    
    print(f"\n[DONE] Verified {len(results)} samples")
    print(f"[COST] INR {cost_after - cost_before:.2f}")
    
    return results


def generate_csv(results: list, output_path: str):
    """Generate results.csv in required format."""
    print("\n" + "=" * 70)
    print("GENERATING CSV")
    print("=" * 70)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Story ID', 'Prediction', 'Rationale'])
        
        for result in results:
            # Convert verdict to 1/0
            prediction = 1 if result.final_verdict == "consistent" else 0
            
            # Build clean rationale
            rationale = build_clean_rationale(result, prediction)
            
            writer.writerow([result.sample_id, prediction, rationale])
    
    print(f"[SAVED] {output_path}")
    print(f"  Samples: {len(results)}")


def build_clean_rationale(result, prediction: int) -> str:
    """Build a clean rationale for the CSV."""
    
    # Get the reasoning from the deciding agent
    if result.agents_agree:
        reason = result.conservative.reasoning
    else:
        if result.supervisor_chosen == "conservative":
            reason = result.conservative.reasoning
        else:
            reason = result.aggressive.reasoning
    
    # Basic cleanup
    reason = ' '.join(reason.split()).strip()
    
    # Ensure starts with capital
    if reason and reason[0].islower():
        reason = reason[0].upper() + reason[1:]
    
    # Limit to complete sentences, prefer longer rationales
    max_len = 200
    if len(reason) > max_len:
        # Find last complete sentence within limit
        for end in ['. ', '! ', '? ']:
            pos = reason.rfind(end, 50, max_len)
            if pos > 50:
                return reason[:pos + 1]
        # No sentence end found - cut at last comma or word
        for sep in [', ', ' - ', '; ', ' ']:
            pos = reason.rfind(sep, 80, max_len)
            if pos > 80:
                return reason[:pos] + '.'
        return reason[:max_len] + '...'
    
    # Ensure ends with punctuation
    if reason and reason[-1] not in '.!?':
        reason = reason + '.'
    
    return reason if reason else ("No contradictions found with novel." if prediction == 1 else "Factual inconsistency detected.")


def generate_summary(results: list, output_path: str):
    """Generate summary statistics."""
    total = len(results)
    consistent = sum(1 for r in results if r.final_verdict == "consistent")
    contradict = total - consistent
    
    agreements = sum(1 for r in results if r.agents_agree)
    supervisor_calls = total - agreements
    
    total_cost = cost_tracker.get_total_cost_inr()
    
    summary = f"""# Test Run Summary

## Results
- **Total Samples**: {total}
- **Consistent**: {consistent} ({consistent/total*100:.1f}%)
- **Contradict**: {contradict} ({contradict/total*100:.1f}%)

## Pipeline Stats
- **Agent Agreements**: {agreements}/{total} ({agreements/total*100:.1f}%)
- **Supervisor Calls**: {supervisor_calls}
- **Total Cost**: INR {total_cost:.2f}

## Output
- Results saved to: `{output_path}`

## Timestamp
- Generated: {datetime.now().isoformat()}
"""
    
    summary_file = REPORT_DIR / "test_summary.md"
    summary_file.write_text(summary, encoding='utf-8')
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total: {total} samples")
    print(f"Consistent: {consistent} | Contradict: {contradict}")
    print(f"Agent agreements: {agreements}/{total}")
    print(f"Total cost: INR {total_cost:.2f}")
    print(f"\nSummary saved to: {summary_file}")


def main():
    """Main execution."""
    args = parse_args()
    
    print("=" * 70)
    print("NARRATIVE CONSISTENCY CHECKER - TEST PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Clear cache if requested
    if args.clear_cache:
        clear_cache()
    
    # Load test data
    print("\n[LOAD] Loading test data...")
    all_samples = load_test_data()
    
    if args.limit:
        samples = all_samples[:args.limit]
        print(f"[LIMIT] Using {len(samples)}/{len(all_samples)} samples")
    else:
        samples = all_samples
        print(f"[LOAD] {len(samples)} test samples")
    
    # Get unique books
    books = list(set(s.book_name for s in samples))
    print(f"[BOOKS] {len(books)} books: {books}")
    
    # Ensure constraints exist
    if args.extract or args.clear_cache:
        constraints_cache = ensure_constraints(books, force_extract=True)
    else:
        constraints_cache = load_all_constraints()
        missing_books = [b for b in books if b not in constraints_cache]
        if missing_books:
            print(f"\n[WARNING] Missing constraints for: {missing_books}")
            print("[EXTRACT] Running extraction for missing books...")
            constraints_cache = ensure_constraints(missing_books, force_extract=True)
            constraints_cache = load_all_constraints()
    
    # Load novels
    print("\n[LOAD] Loading novels...")
    novels_cache = {}
    for book in books:
        novels_cache[book] = get_novel_text(book)
        print(f"  {book}: {len(novels_cache[book]):,} chars")
    
    # Run verification
    results = run_verification(samples, constraints_cache, novels_cache)
    
    # Generate CSV
    generate_csv(results, args.output)
    
    # Generate summary
    generate_summary(results, args.output)
    
    print("\n" + "=" * 70)
    print("TEST PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Results: {args.output}")
    print(f"Cost: INR {cost_tracker.get_total_cost_inr():.2f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Test pipeline stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

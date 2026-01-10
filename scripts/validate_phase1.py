"""
Phase 1 Validation: Test dense constraint extraction with logging.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import cost_tracker, REPORT_DIR, CACHE_DIR
from data_loader import load_training_data, get_balanced_samples, get_novel_text
from phase1_extractor import extract_constraints_dense, cache_constraints


def validate_phase1():
    """Validate Phase 1: Dense RLM Constraint Extraction"""
    print("=" * 70)
    print("PHASE 1 VALIDATION: Dense RLM Constraint Extraction")
    print("=" * 70)
    print(f"Logs will be saved to: {CACHE_DIR / 'logs'}")
    
    # Get unique books from samples
    all_samples = load_training_data()
    books = list(set(s.book_name for s in all_samples))
    print(f"\n[INFO] Books to process: {books}")
    
    results = {}
    
    for book_name in books:
        print(f"\n{'='*60}")
        print(f"Processing: {book_name}")
        print(f"{'='*60}")
        
        cost_before = cost_tracker.get_total_cost_inr()
        
        try:
            novel_text = get_novel_text(book_name)
            print(f"[OK] Loaded novel: {len(novel_text):,} characters")
            
            # Extract with dense approach - pass book name for logging
            constraints = extract_constraints_dense(
                novel_text, 
                book_name=book_name,
                max_chunks=8, 
                parallel_workers=4
            )
            
            cache_constraints(book_name, constraints)
            
            cost_after = cost_tracker.get_total_cost_inr()
            cost_for_book = cost_after - cost_before
            
            results[book_name] = {
                "success": True,
                "events": len(constraints.get("events", [])),
                "relationships": len(constraints.get("relationships", [])),
                "capabilities": len(constraints.get("capabilities", [])),
                "timeline_facts": len(constraints.get("temporal", {}).get("timeline_facts", [])),
                "ages": len(constraints.get("temporal", {}).get("ages", [])),
                "characters": len(constraints.get("characters", [])),
                "cost": cost_for_book,
                "log_file": constraints.get("metadata", {}).get("log_file", "")
            }
            
            print(f"\n[SUMMARY] {book_name}:")
            for key, val in results[book_name].items():
                if key not in ['success', 'log_file']:
                    print(f"   {key}: {val}")
            
        except Exception as e:
            print(f"[ERROR] {book_name}: {e}")
            import traceback
            traceback.print_exc()
            results[book_name] = {"success": False, "error": str(e)}
    
    # Generate report
    generate_report(results)
    
    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)
    print(f"Total cost: INR {cost_tracker.get_total_cost_inr():.2f}")
    print(cost_tracker.summary())
    
    all_success = all(r.get("success", False) for r in results.values())
    print(f"\n{'[PASS]' if all_success else '[FAIL]'} Phase 1 Validation")
    return all_success


def generate_report(results: dict):
    import datetime
    
    report = f"""# Phase 1 Report: Dense RLM Constraint Extraction

## Overview
- **Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Cost**: INR {cost_tracker.get_total_cost_inr():.2f}

## Results

| Book | Events | Relationships | Capabilities | Age Facts | Cost |
|------|--------|---------------|--------------|-----------|------|
"""
    for book, data in results.items():
        if data.get("success"):
            report += f"| {book[:30]} | {data['events']} | {data['relationships']} | {data['capabilities']} | {data['ages']} | INR {data['cost']:.2f} |\n"
        else:
            report += f"| {book[:30]} | ERROR: {data.get('error', 'unknown')[:20]} | - | - | - | - |\n"
    
    report += f"\n## Log Files\n"
    for book, data in results.items():
        if data.get("log_file"):
            report += f"- {book}: `{data['log_file']}`\n"
    
    report += f"\n## Cost Breakdown\n{cost_tracker.summary()}"
    
    report_path = REPORT_DIR / "stage_1.md"
    report_path.write_text(report, encoding='utf-8')
    print(f"\n[REPORT] Saved to: {report_path}")


if __name__ == "__main__":
    try:
        success = validate_phase1()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[CRITICAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

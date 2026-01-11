"""
Unified Logging System for Narrative Consistency Checker.

Provides:
- Single log file per phase run
- Structured JSON results
- Concise terminal output with progress bars
- No duplicate logging
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Any

from config import CACHE_DIR

# Log directory
LOG_DIR = CACHE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


class ProgressBar:
    """Simple progress bar for async operations."""
    
    def __init__(self, total: int, description: str = "Processing", width: int = 40):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = datetime.now()
        self.extra_info = {}
    
    def update(self, step: int = 1, **kwargs):
        """Update progress bar."""
        self.current += step
        self.extra_info.update(kwargs)
        self._render()
    
    def set_info(self, **kwargs):
        """Set extra info without incrementing."""
        self.extra_info.update(kwargs)
        self._render()
    
    def _render(self):
        """Render the progress bar."""
        pct = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)
        
        # Calculate ETA
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if pct > 0:
            eta = elapsed / pct - elapsed
            eta_str = f"{eta:.0f}s" if eta < 60 else f"{eta/60:.1f}m"
        else:
            eta_str = "..."
        
        # Build info string
        info_parts = [f"{self.current}/{self.total}"]
        if "accuracy" in self.extra_info:
            info_parts.append(f"Acc:{self.extra_info['accuracy']:.1f}%")
        if "cost" in self.extra_info:
            info_parts.append(f"₹{self.extra_info['cost']:.2f}")
        info_parts.append(f"ETA:{eta_str}")
        
        info_str = " | ".join(info_parts)
        
        # Print with carriage return
        sys.stdout.write(f"\r{self.description}: |{bar}| {info_str}   ")
        sys.stdout.flush()
    
    def finish(self, message: str = None):
        """Finish the progress bar."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        final_msg = message or f"Completed in {elapsed:.1f}s"
        pct = self.current / self.total if self.total > 0 else 1.0
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)
        sys.stdout.write(f"\r{self.description}: |{bar}| {final_msg}   \n")
        sys.stdout.flush()


class PhaseLogger:
    """
    Unified logger for a single phase run.
    
    Creates one log file and one results file per run.
    """
    
    def __init__(self, phase_name: str):
        """
        Initialize logger for a phase.
        
        Args:
            phase_name: e.g., "phase1_extraction", "phase2_ensemble", "phase3_training"
        """
        self.phase_name = phase_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Single log file
        self.log_file = LOG_DIR / f"{phase_name}_{self.timestamp}.log"
        
        # Results file (JSONL for easy parsing)
        self.results_file = LOG_DIR / f"{phase_name}_{self.timestamp}_results.jsonl"
        
        # Track summary stats
        self.stats = {
            "phase": phase_name,
            "started": datetime.now().isoformat(),
            "total_samples": 0,
            "processed": 0,
            "correct": 0,
            "errors": 0
        }
        
        # Write header
        self._log(f"{'=' * 60}")
        self._log(f"{phase_name.upper()} LOG")
        self._log(f"Started: {self.stats['started']}")
        self._log(f"{'=' * 60}")
    
    def _log(self, msg: str):
        """Write to log file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")
    
    def info(self, msg: str):
        """Log info message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log(f"[{timestamp}] INFO: {msg}")
    
    def error(self, msg: str):
        """Log error message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log(f"[{timestamp}] ERROR: {msg}")
        self.stats["errors"] += 1
    
    def section(self, title: str):
        """Log section header."""
        self._log(f"\n--- {title} ---")
    
    def sample_start(self, sample_id: int, character: str, book: str):
        """Log sample processing start."""
        self._log(f"\n[SAMPLE {sample_id}] {character} ({book})")
    
    def sample_result(self, sample_id: int, data: dict):
        """
        Log sample result to JSONL file.
        
        Args:
            sample_id: Sample ID
            data: Dictionary of result data
        """
        data["sample_id"] = sample_id
        data["timestamp"] = datetime.now().isoformat()
        
        with open(self.results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")
        
        # Update stats
        self.stats["processed"] += 1
        if data.get("is_correct"):
            self.stats["correct"] += 1
    
    def bdh_decision(self, sample_id: int, verdict: str, confidence: float, 
                     neurons: list[dict], fusion: dict):
        """Log BDH fallback decision details."""
        self._log(f"  [BDH] Verdict: {verdict} (conf={confidence:.2f})")
        for n in neurons:
            self._log(f"    Neuron[{n.get('type', '?')}]: {n.get('signal', '?')} ({n.get('activation', 0):.2f})")
        self._log(f"    Fusion: E={fusion.get('excitatory', 0):.2f} I={fusion.get('inhibitory', 0):.2f}")
    
    def agent_decision(self, sample_id: int, conservative: dict, aggressive: dict, 
                       supervisor: dict = None):
        """Log agent decision details."""
        self._log(f"  Conservative: {conservative.get('verdict', '?')} ({conservative.get('confidence', 0):.2f})")
        self._log(f"  Aggressive: {aggressive.get('verdict', '?')} ({aggressive.get('confidence', 0):.2f})")
        if supervisor:
            self._log(f"  Supervisor: chose {supervisor.get('chosen', '?')}")
    
    def finalize(self, accuracy: float = None, total_cost: float = None):
        """Finalize the log with summary."""
        self.stats["ended"] = datetime.now().isoformat()
        
        self._log(f"\n{'=' * 60}")
        self._log("SUMMARY")
        self._log(f"{'=' * 60}")
        self._log(f"Processed: {self.stats['processed']}/{self.stats['total_samples']}")
        
        if accuracy is not None:
            self._log(f"Accuracy: {accuracy:.1f}%")
            self.stats["accuracy"] = accuracy
        
        if total_cost is not None:
            self._log(f"Total Cost: INR {total_cost:.2f}")
            self.stats["total_cost"] = total_cost
        
        self._log(f"Errors: {self.stats['errors']}")
        self._log(f"Log: {self.log_file}")
        self._log(f"Results: {self.results_file}")
        
        # Save summary to separate file
        summary_file = LOG_DIR / f"{self.phase_name}_{self.timestamp}_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)
        
        return self.stats


# Convenience function for terminal output
def print_header(title: str):
    """Print a section header to terminal."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_status(message: str):
    """Print a status message."""
    print(f"[INFO] {message}")


def print_result(label: str, value: Any):
    """Print a result."""
    print(f"  {label}: {value}")

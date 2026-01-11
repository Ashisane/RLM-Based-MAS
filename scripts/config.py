"""
Configuration and shared utilities for the Narrative Consistency Checker.
"""
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("CRITICAL ERROR: GEMINI_API_KEY not found in .env file. Terminating.")

# Model Configuration
ROOT_MODEL = "gemini-2.5-flash"
SUB_MODEL = "gemini-2.5-flash"  # Using flash for sub-calls to save cost
PRO_MODEL = "gemini-2.5-pro"    # For critical decisions
SUPERVISOR_MODEL = "gemini-2.5-pro"  # Supervisor uses Pro for better judgment

# Async Rate Limits (Tier 1)
FLASH_RPM = 300  # Requests per minute for Flash
PRO_RPM = 150    # Requests per minute for Pro
MAX_CONCURRENT_FLASH = 10  # Conservative to avoid hitting limits
MAX_CONCURRENT_PRO = 5

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
BOOKS_DIR = PROJECT_ROOT / "dataset" / "books"
CACHE_DIR = PROJECT_ROOT / "cache"
REPORT_DIR = PROJECT_ROOT / "agent" / "report"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True, parents=True)

# Processing Configuration
MAX_OUTPUT_CHARS = 8000
TIMEOUT_PER_CODE_BLOCK = 30
MAX_SUB_CALLS = 15
MAX_RETRIES = 3

# Cost Tracking (USD to INR conversion ~84)
USD_TO_INR = 84
COST_PER_1M_INPUT_FLASH = 0.30  # USD
COST_PER_1M_OUTPUT_FLASH = 2.50  # USD
COST_PER_1M_INPUT_PRO = 1.25    # USD
COST_PER_1M_OUTPUT_PRO = 10.00  # USD

class CostTracker:
    """Tracks API costs across the pipeline."""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.flash_input_tokens = 0
        self.flash_output_tokens = 0
        self.pro_input_tokens = 0
        self.pro_output_tokens = 0
        self.calls = []
        self._load_state()
    
    def _state_file(self):
        return CACHE_DIR / "cost_tracker.json"
    
    def _load_state(self):
        if self._state_file().exists():
            try:
                data = json.loads(self._state_file().read_text())
                self.__dict__.update(data)
            except:
                pass
    
    def _save_state(self):
        self._state_file().write_text(json.dumps({
            k: v for k, v in self.__dict__.items() if not k.startswith('_')
        }, indent=2))
    
    def add_call(self, model: str, input_tokens: int, output_tokens: int, description: str = ""):
        is_pro = "pro" in model.lower()
        
        if is_pro:
            self.pro_input_tokens += input_tokens
            self.pro_output_tokens += output_tokens
        else:
            self.flash_input_tokens += input_tokens
            self.flash_output_tokens += output_tokens
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        self.calls.append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "input": input_tokens,
            "output": output_tokens,
            "desc": description
        })
        
        self._save_state()
        
        # Check budget alert
        cost = self.get_total_cost_inr()
        if cost >= 10000:
            print(f"⚠️  BUDGET ALERT: Current cost ₹{cost:.2f} has reached ₹10,000 threshold!")
    
    def get_total_cost_inr(self) -> float:
        flash_cost = (
            (self.flash_input_tokens / 1_000_000) * COST_PER_1M_INPUT_FLASH +
            (self.flash_output_tokens / 1_000_000) * COST_PER_1M_OUTPUT_FLASH
        )
        pro_cost = (
            (self.pro_input_tokens / 1_000_000) * COST_PER_1M_INPUT_PRO +
            (self.pro_output_tokens / 1_000_000) * COST_PER_1M_OUTPUT_PRO
        )
        return (flash_cost + pro_cost) * USD_TO_INR
    
    def summary(self) -> str:
        return f"""
Cost Summary:
  Flash: {self.flash_input_tokens:,} input / {self.flash_output_tokens:,} output
  Pro:   {self.pro_input_tokens:,} input / {self.pro_output_tokens:,} output
  Total: ₹{self.get_total_cost_inr():.2f} / ₹12,000
"""

# Global cost tracker
cost_tracker = CostTracker()


def log_test_result(test_num: int, novel: str, backstory: str, 
                    correct_response: str, our_response: str,
                    model: str, confidence: float, reasoning: str):
    """Log test result in standardized format."""
    print(f"""
{'='*60}
Test {test_num}: ({novel}) 
Backstory: {backstory[:100]}...
correct response: {correct_response}
our response: {our_response}
model used: {model}
confidence: {confidence}
reasoning: {reasoning}
{'='*60}
""")

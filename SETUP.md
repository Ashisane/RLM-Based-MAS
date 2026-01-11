# Setup and Usage Guide

## Initial Setup

1. **Clone/Download the project**
   ```bash
   cd RLM-Based-MAS
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API key**
   Create a `.env` file with your Gemini API key:
   ```bash
   echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
   ```

4. **Verify dataset structure**
   ```
   dataset/
   ├── official/
   │   ├── training.csv    # 80 samples
   │   └── test.csv        # Test samples
   └── books/
       ├── In Search of the Castaways.txt
       └── The Count of Monte Cristo.txt
   ```

## Running the Test Pipeline

### Option 1: Full Run (Recommended for First Time)

```bash
python run_test.py --extract
```

This will:
1. Extract constraints from novels (if not cached)
2. Run verification on all test samples
3. Generate `results.csv` with predictions

**Expected time:** ~30-45 minutes for full test set  
**Expected cost:** ~INR 50-100

### Option 2: Quick Test (Validation)

```bash
python run_test.py --limit 5
```

Test with only 5 samples to verify everything works.

**Expected time:** ~2-3 minutes  
**Expected cost:** ~INR 5-10

### Option 3: Fresh Start

```bash
python run_test.py --clear-cache --extract
```

Clears all cached data and re-extracts everything from scratch.

### Option 4: Use Cached Constraints (Fastest)

```bash
python run_test.py
```

If constraints are already cached, skip extraction and just run verification.

**Expected time:** ~15-20 minutes  
**Expected cost:** ~INR 30-50

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--clear-cache` | Delete cached constraints before running |
| `--extract` | Force constraint extraction |
| `--limit N` | Process only first N test samples |
| `--output FILE` | Specify output CSV file (default: results.csv) |

## Output Files

### Primary Output

**`results.csv`** - Required submission format:
```csv
Story ID,Prediction,Rationale
1,1,Both agents agreed: backstory matches timeline
2,0,Supervisor chose aggressive: wrong title mentioned
```

### Secondary Outputs

- `cache/logs/test_run_*.log` - Detailed execution log
- `cache/logs/test_run_*_results.jsonl` - Full results in JSON format
- `agent/report/test_summary.md` - Statistics and summary

## Troubleshooting

### API Key Issues
```bash
# Verify .env file exists
cat .env

# Should show: GEMINI_API_KEY=...
```

### Missing Constraints
```bash
# Force re-extraction
python run_test.py --clear-cache --extract
```

### Out of Memory
```bash
# Process in smaller batches
python run_test.py --limit 20
# Then continue with remaining samples
```

### Cost Concerns
Check current cost anytime:
```python
from scripts.config import cost_tracker
print(f"Current cost: INR {cost_tracker.get_total_cost_inr():.2f}")
```

## Training (Optional)

To re-run training on the 80 training samples:

```bash
python scripts/training_pipeline.py
```

This will:
- Process all 80 training samples
- Calculate accuracy metrics
- Generate `agent/report/stage_3.md`
- **Cost:** ~INR 80-120

## File Structure

```
RLM-Based-MAS/
├── run_test.py              ← Main entry point for testing
├── results.csv              ← Generated output (submission file)
├── .env                     ← Your API key (create this)
├── README.md
├── SETUP.md                 ← This file
├── requirements.txt
├── scripts/                 ← Core pipeline code
├── dataset/                 ← Input data
├── cache/                   ← Cached constraints and logs
└── agent/report/            ← Generated reports
```

## Expected Behavior

1. **First run with --extract:**
   - Extracts constraints (~5-10 min per book)
   - Caches results for future use
   - Runs verification
   - Generates CSV

2. **Subsequent runs:**
   - Skips extraction (uses cache)
   - Only runs verification
   - Much faster (~15-20 min)

3. **With --limit N:**
   - Processes only N samples
   - Good for testing/validation
   - Proportional cost and time

## Notes

- Constraint extraction is **one-time** per book (cached)
- Each verification run costs ~INR 30-50 depending on test size
- Progress is shown with live ETA, accuracy, and cost
- All logs are saved for debugging

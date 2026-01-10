# Narrative Consistency Checker

A hybrid RLM + Multi-Agent system for detecting contradictions between character backstories and novel canon.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API key
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Run Phase 1: Extract constraints from novels
python scripts/validate_phase1.py

# 4. Run Phase 3: Full training (80 samples)
python scripts/phase3_training.py
```

## Project Structure

```
├── scripts/
│   ├── config.py              # API keys, costs, model configs
│   ├── gemini_client.py       # Gemini API wrapper (sync + async)
│   ├── data_loader.py         # Load training/test data
│   ├── phase1_extractor.py    # RLM constraint extraction
│   ├── phase2_ensemble.py     # Dual-agent ensemble verification
│   ├── phase3_training.py     # Full training pipeline
│   └── validate_*.py          # Validation scripts
├── dataset/official/          # Training and test CSVs
├── cache/                     # Extracted constraints, logs
├── agent/report/              # Stage reports (stage_1.md, stage_2.md, etc.)
└── .env                       # GEMINI_API_KEY (not committed)
```

## Methodology

### Phase 1: RLM Constraint Extraction
- Extracts timeline, relationships, and capabilities from novels
- Uses overlapping chunks for comprehensive coverage

### Phase 2: Ensemble Verification
- **Conservative Agent**: Leans toward "consistent"
- **Aggressive Agent**: Flags any potential error
- **Supervisor (Pro)**: Arbitrates disagreements

### Phase 3: Training
- Processes all 80 training samples
- Threshold tuning and error analysis
- Target: ≥65% accuracy

## Current Results

| Metric | Value |
|--------|-------|
| Accuracy | 70.0% |
| False Positive Rate | 19.6% |
| False Negative Rate | 48.3% |
| Total Cost | ~INR 94 |

## Requirements

- Python 3.10+
- Gemini API key (Tier 1 recommended)
- ~INR 100-150 budget for full training

## License

MIT
